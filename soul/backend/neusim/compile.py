import copy
import io
import math
from pathlib import Path

import networkx as nx
import numpy as np
import onnx
import torch
import tqdm
from rich.progress import track
from torchinfo import summary

from soul.utils.monitor import BaseMonitor

from .arch import NeuSimArch


class CompileResult:
    onnx_model: onnx.ModelProto
    clean_onnx_model: onnx.ModelProto
    graph: nx.DiGraph
    neuron_graph: nx.DiGraph
    num_neurons: int
    num_synapses: int
    num_cores: int
    num_params: int
    position: np.ndarray
    mapping_l2p: np.ndarray
    phy_position: np.ndarray
    phy_core_conns: np.ndarray

    def save(self, file):
        file = Path(file)
        np.savez(
            file,
            num_neurons=self.num_neurons,
            num_synapses=self.num_synapses,
            num_cores=self.num_cores,
            num_params=self.num_params,
            position=self.position,
            mapping_l2p=self.mapping_l2p,
            phy_position=self.phy_position,
            phy_core_conns=self.phy_core_conns,
        )
    
    def load(self, file):
        file = Path(file)
        data = np.load(file)
        self.num_neurons = int(data["num_neurons"])
        self.num_synapses = int(data["num_synapses"])
        self.num_cores = int(data["num_cores"])
        self.num_params = int(data["num_params"])
        self.position = data["position"]
        self.mapping_l2p = data["mapping_l2p"]
        self.phy_position = data["phy_position"]
        self.phy_core_conns = data["phy_core_conns"]


def export_onnx(model, input_shape: tuple[int, ...]):
    dummy_input = torch.randn((1, 1, *input_shape))  # T, B, input_shape
    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        dummy_input,
        buffer,
        export_params=False,  # Must be True to avoid naming issues
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
        verbose=False,
        # Use legacy exporter
        dynamo=False,
    )
    buffer.seek(0)
    onnx_model = onnx.load(buffer)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    raw_onnx_model = copy.deepcopy(onnx_model)
    clean_onnx_model = clean_weights_and_biases(onnx_model)
    clean_onnx_model = clean_shape_nodes(clean_onnx_model)
    clean_onnx_model = clean_hanging_nodes(clean_onnx_model)
    clean_onnx_model = process_unsqueeze_nodes(clean_onnx_model)

    return raw_onnx_model, clean_onnx_model


def clean_weights_and_biases(onnx_model):
    graph = onnx_model.graph

    removed_count = 0

    for node in graph.node:
        cleaned_inputs = [
            input_name
            for input_name in node.input
            if "weight" not in input_name
            and "bias" not in input_name
            and "running_mean" not in input_name
            and "running_var" not in input_name
        ]

        if node.op_type in ["Conv", "Gemm"]:
            input_names = [inp.name for inp in graph.input]
            weight = graph.input[input_names.index(node.input[1])]
            shape = []
            for d in weight.type.tensor_type.shape.dim:
                shape.append(d.dim_value)
            new_attr = onnx.helper.make_attribute("weight_shape", shape)
            node.attribute.append(new_attr)

        if len(cleaned_inputs) != len(node.input):
            removed_count += len(node.input) - len(cleaned_inputs)
            del node.input[:]
            node.input.extend(cleaned_inputs)

    valid_inputs = [
        inp
        for inp in graph.input
        if "weight" not in inp.name
        and "bias" not in inp.name
        and "running_mean" not in inp.name
        and "running_var" not in inp.name
    ]
    del graph.input[:]
    graph.input.extend(valid_inputs)

    return onnx_model


def clean_shape_nodes(onnx_model):
    graph = onnx_model.graph

    for node in graph.node:
        cleaned_inputs = [
            input_name
            for input_name in node.input
            if not input_name.startswith("/Shape")
        ]

        if len(cleaned_inputs) != len(node.input):
            del node.input[:]
            node.input.extend(cleaned_inputs)

    nodes_to_keep = [n for n in graph.node if n.op_type != "Shape"]
    del graph.node[:]
    graph.node.extend(nodes_to_keep)

    return onnx_model


def clean_hanging_nodes(onnx_model):
    graph = onnx_model.graph

    while True:
        nodes_to_keep = [n for n in graph.node if len(n.input) > 0]
        if len(nodes_to_keep) == len(graph.node):
            break
        del graph.node[:]
        graph.node.extend(nodes_to_keep)
        all_outputs = []
        for node in graph.node:
            all_outputs.extend(node.output)
        all_outputs += [inp.name for inp in graph.input]

        for node in graph.node:
            cleaned_inputs = [
                input_name for input_name in node.input if input_name in all_outputs
            ]

            if len(cleaned_inputs) != len(node.input):
                del node.input[:]
                node.input.extend(cleaned_inputs)

    return onnx_model


def process_unsqueeze_nodes(onnx_model):
    graph = onnx_model.graph

    value_info_map = {vi.name: vi for vi in graph.value_info}
    for node in graph.node:
        if node.op_type == "Unsqueeze":
            shape = []
            for d in value_info_map[node.output[0]].type.tensor_type.shape.dim:
                shape.append(d.dim_value)
            new_attr = onnx.helper.make_attribute("shape", shape)
            node.attribute.append(new_attr)

    return onnx_model


def onnx_to_networkx(onnx_model):
    model = onnx_model
    graph = model.graph

    # 创建有向图
    G = nx.DiGraph()

    # 辅助字典：记录每个 Tensor 是由哪个节点产生的 (Tensor Name -> Node ID)
    # 这用于后续构建边
    tensor_producer_map = {}

    for inp in graph.input:
        # 过滤掉作为权重的输入（通常 initializer 也在 input 列表中）
        # 这里做一个简单的判断：如果它也在 initializer 中，通常视为权重而非动态输入
        # (视具体需求，这里也可以简化为全部添加)
        input_id = f"Input_{inp.name}"
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            shape.append(d.dim_value)
        # (T, B, ...) -> (...)
        shape = shape[2:]
        G.add_node(
            input_id, type="Input", op_type="Input", attributes={"spike_shape": shape}
        )
        tensor_producer_map[inp.name] = input_id

    for idx, node in enumerate(graph.node):
        # 如果节点没有名字，使用 "Op类型_索引" 作为唯一ID
        node_id = node.name if node.name else f"{node.op_type}_{idx}"

        # 将节点添加到 NetworkX，并保存 Op 类型作为属性
        attr_dict = {}
        for attr in node.attribute:
            # helper.get_attribute_value 自动识别类型并提取值
            val = onnx.helper.get_attribute_value(attr)
            # 额外处理：ONNX 的字符串通常是 bytes，为了在 networkx/可视化中易读，转为 str
            if isinstance(val, bytes):
                try:
                    val = val.decode("utf-8")
                except Exception:
                    pass
            attr_dict[attr.name] = val
            if attr.name == "spike_shape":
                # (T, B, ...) -> (...)
                attr_dict[attr.name] = val[2:]
                if node.op_type not in ["LIFNode"]:
                    raise NotImplementedError(
                        f"Only support LIFNode neuron for now, got {node.op_type}"
                    )
            elif attr.name == "shape" and node.op_type == "Unsqueeze":
                # (T, B, ...) -> (...)
                attr_dict[attr.name] = val[2:]
        G.add_node(node_id, type="Op", op_type=node.op_type, attributes=attr_dict)

        # 记录该节点产生的输出张量
        for output_tensor in node.output:
            tensor_producer_map[output_tensor] = node_id

    for outp in graph.output:
        # 过滤掉作为权重的输入（通常 initializer 也在 input 列表中）
        # 这里做一个简单的判断：如果它也在 initializer 中，通常视为权重而非动态输入
        # (视具体需求，这里也可以简化为全部添加)
        output_id = f"Output_{outp.name}"
        shape = []
        for d in outp.type.tensor_type.shape.dim:
            shape.append(d.dim_value)
        # (B, ...) -> (...)
        shape = shape[1:]
        G.add_node(
            output_id,
            type="Output",
            op_type="Output",
            attributes={"spike_shape": shape},
        )
        producer_id = tensor_producer_map.get(outp.name)
        if producer_id:
            G.add_edge(producer_id, output_id, tensor_name=outp.name)

    for idx, node in enumerate(graph.node):
        node_id = node.name if node.name else f"{node.op_type}_{idx}"

        for input_tensor in node.input:
            # 查找该输入的生产者
            producer_id = tensor_producer_map.get(input_tensor)

            if producer_id:
                # 添加有向边：生产者 -> 消费者
                # 边属性可以是张量名称
                G.add_edge(producer_id, node_id, tensor_name=input_tensor)
    return G


def find_neuron_paths(G):
    """
    寻找两个 op_type='Neuron' 结点之间的路径，
    要求路径中间不包含其它 Neuron 结点。
    """

    def node_is_neuron(node_id):
        return (
            "attributes" in G.nodes[node_id]
            and "spike_shape" in G.nodes[node_id]["attributes"]
        )

    neuron_nodes = [n for n in G.nodes() if node_is_neuron(n)]

    found_paths = []

    for start_node in neuron_nodes:
        stack = [(start_node, [start_node])]

        while stack:
            curr_node, path = stack.pop()

            for neighbor in G.neighbors(curr_node):
                if node_is_neuron(neighbor):
                    found_paths.append(
                        {
                            "source": start_node,
                            "target": neighbor,
                            "path": path + [neighbor],
                            "intermediate_nodes": path[1:],
                        }
                    )
                else:
                    if neighbor not in path:
                        stack.append((neighbor, path + [neighbor]))
    return get_neuron_graph(G, neuron_nodes, found_paths)


def get_neuron_graph(graph, neuron_nodes, found_paths):
    neuron_graph = nx.DiGraph()
    for node in neuron_nodes:
        neuron_graph.add_node(node, **graph.nodes[node])
    for path in found_paths:
        src = path["source"]
        tgt = path["target"]
        neuron_graph.add_edge(src, tgt, path=path["path"])

    neuron_graph.nodes["Input_input"]["depth"] = 0
    queue = ["Input_input"]
    while queue:
        node = queue.pop(0)
        depth = neuron_graph.nodes[node]["depth"]
        for succ in neuron_graph.successors(node):
            this_depth = depth + 1
            if (
                "depth" not in neuron_graph.nodes[succ]
                or neuron_graph.nodes[succ]["depth"] < this_depth
            ):
                neuron_graph.nodes[succ]["depth"] = this_depth
                queue.append(succ)
    return neuron_graph


def parse_neuron_graph(graph, neuron_graph, is_vgg=False):
    for edge in track(
        neuron_graph.edges(data=True), description="Parsing neuron graph"
    ):
        source, target, attrs = edge
        path = attrs["path"]
        assert len(path) > 2, "路径中间节点数应至少为1"
        source_node = neuron_graph.nodes[source]
        target_node = neuron_graph.nodes[target]
        in_shape = source_node["attributes"]["spike_shape"]
        out_shape = target_node["attributes"]["spike_shape"]
        path_ops = [graph.nodes[n].get("op_type") for n in path[1:-1]]
        src = None
        if "Gemm" in path_ops:
            src = fusion_layers(src, dump_linear(np.prod(in_shape), np.prod(out_shape)))
        else:
            tmp_shape = in_shape
            for node in path[1:-1]:
                this_src = None
                op_attr = graph.nodes[node]["attributes"]
                if graph.nodes[node]["op_type"] == "Conv":
                    this_src, tmp_shape = dump_conv(tmp_shape, op_attr)
                elif graph.nodes[node]["op_type"] in ["MaxPool", "AvgPool"]:
                    this_src, tmp_shape = dump_pool(tmp_shape, op_attr)
                elif graph.nodes[node]["op_type"] in [
                    "GlobalAveragePool",
                    "GlobalMaxPool",
                ]:
                    this_src, tmp_shape = dump_globalpool(tmp_shape)
                elif graph.nodes[node]["op_type"] in ["Resize"]:
                    assert is_vgg, "Only VGG model use Resize for avgpool"
                    this_src, tmp_shape = dump_avgpool(tmp_shape, (7, 7))
                elif graph.nodes[node]["op_type"] == "Unsqueeze":
                    tmp_shape = op_attr["shape"]
                elif graph.nodes[node]["op_type"] == "Transpose":
                    this_src, tmp_shape = dump_trasnpose(tmp_shape, op_attr["perm"])
                # print("fusion_layers", f"{source} -> {target}", tmp_shape)
                src = fusion_layers(src, this_src)
                # print("fusion_layers done")
        if len(src) != np.prod(out_shape):
            raise ValueError(
                f"{source} -> {target}: {path} \n"
                f"{in_shape} -> {out_shape}: {len(src)} != {np.prod(out_shape)}"
            )
        num_synapses = sum(len(s) for s in src)
        attrs["synapses"] = src
        attrs["num_synapses"] = num_synapses
        attrs["delay"] = target_node["depth"] - source_node["depth"]
        assert attrs["delay"] > 0, "延迟应大于0"

    return neuron_graph


def fusion_layers(src1, src2):
    if src1 is None:
        return src2
    if src2 is None:
        return src1
    srcs = [np.unique(np.concat(src1[s])) for s in src2]
    del src2

    try:
        srcs = np.array(srcs, dtype=np.int32)
    except Exception:
        srcs = np.array(srcs, dtype=np.ndarray)
    return srcs


def dump_linear(in_dim, out_dim):
    src = np.arange(in_dim, dtype=np.int32)
    src = src[np.newaxis, :]
    srcs = np.repeat(src, out_dim, axis=0)
    for src in srcs:
        assert src.dtype == np.int32
    return srcs


def dump_trasnpose(in_shape, perm):
    assert len(in_shape) == len(perm) - 2
    assert perm[0] == 0 and perm[1] == 1
    perm = [p - 2 for p in perm[2:]]
    idx = np.arange(np.prod(in_shape)).reshape(in_shape)
    out_idx = np.transpose(idx, axes=perm)
    out_shape = out_idx.shape

    srcs = out_idx.flatten()[:, np.newaxis]

    return srcs, out_shape


def dump_conv(in_shape, op_attr):
    if len(in_shape) == 2:
        assert len(op_attr["kernel_shape"]) == 1
        return dump_conv1d(in_shape, op_attr)
    elif len(in_shape) == 3:
        assert len(op_attr["kernel_shape"]) == 2
        return dump_conv2d(in_shape, op_attr)
    else:
        raise NotImplementedError(
            f"Only support 1D and 2D conv, got input shape: {in_shape}"
        )


def dump_conv2d(in_shape, op_attr):
    dummy_input = torch.randn((1, *in_shape))  # B, C, H, W
    in_channels = in_shape[0]
    padding = op_attr["pads"]
    assert padding[0] == padding[2] and padding[1] == padding[3]
    padding = (padding[0], padding[1])
    if "weight_shape" in op_attr:
        assert in_channels == op_attr["weight_shape"][1]
        out_channels = op_attr["weight_shape"][0]
        layer = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=op_attr["kernel_shape"],
            stride=op_attr["strides"],
            padding=padding,
            dilation=op_attr["dilations"],
            groups=op_attr["group"],
            bias=False,
        )
        if op_attr["group"] != 1:
            # TODO: handle depthwise conv
            raise NotImplementedError("group conv is not supported yet")
    else:
        ceil_mode = op_attr["ceil_mode"] == 1
        layer = torch.nn.MaxPool2d(
            kernel_size=op_attr["kernel_shape"],
            stride=op_attr["strides"],
            padding=padding,
            dilation=op_attr["dilations"],
            ceil_mode=ceil_mode,
        )
        if ceil_mode:
            # TODO: handle ceil_mode
            raise NotImplementedError("ceil_mode is not supported yet")
    dummpy_output = layer(dummy_input)
    out_shape = dummpy_output.shape[1:]  # C, H, W

    C, H, W = in_shape
    in_feats = C * H * W
    in_idx = torch.arange(in_feats) + 1
    in_idx = in_idx.reshape([C, H, W]).float()
    in_idx_unfold = (
        torch.nn.functional.unfold(
            in_idx,
            kernel_size=op_attr["kernel_shape"],
            padding=padding,
            stride=op_attr["strides"],
            dilation=op_attr["dilations"],
        )
        .long()
        .numpy()
        .astype(np.int32)
    )
    out_feats_per_ch = np.prod(out_shape[-2:])
    assert in_idx_unfold.shape[1] == out_feats_per_ch

    srcs = []
    num_conns = 0
    for j in range(out_feats_per_ch):
        src = in_idx_unfold[:, j]
        src = src[src > 0] - 1
        num_conns += len(src)
        srcs.append(src)

    srcs = srcs * out_shape[0]
    try:
        srcs = np.array(srcs, dtype=np.int32)
    except Exception:
        srcs = np.array(srcs, dtype=np.ndarray)

    for src in srcs:
        assert src.dtype == np.int32

    return srcs, out_shape


def dump_conv1d(in_shape, op_attr):
    in_channels = in_shape[0]
    padding = op_attr["pads"]
    assert padding[0] == padding[1]
    padding = padding[0]
    if "weight_shape" in op_attr:
        assert in_channels == op_attr["weight_shape"][1]
        out_channels = op_attr["weight_shape"][0]
        layer = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=op_attr["kernel_shape"],
            stride=op_attr["strides"],
            padding=padding,
            dilation=op_attr["dilations"],
            groups=op_attr["group"],
            bias=False,
        )
        if op_attr["group"] != 1:
            # TODO: handle depthwise conv
            raise NotImplementedError("group conv is not supported yet")
    else:
        ceil_mode = op_attr["ceil_mode"] == 1
        layer = torch.nn.MaxPool1d(
            kernel_size=op_attr["kernel_shape"],
            stride=op_attr["strides"],
            padding=padding,
            dilation=op_attr["dilations"],
            ceil_mode=ceil_mode,
        )
        if ceil_mode:
            # TODO: handle ceil_mode
            raise NotImplementedError("ceil_mode is not supported yet")
    dummy_input = torch.randn((1, *in_shape))  # B, C, W
    dummpy_output = layer(dummy_input)
    out_shape = dummpy_output.shape[1:]  # C, W

    C, W = in_shape
    in_feats = C * W
    in_idx = torch.arange(in_feats) + 1
    in_idx = in_idx.reshape([C, 1, W]).float()
    in_idx_unfold = (
        torch.nn.functional.unfold(
            in_idx,
            kernel_size=(1, op_attr["kernel_shape"][0]),
            padding=(0, padding),
            stride=(1, op_attr["strides"][0]),
            dilation=(1, op_attr["dilations"][0]),
        )
        .long()
        .numpy()
        .astype(np.int32)
    )
    out_feats_per_ch = np.prod(out_shape[1:])
    assert in_idx_unfold.shape[1] == out_feats_per_ch

    srcs = []
    num_conns = 0
    for j in range(out_feats_per_ch):
        src = in_idx_unfold[:, j]
        src = src[src > 0] - 1
        num_conns += len(src)
        srcs.append(src)

    srcs = srcs * out_shape[0]
    try:
        srcs = np.array(srcs, dtype=np.int32)
    except Exception:
        srcs = np.array(srcs, dtype=np.ndarray)

    for src in srcs:
        assert src.dtype == np.int32

    return srcs, out_shape


def dump_pool(in_shape, op_attr):
    if len(in_shape) == 2:
        assert len(op_attr["kernel_shape"]) == 1
        return dump_pool1d(in_shape, op_attr)
    elif len(in_shape) == 3:
        assert len(op_attr["kernel_shape"]) == 2
        return dump_pool2d(in_shape, op_attr)
    else:
        raise NotImplementedError(
            f"Only support 1D and 2D pool, got input shape: {in_shape}"
        )


def dump_pool2d(in_shape, op_attr):
    dummy_input = torch.randn((1, *in_shape))  # B, C, H, W
    padding = op_attr["pads"]
    assert padding[0] == padding[2] and padding[1] == padding[3]
    padding = (padding[0], padding[1])

    ceil_mode = op_attr["ceil_mode"] == 1
    layer = torch.nn.MaxPool2d(
        kernel_size=op_attr["kernel_shape"],
        stride=op_attr["strides"],
        padding=padding,
        dilation=op_attr["dilations"],
        ceil_mode=ceil_mode,
    )
    if ceil_mode:
        # TODO: handle ceil_mode
        raise NotImplementedError("ceil_mode is not supported yet")
    dummpy_output = layer(dummy_input)
    out_shape = dummpy_output.shape[1:]  # C, H, W

    C, H, W = in_shape
    in_feats = 1 * H * W
    in_feats_per_ch = H * W
    in_idx = torch.arange(in_feats) + 1
    in_idx = in_idx.reshape([1, H, W]).float()
    in_idx_unfold = (
        torch.nn.functional.unfold(
            in_idx,
            kernel_size=op_attr["kernel_shape"],
            padding=padding,
            stride=op_attr["strides"],
            dilation=op_attr["dilations"],
        )
        .long()
        .numpy()
        .astype(np.int32)
    )
    out_feats_per_ch = np.prod(out_shape[-2:])
    assert in_idx_unfold.shape[1] == out_feats_per_ch

    srcs = []
    num_conns = 0
    for i in range(C):
        for j in range(out_feats_per_ch):
            src = in_idx_unfold[:, j]
            src = src[src > 0] - 1 + i * in_feats_per_ch
            num_conns += len(src)
            srcs.append(src)

    try:
        srcs = np.array(srcs, dtype=np.int32)
    except Exception:
        srcs = np.array(srcs, dtype=np.ndarray)

    for src in srcs:
        assert src.dtype == np.int32
    return srcs, out_shape


def dump_pool1d(in_shape, op_attr):
    padding = op_attr["pads"]
    assert padding[0] == padding[1]
    padding = padding[0]

    ceil_mode = op_attr["ceil_mode"] == 1
    layer = torch.nn.MaxPool1d(
        kernel_size=op_attr["kernel_shape"][0],
        stride=op_attr["strides"][0],
        padding=padding,
        dilation=op_attr["dilations"][0],
        ceil_mode=ceil_mode,
    )
    if ceil_mode:
        # TODO: handle ceil_mode
        raise NotImplementedError("ceil_mode is not supported yet")
    dummy_input = torch.randn((1, *in_shape))  # B, C, W
    dummpy_output = layer(dummy_input)
    out_shape = dummpy_output.shape[1:]  # C, W

    C, W = in_shape
    in_feats = 1 * 1 * W
    in_feats_per_ch = 1 * W
    in_idx = torch.arange(in_feats) + 1
    in_idx = in_idx.reshape([1, 1, W]).float()
    in_idx_unfold = (
        torch.nn.functional.unfold(
            in_idx,
            kernel_size=(1, op_attr["kernel_shape"][0]),
            padding=(0, padding),
            stride=(1, op_attr["strides"][0]),
            dilation=(1, op_attr["dilations"][0]),
        )
        .long()
        .numpy()
        .astype(np.int32)
    )
    out_feats_per_ch = np.prod(out_shape[1:])
    assert in_idx_unfold.shape[1] == out_feats_per_ch

    srcs = []
    num_conns = 0
    for i in range(C):
        for j in range(out_feats_per_ch):
            src = in_idx_unfold[:, j]
            src = src[src > 0] - 1 + i * in_feats_per_ch
            num_conns += len(src)
            srcs.append(src)

    try:
        srcs = np.array(srcs, dtype=np.int32)
    except Exception:
        srcs = np.array(srcs, dtype=np.ndarray)

    for src in srcs:
        assert src.dtype == np.int32
    return srcs, out_shape


def dump_globalpool(in_shape):
    if len(in_shape) == 2:
        in_shape = (in_shape[0], 1, in_shape[1])
        srcs, out_shape = dump_globalpool2d(in_shape)
        return srcs, (out_shape[0], out_shape[2])
    elif len(in_shape) == 3:
        return dump_globalpool2d(in_shape)
    else:
        raise NotImplementedError(
            f"Only support 1D and 2D globalpool, got input shape: {in_shape}"
        )


def dump_globalpool2d(in_shape):
    C, H, W = in_shape
    out_shape = (C, 1, 1)
    srcs = []
    for i in range(C):
        src = np.arange(H * W, dtype=np.int32) + i * H * W
        srcs.append(src)
    srcs = np.array(srcs, dtype=np.ndarray)

    try:
        srcs = np.array(srcs, dtype=np.int32)
    except Exception:
        srcs = np.array(srcs, dtype=np.ndarray)

    for src in srcs:
        assert src.dtype == np.int32
    return srcs, out_shape


def dump_avgpool(in_shape, out_shape):
    if len(in_shape) == 2:
        in_shape = (in_shape[0], 1, in_shape[1])
        out_shape = (in_shape[0], 1, out_shape[1])
        srcs, out_shape = dump_avgpool2d(in_shape, out_shape)
        return srcs, (out_shape[0], out_shape[2])
    elif len(in_shape) == 3:
        assert len(out_shape) == 2
        return dump_avgpool2d(in_shape, (in_shape[0], *out_shape))
    else:
        raise NotImplementedError(
            f"Only support 1D and 2D avgpool, got input shape: {in_shape}"
        )


def dump_avgpool2d(in_shape, out_shape):
    iC, iH, iW = in_shape
    oC, oH, oW = out_shape
    # in SpikingVGG, output_size = (max(7, H), max(7, W))
    oH = max(oH, iH)
    oW = max(oW, iW)
    out_shape = (iC, oH, oW)
    assert iC == oC
    srcs = []
    iidx = np.arange(iH * iW, dtype=np.int32).reshape(iH, iW)
    for i in range(iC):
        for oy in range(oH):
            for ox in range(oW):
                sy = math.floor(oy * (iH / oH))
                ey = math.ceil((oy + 1) * (iH / oH))
                sx = math.floor(ox * (iW / oW))
                ex = math.ceil((ox + 1) * (iW / oW))
                src = iidx[sy:ey, sx:ex].flatten() + i * iH * iW
                srcs.append(src)
    srcs = np.array(srcs, dtype=np.ndarray)

    try:
        srcs = np.array(srcs, dtype=np.int32)
    except Exception:
        srcs = np.array(srcs, dtype=np.ndarray)

    for src in srcs:
        assert src.dtype == np.int32
    return srcs, out_shape


def dump_core_conns(src, pre_num, post_idx, pos, num_cores, use_tqdm=False):
    core_conns = np.zeros([pre_num, num_cores], dtype=np.bool_)
    if use_tqdm:
        pbar = tqdm.tqdm(total=len(src))
    for d, ss in enumerate(src):
        d_core = pos[post_idx + d]
        core_conns[ss, d_core] = True
        if use_tqdm:
            pbar.update(1)
    if use_tqdm:
        pbar.close()
    return core_conns


def partition(neuron_graph, core_capacity=4096):
    neuron_names = [n for n in neuron_graph.nodes()]
    neuron_shapes = [
        neuron_graph.nodes[n]["attributes"]["spike_shape"] for n in neuron_graph.nodes()
    ]
    num_neurons_layer = [np.prod(neu) for neu in neuron_shapes]
    num_neurons = np.sum(num_neurons_layer)
    neuron_start = np.cumsum([0] + num_neurons_layer[:-1])
    num_synapses = np.sum(
        [neuron_graph.edges[path]["num_synapses"] for path in neuron_graph.edges()]
    )
    # print(f"Total neurons: {num_neurons}")
    # print(f"Total synapses: {num_synapses}")

    num_cores = int(np.ceil(num_neurons / core_capacity))
    # print(f"Num cores: {num_cores}")
    num_neurons_core = np.ceil(num_neurons / num_cores).astype(np.int32)
    position = np.arange(num_neurons) // num_neurons_core

    core_conns = np.zeros([num_neurons, num_cores], dtype=np.uint8)
    for edge in neuron_graph.edges(data=True):
        source, target, attrs = edge
        source = neuron_names.index(source)
        target = neuron_names.index(target)
        src = attrs["synapses"]
        pre_num = num_neurons_layer[source]
        pre_idx = neuron_start[source]
        post_idx = neuron_start[target]
        core_conns[pre_idx : pre_idx + pre_num] = dump_core_conns(
            src,
            pre_num,
            post_idx,
            position,
            num_cores,
        )

    return (num_neurons, num_synapses, num_cores), position, core_conns


def mapping(num_cores, position, core_conns):
    mapping_l2p = np.arange(num_cores)
    mapping_p2l = np.arange(num_cores)
    for i in range(num_cores):
        mapping_p2l[mapping_l2p[i]] = i
    phy_position = mapping_l2p[position].astype(np.uint32)
    phy_core_conns = core_conns[:, mapping_p2l].astype(np.uint8)
    return mapping_l2p, phy_position, phy_core_conns


def compile(
    model,
    input_shape: tuple[int, ...],
    arch=NeuSimArch("darwin3"),
) -> CompileResult:
    res = CompileResult()
    model_stats = summary(
        copy.deepcopy(model), input_size=(1, 1, *input_shape), verbose=0
    )
    res.num_params = model_stats.total_params

    onnx_model, clean_onnx_model = export_onnx(
        model,
        input_shape,
    )
    res.onnx_model = onnx_model
    res.clean_onnx_model = clean_onnx_model

    graph = onnx_to_networkx(clean_onnx_model)
    res.graph = graph

    neuron_graph = find_neuron_paths(graph)
    res.neuron_graph = copy.deepcopy(neuron_graph)

    is_vgg = type(model).__name__ == "VGG"
    neuron_graph = parse_neuron_graph(graph, neuron_graph, is_vgg=is_vgg)
    (num_neurons, num_synapses, num_cores), position, core_conns = partition(
        neuron_graph, core_capacity=arch.neurons_per_core
    )
    res.num_neurons = num_neurons
    res.num_synapses = num_synapses
    res.num_cores = num_cores
    res.position = position

    mapping_l2p, phy_position, phy_core_conns = mapping(num_cores, position, core_conns)
    res.mapping_l2p = mapping_l2p
    res.phy_position = phy_position
    res.phy_core_conns = phy_core_conns

    return res


def convert_spikes(compile_res: CompileResult, monitor: BaseMonitor) -> np.ndarray:
    names = compile_res.neuron_graph.nodes()
    spikes_to_sim = []
    for name in names:
        if name == "Input_input":
            T = monitor.input_record.shape[0]
            spikes_to_sim.append(monitor.input_record.flatten(2).detach().cpu().numpy())
        elif name == "Output_output":
            B, num_classes = monitor.output_record.shape
            spikes_to_sim.append(np.zeros((T, B, num_classes), dtype=np.uint8))
        else:
            assert name[0] == "/"
            segs = name[1:].replace("/", ".").split(".")[:-1]
            segs_valid = []
            for i, seg in enumerate(segs):
                if i == len(segs) - 1 or seg != segs[i + 1]:
                    segs_valid.append(seg)
            name2 = ".".join(segs_valid)
            recs = monitor[name2]
            if len(recs) != 1:
                raise ValueError(f"Expected one record for {name2}, got {len(recs)}")
            spikes_to_sim.append(recs[0].flatten(2).detach().cpu().numpy())
    return (
        np.concatenate(spikes_to_sim, axis=-1).astype(np.uint8).transpose(1, 2, 0)
    )  # B, N, T
