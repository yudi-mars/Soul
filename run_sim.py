import io
from pathlib import Path

import numpy as np
import onnx
import tomli
import torch
import tqdm
import yaml

from soul import sim
from soul.model.vision import SEWResNet18
from soul.neuron import LIFNode
from soul.utils.surrogate import surrogate_map


def strip_weights_and_biases(onnx_model):
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

        # 如果长度有变化，说明删除了某些输入
        if len(cleaned_inputs) != len(node.input):
            removed_count += len(node.input) - len(cleaned_inputs)
            # 修改节点的输入列表
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


def clean_shape(onnx_model):
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


def clean_hangling(onnx_model):
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


lif_conf = yaml.safe_load(open("soul/config/neuron/lif.yaml", "r", encoding="utf-8"))
lif_conf["surrogate_function"] = surrogate_map[lif_conf["surrogate"]]

conf = {
    "num_classes": 10,
    "time_step": 4,
    "input_channels": 3,
    "input_height": 32,
    "input_width": 32,
    "hidden_dim": 1024,
    "neuron": LIFNode(lif_conf),
    "groups": 1,
    "base_width": 64,
    "connect_function": "ADD",
    "width_per_group": 64,
    "zero_init_residual": False,
}
# model = SpikingMLP(conf)
model = SEWResNet18(conf)

dummy_input = torch.randn((1, 1, 3, 32, 32))
buffer = io.BytesIO()
print(torch.onnx.is_in_onnx_export())
torch.onnx.export(
    model,
    dummy_input,
    buffer,
    export_params=False,  # Must be True to avoid naming issues
    # opset_version=11,
    do_constant_folding=False,
    input_names=["input"],
    output_names=["output"],
    verbose=False,
    # Use legacy exporter
    dynamo=False,
)
buffer.seek(0)
onnx_model = onnx.load(buffer)
print(torch.onnx.is_in_onnx_export())

onnx.save(onnx_model, "sewresnet18_cifar10.onnx")
onnx_model = strip_weights_and_biases(onnx_model)
onnx_model = clean_shape(onnx_model)
onnx_model = clean_hangling(onnx_model)
onnx.save(onnx_model, "sewresnet18_cifar10_cleaned.onnx")

# graph = export_graph(model, input_shape=(1, 3, 32, 32))

# graph = export_graph(model, input_shape=(1, 3, 32, 32), debug=False)
# print("=" * 20)
# print(graph)

# graph = nx.node_link_graph(
#     json.loads(Path("sewresnet18_cifar10_graph.json").read_text()), edges="links"
# )
# print(f"节点数: {graph.number_of_nodes()}")
# print(f"边数: {graph.number_of_edges()}")


# def find_neuron_to_neuron_paths(G):
#     """
#     寻找两个 op_type='Neuron' 结点之间的路径，
#     要求路径中间不包含其它 Neuron 结点。
#     """

#     # 1. 筛选出所有 Neuron 结点
#     neuron_nodes = [n for n, d in G.nodes(data=True) if d.get("op_type") == "Neuron"]

#     found_paths = []

#     # 2. 对每个 Neuron 结点作为起点发起搜索
#     for start_node in neuron_nodes:
#         # 使用栈进行 DFS (节点, 当前路径)
#         # 路径初始化包含起点
#         stack = [(start_node, [start_node])]

#         while stack:
#             curr_node, path = stack.pop()

#             # 获取所有邻居（如果是无向图用 G.neighbors，有向图会自动处理）
#             for neighbor in G.neighbors(curr_node):
#                 # 获取邻居的属性
#                 neighbor_type = G.nodes[neighbor].get("op_type")

#                 if neighbor_type == "Neuron":
#                     # A. 找到了终点是 Neuron
#                     # 记录路径 (起点, 终点, 完整路径)
#                     found_paths.append(
#                         {
#                             "source": start_node,
#                             "target": neighbor,
#                             "path": path + [neighbor],
#                             "intermediate_nodes": path[1:],  # 不含起点，这里还没加终点
#                         }
#                     )
#                     # 注意：这里不再将 neighbor 加入 stack，
#                     # 因为题目要求路径中间不能有 Neuron，所以不能穿过它继续找

#                 else:
#                     # B. 遇到了非 Neuron 结点（中间结点）
#                     # 只有当 neighbor 不在当前 path 中时才继续（防止环路死循环）
#                     if neighbor not in path:
#                         stack.append((neighbor, path + [neighbor]))
#     return neuron_nodes, found_paths


# def dump_linear(in_dim, out_dim):
#     src = np.arange(in_dim, dtype=np.int32)
#     src = src[np.newaxis, :]
#     src = np.repeat(src, out_dim, axis=0)
#     return src


# def fusion_layers(src1, src2):
#     if src1 is None:
#         return src2
#     if src2 is None:
#         return src1
#     src = np.array([np.unique(np.concat(src1[s])) for s in src2])
#     return src


# def dump_conv2d_img2col(layer, in_shape, out_shape):
#     ky, kx = layer.kernel_size
#     sy, sx = layer.stride
#     py, px = layer.padding

#     assert layer.in_channels == in_shape[0]
#     assert layer.out_channels == out_shape[0]
#     C, H, W = in_shape[-3:]
#     in_feats = C * H * W
#     in_idx = torch.arange(in_feats) + 1
#     in_idx = in_idx.reshape([C, H, W]).float()
#     in_idx_unfold = (
#         torch.nn.functional.unfold(
#             in_idx,
#             kernel_size=layer.kernel_size,
#             padding=layer.padding,
#             stride=layer.stride,
#         )
#         .long()
#         .numpy()
#         .astype(np.int32)
#     )
#     out_feats = np.prod(out_shape[-3:])
#     out_feats_per_ch = np.prod(out_shape[-2:])
#     assert in_idx_unfold.shape[1] == out_feats_per_ch

#     srcs = []
#     num_conns = 0
#     for j in range(out_feats_per_ch):
#         src = in_idx_unfold[:, j]
#         src = src[src > 0] - 1
#         num_conns += len(src)
#         srcs.append(src)

#     srcs = srcs * out_shape[-3]
#     num_conns = num_conns * out_shape[-3]
#     sources = np.empty(shape=(out_feats,), dtype=np.ndarray)
#     sources[:] = srcs
#     print(f"parse Conv2d done, {num_conns}")
#     return sources, np.arange(out_feats, dtype=np.int32), num_conns


# neuron_nodes, paths = find_neuron_to_neuron_paths(graph)
# for path in paths:
#     assert len(path["intermediate_nodes"]) >= 1, "路径中间节点数应至少为1"
#     if path["source"] == "sn1.neuron":
#         print(path["path"])
#         in_shape = graph.nodes[path["source"]]["attributes"]["shape"]
#         out_shape = graph.nodes[path["target"]]["attributes"]["shape"]
#         src = None
#         if "Linear" in [
#             graph.nodes[node].get("op_type") for node in path["intermediate_nodes"]
#         ]:
#             src = fusion_layers(src, dump_linear(np.prod(in_shape), np.prod(out_shape)))
#         else:
#             for node in path["intermediate_nodes"]:
#                 this_src = None
#                 if graph.nodes[node].get("op_type") == "Conv":
#                     this_src = dump_conv2d_img2col()
#                 elif graph.nodes[node].get("op_type") == "MaxPool":
#                     this_src = dump_conv2d_img2col()
#                 elif graph.nodes[node].get("op_type") == "AvgPool":
#                     this_src = dump_conv2d_img2col()
#                 src = fusion_layers(src, this_src)
#         assert len(src) == np.prod(out_shape), "源节点数应等于输出特征数"
#         num_synapses = sum(len(s) for s in src)
#         path["synapses"] = src
#         path["num_synapses"] = num_synapses


# num_neurons_layer = [np.prod(neu) for neu in neuron_nodes]
# num_neurons = np.sum(num_neurons_layer)
# neuron_start = np.cumsum([0] + num_neurons_layer[:-1])
# print(f"Total neurons: {num_neurons}")
# print(f"Total synapses: {np.sum([path['num_synapses'] for path in paths])}")

# # partition
# core_capacity = 64
# num_cores = int(np.ceil(num_neurons / core_capacity))
# print(f"Num cores: {num_cores}")
# num_neurons_core = np.ceil(num_neurons / num_cores).astype(np.int32)
# position = np.arange(num_neurons) // num_neurons_core


# def dump_core_conns(src, pre_num, post_idx, pos, num_cores, use_tqdm=False):
#     core_conns = np.zeros([pre_num, num_cores], dtype=np.bool_)
#     if use_tqdm:
#         pbar = tqdm.tqdm(total=len(src))
#     for d, ss in enumerate(src):
#         d_core = pos[post_idx + d]
#         core_conns[ss, d_core] = True
#         if use_tqdm:
#             pbar.update(1)
#     if use_tqdm:
#         pbar.close()
#     return core_conns


# core_conns = np.zeros([num_neurons, num_cores], dtype=np.uint8)
# for path in paths:
#     src = path["synapses"]
#     pre_num = num_neurons_layer[path["source"]]
#     pre_idx = neuron_start[path["source"]]
#     post_idx = neuron_start[path["target"]]
#     core_conns[pre_idx : pre_idx + pre_num] = dump_core_conns(
#         src,
#         pre_num,
#         post_idx,
#         position,
#         num_cores,
#         use_tqdm=True,
#     )

# # mapping
# mapping_l2p = np.arange(num_cores)
# mapping_p2l = np.arange(num_cores)
# for i in range(num_cores):
#     mapping_p2l[mapping_l2p[i]] = i
# phy_position = mapping_l2p[position].astype(np.uint32)
# phy_core_conns = core_conns[:, mapping_p2l].astype(np.uint8)

# # prepare spikes
# rng = np.random.default_rng(seed=42)
# p = 0.1
# spikes = (rng.random((num_neurons, conf["time_step"])) < p).astype(np.uint8)

# true_total_spikes = np.sum(np.sum(phy_core_conns, axis=1) * np.sum(spikes, axis=1))

# # simulation, using NeuSim
# sim_conf_file = Path("external/NeuSim/tests/noc_tests/configs/config.toml")
# sim_conf = tomli.loads(sim_conf_file.read_text())
# res = sim.run(
#     position=phy_position,
#     core_conns=phy_core_conns,
#     spikes=spikes,
#     packet_size=1,
#     topology_size=(2, 2),
#     num_threads=1,
# )
# # print(res.retcode)
# print(res.total_cycles)
# print(res.total_recv_flits)
# print(res.total_firing_cnt, np.sum(spikes))
# print(res.total_recv_spikes, res.total_sent_spikes, true_total_spikes)
# assert res.total_recv_spikes == true_total_spikes
