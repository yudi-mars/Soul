import onnx
from onnx import shape_inference
import networkx as nx
from networkx.readwrite import json_graph
import json
import os
import argparse
import torch
import re

# Try to import matplotlib for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for saving figures
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, visualization will be skipped")

NEURON_NODE_REGEX = re.compile("sn")

def post_process_state_keys(state_keys):
    post_state_keys = dict()
    for state_key in state_keys:
        prefix = ".".join(state_key.split(".")[:-1])
        type = state_key.split(".")[-1]
        if prefix not in post_state_keys:
            post_state_keys[prefix] = []
        post_state_keys[prefix].append(type)
    norm_keys = []
    for key,values in post_state_keys.items():
        if "running_mean" in values or "running_var" in values or "num_batches_tracked" in values:
            norm_keys.append(key)
    for key in norm_keys:
        del post_state_keys[key]
    return list(post_state_keys.keys())


def should_keep_node(node_name, post_state_keys):
    """
    Check if a node should be kept based on post_state_keys.
    Returns True if any key in post_state_keys is contained in node_name.
    """
    for key in post_state_keys:
        if key in node_name or "MaxPool" in node_name or "AveragePool" in node_name or "Add" in node_name:
            return True
    return False


def filter_nodes(graph, post_state_keys):
    """
    Filter nodes from the ONNX graph:
    1. Keep nodes related to post_state_keys
    2. Keep nodes matching NEURON_NODE_REGEX (only the last one)
    
    Returns a list of filtered nodes.
    """
    filtered_nodes = []
    neuron_node = None  # Store the last neuron node
    
    for node in graph.node:
        node_name = ".".join(node.name.split('/')[1:])
        
        # Check if node matches NEURON_NODE_REGEX
        if NEURON_NODE_REGEX.search(node_name):
            # Keep track of the last neuron node
            neuron_node = node
            continue
        
        if neuron_node is not None:
            filtered_nodes.append(neuron_node)
        # neuron_node = None
        # Check if node is related to post_state_keys
        if should_keep_node(node_name, post_state_keys):
            filtered_nodes.append(node)
    
    # Add the last neuron node if it exists
    if neuron_node is not None:
        filtered_nodes.append(neuron_node)
    
    return filtered_nodes


def get_constant_value(graph, tensor_name):
    """
    Get the value from a Constant node that produces the given tensor.
    
    Args:
        graph: ONNX graph
        tensor_name: Name of the tensor (output of a Constant node)
    
    Returns:
        The constant value, or None if not found
    """
    # Find the Constant node that produces this tensor
    for node in graph.node:
        if node.op_type == "Constant" and tensor_name in node.output:
            # Get the value from the Constant node
            for attr in node.attribute:
                if attr.name == "value":
                    # value is a TensorProto
                    try:
                        from onnx import numpy_helper
                        tensor = attr.t
                        value_array = numpy_helper.to_array(tensor)
                        # Return the first element if it's an array
                        if value_array.size > 0:
                            return int(value_array.flat[0])
                    except:
                        pass
    
    # Also check in initializers
    for init in graph.initializer:
        if init.name == tensor_name:
            try:
                from onnx import numpy_helper
                value_array = numpy_helper.to_array(init)
                if value_array.size > 0:
                    return int(value_array.flat[0])
            except:
                pass
    
    return None


def should_skip_gather_node(node, graph):
    """
    Check if a Gather node should be skipped based on its indices input.
    Returns True if the node is a Gather with indices <= 1.
    
    Exception: Gather nodes within sn layers (matching \.sn\d*\.) are never skipped,
    as they are part of the neuron structure.
    
    For Gather nodes, indices is typically the second input (input[1]).
    This input usually comes from a Constant node.
    """
    if node.op_type != "Gather":
        return False
    
    # Get node name
    node_name = ".".join(node.name.split('/')[1:])
    
    # Don't skip Gather nodes that are part of sn layers
    if NEURON_NODE_REGEX.search(node_name):
        return False
    
    # Gather node typically has:
    # input[0]: data
    # input[1]: indices
    if len(node.input) < 2:
        return False
    
    indices_input = node.input[1]
    
    # Get the constant value
    indices_value = get_constant_value(graph, indices_input)
    
    if indices_value is not None and indices_value <= 1:
        return True
    
    return False


def build_full_graph(graph):
    """
    Build a complete graph from the ONNX model including all nodes.
    Filters out Gather nodes with indices <= 1.
    Returns a NetworkX DiGraph with all nodes and edges.
    """
    full_graph = nx.DiGraph()
    
    # Get all initializer names (weights/biases)
    initializer_names = set(init.name for init in graph.initializer)
    
    # Add actual input nodes (exclude initializers)
    for input_tensor in graph.input:
        if input_tensor.name not in initializer_names:
            full_graph.add_node(input_tensor.name, op_type="Input", node_type="input")
    
    # First pass: identify nodes to skip
    skipped_nodes = set()
    node_outputs_map = {}  # Maps output_name -> node
    
    for node in graph.node:
        node_name = ".".join(node.name.split('/')[1:])
        
        # Check if this Gather node should be skipped
        if node.op_type == "Gather":
            # Check if it's in an sn layer
            is_sn_node = NEURON_NODE_REGEX.search(node_name) is not None
            
            # Get indices value
            indices_value = None
            if len(node.input) >= 2:
                indices_value = get_constant_value(graph, node.input[1])
            
            if is_sn_node:
                # Preserve Gather nodes in sn layers
                print(f"  Keeping Gather in sn layer: {node_name} (indices={indices_value})")
            elif should_skip_gather_node(node, graph):
                # Skip other Gather nodes with indices <= 1
                skipped_nodes.add(node_name)
                print(f"  Skipping Gather node: {node_name} (indices={indices_value})")
        
        # Store output mappings for all nodes
        for output in node.output:
            node_outputs_map[output] = node_name
    
    # Second pass: build graph with direct connections bypassing skipped nodes
    output_to_node = {}  # Maps output_name -> final_node (after bypassing)
    
    for node in graph.node:
        node_name = ".".join(node.name.split('/')[1:])
        
        # Skip filtered Gather nodes
        if node_name in skipped_nodes:
            # For skipped nodes, map their outputs to their inputs
            # This creates a pass-through connection
            for output in node.output:
                for input_name in node.input:
                    if input_name in output_to_node:
                        output_to_node[output] = output_to_node[input_name]
                    elif input_name in node_outputs_map and node_outputs_map[input_name] not in skipped_nodes:
                        output_to_node[output] = node_outputs_map[input_name]
                    elif input_name in full_graph.nodes:  # Is an input node
                        output_to_node[output] = input_name
            continue
        
        # Add this node to the graph
        full_graph.add_node(node_name, op_type=node.op_type, node_type="operation")
        
        # Map outputs to this node
        for output in node.output:
            output_to_node[output] = node_name
        
        # Add edges from inputs to this node
        for input_name in node.input:
            # Skip initializers (weights/biases)
            if input_name in initializer_names:
                continue
            
            # Resolve the actual source node (may bypass skipped nodes)
            source_node = None
            
            if input_name in output_to_node:
                source_node = output_to_node[input_name]
            elif input_name in full_graph.nodes:  # Is an input node
                source_node = input_name
            
            # Add edge if we found a valid source
            if source_node and source_node != node_name:
                full_graph.add_edge(source_node, node_name)
    
    if skipped_nodes:
        print(f"  Total skipped Gather nodes: {len(skipped_nodes)}")
    
    return full_graph


def find_reachable_predecessors(full_graph, node, kept_nodes):
    """
    Find all kept predecessor nodes that can reach the given node.
    This traverses backwards through the graph, skipping deleted nodes.
    """
    predecessors = set()
    visited = set()
    queue = [node]
    
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        
        # Get all predecessor nodes
        for pred in full_graph.predecessors(current):
            if pred in kept_nodes and pred != node:
                # Found a kept predecessor
                predecessors.add(pred)
            else:
                # Continue searching backwards through deleted nodes
                queue.append(pred)
    
    return predecessors


def is_truly_transitive(full_graph, pred_a, pred_b, target_node):
    """
    Check if pred_a is truly transitive through pred_b to reach target_node.
    
    A predecessor A is truly transitive through B if:
    - ALL paths from A to target_node MUST go through B
    - In other words, removing B would disconnect A from target_node
    
    This distinguishes:
    - Transitive: A -> B -> D (only one path, A is redundant)
    - Parallel paths: A -> D and B -> D (two independent paths, both needed)
    
    Args:
        full_graph: The complete graph
        pred_a: First predecessor to check
        pred_b: Second predecessor (potential intermediate)
        target_node: The target node
    
    Returns:
        True if pred_a is transitive through pred_b, False otherwise
    """
    # If pred_a cannot reach pred_b, they are on parallel paths
    try:
        if not nx.has_path(full_graph, pred_a, pred_b):
            return False
    except:
        return False
    
    # Create a temporary graph without pred_b
    temp_graph = full_graph.copy()
    try:
        temp_graph.remove_node(pred_b)
    except:
        return False
    
    # If pred_a can still reach target_node without pred_b,
    # then there's a parallel path (e.g., residual connection)
    try:
        if nx.has_path(temp_graph, pred_a, target_node):
            return False  # Parallel path exists, keep both
        else:
            return True   # All paths go through pred_b, pred_a is transitive
    except:
        return False


def filter_transitive_predecessors(full_graph, predecessors, node):
    """
    Filter out transitive predecessors while preserving parallel paths.
    
    This works for both:
    - Sequential networks: A -> B -> C (remove transitive A -> C)
    - Residual networks: A -> C and A -> B -> C (keep both paths)
    
    Args:
        full_graph: The complete graph
        predecessors: Set of all reachable predecessors
        node: The target node
    
    Returns:
        Set of direct predecessors (with parallel paths preserved)
    """
    if len(predecessors) <= 1:
        return predecessors
    
    # Convert to list for iteration
    pred_list = list(predecessors)
    direct_predecessors = set(predecessors)
    
    # For each predecessor, check if it's truly transitive through another
    for i, pred_a in enumerate(pred_list):
        if pred_a not in direct_predecessors:
            continue  # Already removed
        
        for j, pred_b in enumerate(pred_list):
            if i != j and pred_a in direct_predecessors:
                # Check if ALL paths from pred_a to node go through pred_b
                if is_truly_transitive(full_graph, pred_a, pred_b, node):
                    # pred_a is transitive through pred_b, remove it
                    direct_predecessors.discard(pred_a)
                    break  # No need to check other pred_b
    
    return direct_predecessors


def extract_node_attributes(node):
    """
    Extract important attributes from an ONNX node.
    
    Returns a dictionary of attribute_name -> value.
    """
    attributes = {}
    
    for attr in node.attribute:
        attr_value = get_attribute_value(attr)
        if attr_value is not None:
            # Convert bytes to string for JSON serialization
            if isinstance(attr_value, bytes):
                try:
                    attr_value = attr_value.decode('utf-8')
                except:
                    attr_value = str(attr_value)
            # Convert numpy arrays/lists to regular Python lists
            elif hasattr(attr_value, 'tolist'):
                attr_value = attr_value.tolist()
            
            attributes[attr.name] = attr_value
    
    return attributes


def onnx_decode(graph,state_keys):
    # preprocess-batchnorm keys:
    post_state_keys = post_process_state_keys(state_keys)
    
    # Build complete graph with all nodes
    full_graph = build_full_graph(graph)
    # Filter nodes to keep only relevant ones
    filtered_nodes = filter_nodes(graph, post_state_keys)
    
    # Create mapping: node_name -> original_node for attribute extraction
    node_map = {}
    for node in filtered_nodes:
        node_name = ".".join(node.name.split('/')[1:])
        node_map[node_name] = node
    
    # Create set of kept node names for fast lookup
    kept_node_names = set()
    for node in filtered_nodes:
        node_name = ".".join(node.name.split('/')[1:])
        kept_node_names.add(node_name)
    
    # Add input nodes that are actually used
    initializer_names = set(init.name for init in graph.initializer)
    for input_tensor in graph.input:
        if input_tensor.name not in initializer_names:
            kept_node_names.add(input_tensor.name)
    
    # Build the simplified neural graph
    neuro_graph = nx.DiGraph()
    
    # Create mapping for renamed nodes (original_name -> new_name)
    node_name_mapping = {}
    
    # Add input nodes with shape information
    for input_tensor in graph.input:
        if input_tensor.name not in initializer_names:
            # Extract shape information if available
            shape_info = []
            if input_tensor.type.HasField('tensor_type'):
                tensor_type = input_tensor.type.tensor_type
                if tensor_type.HasField('shape'):
                    for dim in tensor_type.shape.dim:
                        if dim.HasField('dim_value'):
                            shape_info.append(int(dim.dim_value))
                        else:
                            shape_info.append(-1)  # Dynamic dimension
            
            neuro_graph.add_node(input_tensor.name, 
                               op_type="Input", 
                               node_type="input",
                               shape=shape_info if shape_info else None)
            node_name_mapping[input_tensor.name] = input_tensor.name
    
    # Add filtered operation nodes with attributes
    for node in filtered_nodes:
        node_name = ".".join(node.name.split('/')[1:])
        
        # Extract node attributes
        node_attributes = extract_node_attributes(node)
        
        # Check if this is a neuron (sn) node
        if NEURON_NODE_REGEX.search(node_name):
            # Replace the last field with "neuron"
            # e.g., "pre_logits.sn2.Concat" -> "pre_logits.sn2.neuron"
            parts = node_name.split('.')
            parts[-1] = 'neuron'
            neuron_name = '.'.join(parts)
            
            # Add node with custom Neuron type and attributes
            neuro_graph.add_node(neuron_name, 
                               op_type="Neuron", 
                               node_type="operation", 
                               original_name=node_name, 
                               original_op_type=node.op_type,
                               attributes=node_attributes)
            
            # Store the mapping for edge construction
            node_name_mapping[node_name] = neuron_name
            print(f"  Renamed neuron node: {node_name} -> {neuron_name} (Neuron)")
        else:
            # Regular node with attributes
            neuro_graph.add_node(node_name, 
                               op_type=node.op_type, 
                               node_type="operation",
                               attributes=node_attributes)
            node_name_mapping[node_name] = node_name
    
    # Build edges: for each kept node, find its kept predecessors
    print("\n=== Building Edges (Smart Filtering: Removes Transitive, Keeps Parallel Paths) ===")
    for node in filtered_nodes:
        node_name = ".".join(node.name.split('/')[1:])
        
        # Get the mapped name (may be renamed for neuron nodes)
        target_name = node_name_mapping.get(node_name, node_name)
        
        # Find all kept predecessors (may skip intermediate deleted nodes)
        all_predecessors = find_reachable_predecessors(full_graph, node_name, kept_node_names)
        
        # Filter out transitive predecessors while preserving parallel paths
        # - Removes: A -> B -> D (A is transitive)
        # - Keeps: A -> D and B -> D (parallel paths, e.g., residual connections)
        direct_predecessors = filter_transitive_predecessors(full_graph, all_predecessors, node_name)
        
        # Debug output for nodes with filtered predecessors
        if len(all_predecessors) != len(direct_predecessors):
            removed = all_predecessors - direct_predecessors
            print(f"\n{target_name}:")
            print(f"  All reachable: {sorted(all_predecessors)}")
            print(f"  Kept (direct + parallel): {sorted(direct_predecessors)}")
            print(f"  Removed (transitive): {sorted(removed)}")
        elif len(all_predecessors) > 1:
            # Multiple predecessors but none removed = parallel paths
            print(f"\n{target_name}:")
            print(f"  Parallel paths detected: {sorted(direct_predecessors)}")
        
        # Add edges from direct predecessors to this node (using mapped names)
        for pred in direct_predecessors:
            pred_mapped = node_name_mapping.get(pred, pred)
            neuro_graph.add_edge(pred_mapped, target_name)
    
    print(f"\n✓ Graph built: {neuro_graph.number_of_nodes()} nodes, {neuro_graph.number_of_edges()} edges")
    
    return neuro_graph


def save_graph_json(neuro_graph, output_path):
    """
    Save the neural graph to JSON format.
    
    Args:
        neuro_graph: NetworkX DiGraph object
        output_path: Path to save the JSON file
    """
    # Convert graph to JSON-serializable format
    graph_data = json_graph.node_link_data(neuro_graph)
    
    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    print(f"Graph saved to {output_path}")


def build_onnx_from_json(json_path, original_graph, output_path):
    """
    Build a simplified ONNX model directly from JSON graph file.
    This is useful for debugging - you can modify the JSON and regenerate ONNX.
    
    Args:
        json_path: Path to the JSON graph file
        original_graph: Original ONNX graph (for getting node details)
        output_path: Path to save the simplified ONNX file
    """
    from onnx import helper, TensorProto
    import json
    
    print("\n=== Building ONNX from JSON ===")
    print(f"Reading JSON: {json_path}")
    
    # Load JSON graph
    with open(json_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # Create mapping from original node names to node objects
    original_nodes_map = {}
    for node in original_graph.node:
        node_name = ".".join(node.name.split('/')[1:])
        original_nodes_map[node_name] = node
    
    # Get initializers mapping
    original_initializers = {init.name: init for init in original_graph.initializer}
    
    # Build node output mapping based on JSON edges
    node_outputs = {}  # Maps node_id -> output_tensor_name
    
    # Process nodes from JSON
    json_nodes = {node['id']: node for node in graph_data['nodes']}
    json_edges = [(link['source'], link['target']) for link in graph_data['links']]
    
    # Create NetworkX graph from JSON to get topological order
    import networkx as nx
    temp_graph = nx.DiGraph()
    for node_id in json_nodes.keys():
        temp_graph.add_node(node_id)
    for src, dst in json_edges:
        temp_graph.add_edge(src, dst)
    
    try:
        node_order = list(nx.topological_sort(temp_graph))
    except:
        print("Warning: Graph has cycles, using node order as-is")
        node_order = list(json_nodes.keys())
    
    # Build ONNX nodes
    new_nodes = []
    new_initializers = []
    graph_inputs = []
    
    for node_id in node_order:
        node_info = json_nodes[node_id]
        node_type = node_info.get('node_type', 'operation')
        
        # Handle input nodes
        if node_type == 'input':
            node_outputs[node_id] = node_id
            # Find original input info
            for orig_input in original_graph.input:
                if orig_input.name == node_id:
                    graph_inputs.append(orig_input)
                    break
            else:
                # Create default input
                graph_inputs.append(
                    helper.make_tensor_value_info(node_id, TensorProto.FLOAT, [1, 3, 32, 32])
                )
            continue
        
        # Get original node
        original_node_name = node_info.get('original_name', node_id)
        if original_node_name not in original_nodes_map:
            print(f"Warning: Node {original_node_name} not found in original graph")
            continue
        
        original_node = original_nodes_map[original_node_name]
        
        # Get predecessors from JSON edges
        predecessors = [src for src, dst in json_edges if dst == node_id]
        
        # Build inputs for this node
        new_inputs = []
        
        # Add data flow inputs
        for pred in predecessors:
            if pred in node_outputs:
                new_inputs.append(node_outputs[pred])
        
        # Add initializers (weights/biases)
        for orig_input in original_node.input:
            if orig_input in original_initializers:
                new_inputs.append(orig_input)
                if orig_input not in [init.name for init in new_initializers]:
                    new_initializers.append(original_initializers[orig_input])
        
        # Create output tensor name
        output_name = f"{node_id}_output"
        node_outputs[node_id] = output_name
        
        # Extract attributes
        node_attributes = {}
        for attr in original_node.attribute:
            attr_value = get_attribute_value(attr)
            if attr_value is not None:
                node_attributes[attr.name] = attr_value
        
        # Create ONNX node
        new_node = helper.make_node(
            original_node.op_type,
            inputs=new_inputs,
            outputs=[output_name],
            name=node_id,
            **node_attributes
        )
        new_nodes.append(new_node)
        
        print(f"  {node_id} ({original_node.op_type}): {len(new_inputs)} inputs -> {output_name}")
    
    # Create graph outputs (leaf nodes)
    graph_outputs = []
    leaf_nodes = [n for n in json_nodes.keys() 
                  if json_nodes[n].get('node_type') == 'operation' 
                  and not any(dst for src, dst in json_edges if src == n)]
    
    for leaf_node in leaf_nodes:
        output_name = node_outputs.get(leaf_node, f"{leaf_node}_output")
        output_info = helper.make_tensor_value_info(
            output_name,
            TensorProto.FLOAT,
            [None]
        )
        graph_outputs.append(output_info)
    
    print(f"\nGraph summary:")
    print(f"  - {len(new_nodes)} operation nodes")
    print(f"  - {len(graph_inputs)} inputs")
    print(f"  - {len(graph_outputs)} outputs")
    print(f"  - {len(new_initializers)} initializers")
    
    # Create the ONNX graph
    simplified_graph = helper.make_graph(
        new_nodes,
        "simplified_graph_from_json",
        graph_inputs,
        graph_outputs if graph_outputs else graph_inputs,
        initializer=new_initializers
    )
    
    # Create and save the model
    simplified_model = helper.make_model(simplified_graph)
    simplified_model.opset_import[0].version = 11
    
    try:
        onnx.save(simplified_model, output_path)
        print(f"\n✓ ONNX model saved to {output_path}")
    except Exception as e:
        print(f"\n✗ Error saving ONNX: {e}")
        print("Attempting with shape inference...")
        try:
            from onnx import shape_inference
            inferred_model = shape_inference.infer_shapes(simplified_model)
            onnx.save(inferred_model, output_path)
            print(f"✓ ONNX model saved with shape inference")
        except Exception as e2:
            print(f"✗ Failed: {e2}")
    
    print("=" * 50 + "\n")


def get_attribute_value(attr):
    """
    Extract the actual value from an ONNX attribute.
    """
    from onnx import AttributeProto
    
    if attr.type == AttributeProto.FLOAT:
        return attr.f
    elif attr.type == AttributeProto.INT:
        return attr.i
    elif attr.type == AttributeProto.STRING:
        return attr.s
    elif attr.type == AttributeProto.TENSOR:
        return attr.t
    elif attr.type == AttributeProto.GRAPH:
        return attr.g
    elif attr.type == AttributeProto.FLOATS:
        return list(attr.floats)
    elif attr.type == AttributeProto.INTS:
        return list(attr.ints)
    elif attr.type == AttributeProto.STRINGS:
        return list(attr.strings)
    elif attr.type == AttributeProto.TENSORS:
        return list(attr.tensors)
    elif attr.type == AttributeProto.GRAPHS:
        return list(attr.graphs)
    else:
        return None


def export_simplified_onnx(neuro_graph, original_graph, output_path):
    """
    Export the simplified neural graph as a new ONNX model for visualization.
    Creates a feed-forward ONNX graph with only the filtered nodes.
    
    Args:
        neuro_graph: NetworkX DiGraph object with filtered nodes
        original_graph: Original ONNX graph
        output_path: Path to save the simplified ONNX file
    """
    from onnx import helper, TensorProto
    
    # Create a mapping from original node names to node objects
    original_nodes_map = {}
    for node in original_graph.node:
        node_name = ".".join(node.name.split('/')[1:])
        original_nodes_map[node_name] = node
    
    # Collect nodes that are in the filtered graph
    simplified_nodes = []
    for node_id, node_data in neuro_graph.nodes(data=True):
        # Skip input nodes (they will be handled separately)
        if node_data.get('node_type') == 'input':
            continue
        
        # Get original node
        if node_id in original_nodes_map:
            original_node = original_nodes_map[node_id]
            simplified_nodes.append(original_node)
    
    # Get input tensors (only those actually used in the graph)
    used_inputs = set()
    for node_id, node_data in neuro_graph.nodes(data=True):
        if node_data.get('node_type') == 'input':
            used_inputs.add(node_id)
    
    graph_inputs = [inp for inp in original_graph.input if inp.name in used_inputs]
    
    # For outputs, we'll use the outputs of leaf nodes in our filtered graph
    leaf_nodes = [n for n in neuro_graph.nodes() if neuro_graph.out_degree(n) == 0 
                  and neuro_graph.nodes[n].get('node_type') == 'operation']
    
    graph_outputs = []
    for leaf_node in leaf_nodes:
        if leaf_node in original_nodes_map:
            original_node = original_nodes_map[leaf_node]
            for output_name in original_node.output:
                # Try to find the output tensor info from original graph
                output_info = None
                for orig_output in original_graph.output:
                    if orig_output.name == output_name:
                        output_info = orig_output
                        break
                
                # If not found in graph outputs, create a dummy output
                if output_info is None:
                    # Try to find in value_info
                    for value_info in original_graph.value_info:
                        if value_info.name == output_name:
                            output_info = value_info
                            break
                
                if output_info:
                    graph_outputs.append(output_info)
                else:
                    # Create a generic output tensor
                    output_info = helper.make_tensor_value_info(
                        output_name,
                        TensorProto.FLOAT,
                        [None]  # Unknown shape
                    )
                    graph_outputs.append(output_info)
    
    # Get initializers that are used by the filtered nodes
    used_initializers = set()
    for node in simplified_nodes:
        for input_name in node.input:
            used_initializers.add(input_name)
    
    graph_initializers = [init for init in original_graph.initializer 
                         if init.name in used_initializers]
    
    # Create the simplified graph
    simplified_graph = helper.make_graph(
        simplified_nodes,
        "simplified_neural_graph",
        graph_inputs,
        graph_outputs if graph_outputs else graph_inputs,  # Fallback to inputs if no outputs
        initializer=graph_initializers
    )
    
    # Create the model
    simplified_model = helper.make_model(simplified_graph)
    simplified_model.opset_import[0].version = 11  # Use opset version 11
    
    # Save the model
    onnx.save(simplified_model, output_path)
    print(f"Simplified ONNX graph saved to {output_path}")
    print(f"  - {len(simplified_nodes)} operation nodes")
    print(f"  - {len(graph_inputs)} input tensors")
    print(f"  - {len(graph_outputs)} output tensors")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX model to simplified neuron graph"
    )
    parser.add_argument("--onnx_path", type=str, required=True,
                       help="Path to original ONNX model")
    parser.add_argument("--ckpt_path", type=str,
                       help="Path to checkpoint file (required for normal mode)")
    parser.add_argument("--output_dir", type=str, default="./output", 
                       help="Directory to save output files")
    parser.add_argument("--from_json", type=str,
                       help="Build ONNX from existing JSON file (debug mode)")
    args = parser.parse_args()
    
    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load original ONNX model
    onnx_graph = onnx.load(args.onnx_path)
    inferred_model = shape_inference.infer_shapes(onnx_graph)
    graph = inferred_model.graph
    
    # Mode 1: Build ONNX from existing JSON (Debug mode)
    if args.from_json:
        print("\n" + "="*60)
        print("DEBUG MODE: Building ONNX from JSON")
        print("="*60)
        
        if not os.path.exists(args.from_json):
            print(f"Error: JSON file not found: {args.from_json}")
            return
        
        # Generate output path
        json_basename = os.path.splitext(os.path.basename(args.from_json))[0]
        if json_basename.endswith('_graph'):
            json_basename = json_basename[:-6]  # Remove '_graph' suffix
        onnx_output = os.path.join(args.output_dir, f"{json_basename}_from_json.onnx")
        
        # Build ONNX from JSON
        build_onnx_from_json(args.from_json, graph, onnx_output)
        
        print(f"\nYou can now visualize the ONNX model:")
        print(f"  netron {onnx_output}")
        return
    
    # Mode 2: Normal mode - Process ONNX and create simplified graph
    if not args.ckpt_path:
        print("Error: --ckpt_path is required for normal mode")
        print("Use --from_json to build ONNX from existing JSON")
        return
    
    print("\n" + "="*60)
    print("NORMAL MODE: Processing ONNX and creating simplified graph")
    print("="*60)
    
    state_dicts = torch.load(args.ckpt_path, map_location='cpu')
    neuro_graph = onnx_decode(graph, state_dicts.keys())
    
    # Print simplified graph information
    print(f"\nFiltered graph has {neuro_graph.number_of_nodes()} nodes and {neuro_graph.number_of_edges()} edges")
    
    # Generate output file paths
    model_name = os.path.splitext(os.path.basename(args.onnx_path))[0]
    json_path = os.path.join(args.output_dir, f"{model_name}_graph.json")
    onnx_viz_path = os.path.join(args.output_dir, f"{model_name}_simplified.onnx")
    
    # Save graph as JSON
    save_graph_json(neuro_graph, json_path)
    
    # Export simplified ONNX graph for visualization
    export_simplified_onnx(neuro_graph, graph, onnx_viz_path)
    
    print(f"\n" + "="*60)
    print("OUTPUTS:")
    print(f"  JSON: {json_path}")
    print(f"  ONNX: {onnx_viz_path}")
    print("\nTo rebuild ONNX from JSON (after editing):")
    print(f"  python {__file__} --onnx_path {args.onnx_path} --from_json {json_path}")
    print("="*60 + "\n")
    
    return neuro_graph

if __name__ == "__main__":
    main()