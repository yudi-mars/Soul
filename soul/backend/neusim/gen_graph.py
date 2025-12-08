import sys
sys.path.append("../../")
sys.path.append("../../../")
# the above codes are for debug, you can remove in formal project
from backend.neusim.export_onnx import fix_batchnorm_for_export,create_demo_model,merge_configs 
from backend.neusim.sim_onnx import onnx_decode,save_graph_json,build_onnx_from_json
import torch
import os
import onnx
from onnx import shape_inference
import networkx
from networkx.readwrite import json_graph


def export_temp_onnx(model,input_shape):
    model.eval()
    # Fix BatchNorm for ONNX export
    model = fix_batchnorm_for_export(model)
    print("Model prepared for ONNX export")
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    dummy_input = dummy_input.unsqueeze(0)
    # Test forward pass
    print(f"Testing forward pass with input shape: {dummy_input.shape}")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    # try:

    # except Exception as e:
    #     print(f"Error in forward pass: {e}")
    #     return

    # Export to ONNX
    save_path = os.path.join("./temp_onnx_graph.onnx")
    
    # Check PyTorch version to determine which exporter to use
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    try:
        if torch_version >= (2, 1):
            # PyTorch 2.1+ - Use legacy exporter to avoid BatchNorm issues
            print("Using legacy ONNX exporter (PyTorch 2.1+)")
            
            # Force use of legacy exporter
            import os as env_os
            env_os.environ["TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK"] = "0"
            
            torch.onnx.export(
                model,
                dummy_input,
                save_path,
                export_params=False,  # Must be True to avoid naming issues
                opset_version=11,
                do_constant_folding=False,
                input_names=['input'],
                output_names=['output'],
                verbose=False,
                # Use legacy exporter
                dynamo=False
            )
        else:
            # PyTorch < 2.1 - Use standard export
            print("Using standard ONNX exporter (PyTorch < 2.1)")
            torch.onnx.export(
                model,
                dummy_input,
                save_path,
                export_params=True,  # Changed to True
                opset_version=11,
                do_constant_folding=False,
                input_names=['input'],
                output_names=['output'],
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH
            )
        
        print(f"ONNX export successful! Model saved to: {save_path}")
        
        # Verify the exported model
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification successful!")
        
    except Exception as e:
        print(f"Error during ONNX export with current settings: {e}")
        print("\nTrying alternative export method...")
        
        # Alternative method: Use older opset or different settings
        try:
            torch.onnx.export(
                model,
                dummy_input,
                save_path,
                export_params=True,
                opset_version=18,  # Try newer opset
                do_constant_folding=False,
                input_names=['input'],
                output_names=['output'],
                verbose=False
            )
            print(f"ONNX export successful with opset 18! Model saved to: {save_path}")
        except Exception as e2:
            print(f"Alternative export also failed: {e2}")
            import traceback
            traceback.print_exc()
            return
    sd_save_path = save_path.replace(".onnx",".pth")
    torch.save(model.state_dict(),sd_save_path)

def simplify_onnx_graph(debug):
    onnx_graph = onnx.load("./temp_onnx_graph.onnx")
    inferred_model = shape_inference.infer_shapes(onnx_graph)
    graph = inferred_model.graph
    
    print("\n" + "="*60)
    print("NORMAL MODE: Processing ONNX and creating simplified graph")
    print("="*60)
    
    state_dicts = torch.load("./temp_onnx_graph.pth", map_location='cpu')
    neuro_graph = onnx_decode(graph, state_dicts.keys())
    
    # Print simplified graph information
    print(f"\nFiltered graph has {neuro_graph.number_of_nodes()} nodes and {neuro_graph.number_of_edges()} edges")
    graph_data = json_graph.node_link_data(neuro_graph)
    
    # Save graph as JSON
    if debug:
        json_path = "./temp_onnx_graph.json"
        save_graph_json(neuro_graph, json_path)
        print("\n" + "="*60)
        print("DEBUG MODE: Building ONNX from JSON")
        print("="*60)
        onnx_output = "temp_onnx_from_json.onnx"
        # Build ONNX from JSON
        build_onnx_from_json(json_path, graph, onnx_output)
        
        print(f"\nYou can now visualize the ONNX model:")
        print(f"  netron {onnx_output}")
    else:
        clear_temp_files()
    return graph_data

def clear_temp_files():
    os.remove("./temp_onnx_from_json.onnx") if os.path.exists("./temp_onnx_from_json.onnx") else 0
    os.remove("./temp_onnx_graph.onnx") if os.path.exists("./temp_onnx_graph.onnx") else 0
    os.remove("./temp_onnx_graph.json") if os.path.exists("./temp_onnx_graph.json") else 0
    os.remove("./temp_onnx_graph.pth")  if os.path.exists("./temp_onnx_graph.pth") else 0

def export_graph(model,input_shape,debug=False):
    export_temp_onnx(model,input_shape)
    graph_data = simplify_onnx_graph(debug)
    return graph_data


if __name__ == "__main__":
    input_shape = (1,3,32,32)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes",type=int,default=10)
    parser.add_argument("--neuron_config",type=str,default = "../../config/neuron/LIF.yaml")
    parser.add_argument("--model_config",type=str,default = "../../config/model/vision/sewresnet.yaml")
    parser.add_argument("--debug",type=bool,default = False)
    args = parser.parse_args()
    configs = merge_configs(args.neuron_config,args.model_config)
    configs['time_step'] = 1
    configs['input_channels'] = input_shape[1]
    configs["num_classes"] = args.num_classes
    model = create_demo_model(configs,type="sewresnet")
    graph_data = export_graph(model,input_shape,args.debug)
    print(graph_data)
    pass