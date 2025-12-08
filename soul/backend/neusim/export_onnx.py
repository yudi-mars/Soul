import sys
sys.path.append("../")
import onnx
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.vision.spikingvgg import SpikingVGG5
from model.vision.sewresnet import SEWResNet18
from neuron import neuron_map
from utils.surrogate import surrogate_map
import yaml

def merge_configs(neuron_config,model_config):
    with open(neuron_config, 'r') as f:
        neuron_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(model_config, 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    neuron_config['surrogate_function'] = surrogate_map[neuron_config['surrogate'].lower()]
    neuron_config["neuron_type"] = "LIF"
    return {
        **model_config,
        "neuron": neuron_map[neuron_config['neuron_type'].lower()](neuron_config),
    }

def create_demo_model(configs,type):
    if type == "SpikingVGG":
        model = SpikingVGG5(configs)
    elif type == "sewresnet":
        model = SEWResNet18(configs)
    else:
        raise NotImplementedError(f"Model type {type} not implemented")
    return model

def fix_batchnorm_for_export(model):
    """
    Fix BatchNorm layers to avoid ONNX export issues in PyTorch 2.0+.
    Converts BatchNorm to eval mode and fixes parameter names.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            # Ensure running_mean and running_var are properly named
            if hasattr(module, 'running_mean'):
                module.running_mean.requires_grad = False
            if hasattr(module, 'running_var'):
                module.running_var.requires_grad = False
            if hasattr(module, 'num_batches_tracked'):
                module.num_batches_tracked.requires_grad = False
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_shape", type=tuple, default=(1, 3, 32, 32))
    parser.add_argument("--T",type=int,default=1)
    parser.add_argument("--num_classes",type=int,default=10)
    parser.add_argument("--neuron_config",type=str)
    parser.add_argument("--model_config",type=str)
    parser.add_argument("--ckpt_path",type=str,default="None")
    parser.add_argument("--save_dir",type=str)
    parser.add_argument("--save_name",type=str)
    parser.add_argument("--type",type=str)
    parser.add_argument("--use_legacy_exporter", action="store_true",
                       help="Force use of legacy ONNX exporter")

    args = parser.parse_args()
    # Add default parameters for ResNet
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load original multi-step model
    configs = merge_configs(args.neuron_config,args.model_config)
    configs['time_step'] = args.T
    configs['input_channels'] = args.input_shape[1]
    configs["num_classes"] = args.num_classes
    model = create_demo_model(configs,args.type)

    # Load checkpoint
    if os.path.exists(args.ckpt_path):
        print(f"Loading checkpoint from: {args.ckpt_path}")
        state_dict = torch.load(args.ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully!")
    else:
        print(f"Warning: Checkpoint not found at {args.ckpt_path}")
    
    model.eval()
    
    # Fix BatchNorm for ONNX export
    model = fix_batchnorm_for_export(model)
    print("Model prepared for ONNX export")
    # Create dummy input
    dummy_input = torch.randn(args.input_shape)
    dummy_input = dummy_input.unsqueeze(0).repeat(args.T,1,1,1,1)
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
    save_path = os.path.join(args.save_dir,args.save_name)
    print(f"Exporting to: {save_path}")
    
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
    print(f"State dict saved to: {sd_save_path}")

if __name__ == "__main__": 
    # 临时注释掉自定义操作符注册，避免PyTorch 2.0.1的bug
    # register_NeuronNode_op()
    # register_SnnConv2d_op()
    # register_BatchNorm2d_op()
    # register_LIFNode_op()
    print("Warning: Custom operators are disabled due to PyTorch 2.0.1 bug")
    print("The exported ONNX model will use expanded basic operations instead of custom nodes")
    main()

