# Compiler Frontend User Guide

From torch.nn.Module to Json format simplified graph:

1. Run export_onnx.py first, to obtain a coarse-grained graph:

   ```
   python3 export_onnx.py --input_shape [tuple: dummy input shape of the model described with 1,C,H,W] --T [timestep: 1 is best]
    --num_classes [used for model creation] --neuron_config xxx --model_config xxx --ckpt_path [model checkpoint path] --save_dir [folder to save .onnx] --save_name [prefix of .onnx] --type [model type: vgg, sewresnet, spikformer] 
   ```

   After this, you will get a .onnx file in specified folder. Then:

2. Run sim_onnx.py, to obtain a compiler-friendly graph:

   ```
   python3 --onnx_path [path of obtained .onnx] --ckpt_path [path of checkpoint file] --output_dir [output dir of obtained json graph] --from_json [this is for debug, if you do want to debug, ignore this]
   ```

After these, you will get a frontend representation of model, saved in .json



Requirements:

```
onnx >= 1.20.0
networkx >= 3.4.2
torch >= 1.13.1 (torch >= 2.0 may raise unpredictable errors, we never test it) -> (bug of torch >= 2.0 is fixed now)
other necessary packeges if possible.
```

Better api support:

```
from gen_graph import export_graph
graph = export_graph(model,input_shape,debug=False) # Under Debug mode, onnx file and json file will be saved
```

Issues:

You should re-define a new Heaviside method to export graph:

```
From:
def heaviside(x)
    return (x > 0).float()
To:
class HeavisideFunction(torch.nn.Module):
    def forward(self, x):
        return (x > 0).float()     
```

Otherwise there will be bugs, this issue has been pushed to the repo of Soul, @yudi will fix this later.   
