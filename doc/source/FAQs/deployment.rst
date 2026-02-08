使用ncnn进行设备部署
==================

为了将Soul中的SNN模型部署到边缘设备上并使用ncnn框架进行推理，我们要做一下几个主要工作：

* 部署工具准备
* 调整Soul中SNN模型并导出TorchScript
* 使用pnnx工具转换模型并插入神经元算子（此处以为LIFNode例）
* 在ncnn框架中实现神经元算子并进行推理

部署工具准备
----------------

部署过程主要用到三个工具：

* `Soul工具包 <https://github.com/yudi-mars/Soul>`_ 
* `pnnx转换工具 <https://github.com/pnnx>`_
* `ncnn推理框架 <https://github.com/Tencent/ncnn>`_

请下载并配置好相关环境。

Soul模型准备
----------------

我们使用ncnn推理框架进行模型推理。ncnn只支持四维输入（d,c,h,w），然而Soul工具包一般使用五维输入（timestep,batchsize,c,h,w）进行训练以同时兼顾常见算子的计算与
SNN计算过程的并行性。为此，在导出可供ncnn推理的模型之前，我们需要对Soul工具包中的原有模型进行修改。此处，我们以 SpikingMLP 模型（详情见 :doc:`../models/index`）为例：

.. code-block:: python

    class SpikingMLP(nn.Module):
        def __init__(self, config):
            super().__init__()

            self.num_classes = config['num_classes']
            self.T = config['time_step']

            C, H, W = config['input_channels'], config['input_height'], config['input_width']
            lif = config['neuron']

            self.ln1 = nn.Linear(C * H * W, config['hidden_dim'])
            self.lif1 = deepcopy(lif)

            self.ln2 = nn.Linear(config['hidden_dim'], config['hidden_dim'])
            self.lif2 = deepcopy(lif)

            self.head = nn.Linear(config['hidden_dim'], self.num_classes)

        def forward(self, x):
            functional.reset_net(self)

            x = x.flatten(2)  # (T, B, C, H, W) -> (T, B, CHW)

            x = multi_time_forward(x, self.ln1)
            x = self.lif1(x)

            x = multi_time_forward(x, self.ln2)
            x = self.lif2(x)

            x = self.head(x.mean(0)) # (T, B, D) -> (B, D)

            return x

修改为：

.. code-block:: python

    class SpikingMLP(nn.Module):
        def __init__(self, config):
            super().__init__()

            self.num_classes = config['num_classes']
            self.T = config['time_step']

            C, H, W = config['input_channels'], config['input_height'], config['input_width']
            lif = config['neuron']

            self.ln1 = nn.Linear(C * H * W, config['hidden_dim'])
            self.lif1 = deepcopy(lif)

            self.ln2 = nn.Linear(config['hidden_dim'], config['hidden_dim'])
            self.lif2 = deepcopy(lif)

            self.head = nn.Linear(config['hidden_dim'], self.num_classes)

        def forward(self, x):
            functional.reset_net(self)

            #x = x.flatten(2)  # (T, B, C, H, W) -> (T, B, CHW)

            x = multi_time_forward(x, self.ln1)
            x = self.lif1(x)

            x = multi_time_forward(x, self.ln2)
            x = self.lif2(x)

            #x = self.head(x.mean(0)) # (T, B, D) -> (B, D)
            x = self.head(x)

修改 `multi_time_forward()` 函数:

.. code-block:: python

    def multi_time_forward(x_seq, stateless_module):
        y_shape = [x_seq.shape[0], x_seq.shape[1]] # [T, B]
        y = x_seq.flatten(0, 1)
        if isinstance(stateless_module, (list, tuple, nn.Sequential)):
            for m in stateless_module:
                y = m(y)
        else:
            y = stateless_module(y)
        
        y_shape.extend(y.shape[1:]) # [T, B] + [...] -> [T, B, ...]
        return y.view(y_shape)

修改为

.. code-block:: python

    def multi_time_forward(x_seq, stateless_module):
        y = x_seq
        if isinstance(stateless_module, (list, tuple, nn.Sequential)):
            for m in stateless_module:
                y = m(y)
        else:
            y = stateless_module(y)
        
        return y

至此 SpikingMLP 模型修改完成，当然，你也可以在yaml文件（详情见 :doc:`../params`）中添加配置以控制模型接收五维输入还是四维输入。
提示：一般修改位置有 `multi_time_forward()方法`、 `flatten()方法`、 `transpose()方法`、 `reshape()方法`。

模型转换
----------------

.. code-block:: python

    import torch
    import torch.distributed as dist
    from soul.model import *
    from soul.neuron import *
    from soul.utils import *
    import torch.nn as nn

    def replace_lifnode_with_identity(model: nn.Module) -> nn.Module:
        # 遍历模型的所有子模块（带名称）
        for name, module in model.named_children():
            if isinstance(module, LIFNode):
                setattr(model, name, nn.Identity())
            else:
                replace_lifnode_with_identity(module)
        return model

    if __name__ == '__main__':
        # 初始化所有配置
        config = init_config()

        config['is_distributed'] = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        global_rank = 0

        # 初始化日志
        if global_rank == 0:
            log_path = os.path.join(
                config['log_dir'],
                config['dataset_name'].lower(),
                config['model'].lower(),
                config['neuron_type'].lower()
            )
            ensure_dir(log_path)
            logger = setup_logger(os.path.join(log_path, f'record-{get_local_time()}.log'), default_level=config['state'])
            logger.info(f'Distributed Training: {config["is_distributed"]}')
        else:
            logger = None

        # 加载数据集及相关配置
        if global_rank == 0:
            logger.info('Load data...')
        train_dataset, test_dataset = load_dataset(config)

        # 加载SNN模型
        if global_rank == 0:
            logger.info(f'Load SNN model: {config["model"]} featured {config["neuron_type"].upper()} neuron...')
            logger.info(f'#Training Samples: {len(train_dataset)}; #Test Samples: {len(test_dataset)}')

        if global_rank == 0:
            logger.debug(f'surrogate function: {config["surrogate"]}')
        config['surrogate_function'] = surrogate_map[config['surrogate']]
        config['neuron'] = neuron_map[config['neuron_type'].lower()](config)

        model = model_map[config['application']][config['model'].lower()](config)
        model = replace_lifnode_with_identity(model) # 暂时移除LIFNode算子避免torch.jit.trace无法转换的问题
        model.to(device)

        model = torch.jit.load("model_path.pt", map_location=device)

        shape = [1,3,32,32] #数据集输入形状
        traced_model = torch.jit.trace(model, torch.randn(shape))
        torch.jit.save(traced_model, './save_model/jit_model.pt')

至此，生成可以在ncnn框架上部署的模型。
接下来，我们需要使用pnnx转换该模型为ncnn框架可用的param文件及bin文件。

.. code-block:: bash

    .\pnnx.exe jit_model.pt inputshape=input_shape device=cpu optlevel=2 fp16=0

使用优化等级设置为2的原因是我们需要在之后将LIFNode算子插入回pnnx生成的param文件中，其他优化等级会导致算子结构不清晰或无法插入的问题。
保留pnnx生成的.ncnn.param文件（模型结构信息）与.ncnn.bin文件（权重信息），我们需要在.ncnn.param文件中的适当位置插入LIFNode算子。

.ncnn.param文件主要分为三个部分：

* magic数
* 算子数及blob数
* 算子描述（多行）：包括算子类型、算子名称、输入blob数、输出blob数、输入blob名称（数量为输入blob数个）、输出blob名称（数量为输出blob数）、初始化参数（若干）

算子的插入工作可以分为两个部分：

* 删除无用算子：pnnx在处理一些多余的算子时可能出现无法转化的情况（如处理 `flatten()方法` 时失败并保留为torch.flatten算子），此时将其直接去除并修改blob链就可以了。
* 在正确的位置插入LIFNode
* 修改算子数及blob数

例子（以best_mlp_lif_cifar10_T4.ncnn.param为例）：

::

    7767517
    8 8
    Input                    in0                      0 1 in0
    Flatten                  flatten_3                1 1 in0 1
    Gemm                     gemm_0                   1 1 1 3 10=4 2=0 3=1 4=0 5=1 6=1 7=1 8=1024 9=3072
    LIFNODE                  lif0                     1 1 3 lif3 7=1 5=1024 6=1
    Gemm                     gemm_1                   1 1 lif3 4 10=4 2=0 3=1 4=0 5=1 6=1 7=1 8=1024 9=1024
    LIFNODE                  lif1                     1 1 4 lif4 7=1 5=1024 6=1
    Reduction                mean_4                   1 1 lif4 5 0=3 1=0 -23303=1,0 4=0 5=1
    InnerProduct             linear_2                 1 1 5 out0 0=10 1=1 2=10240
 

使用ncnn框架进行推理
----------------

现在，我们已经生成了可以在ncnn框架上执行推理的.ncnn.param文件与.ncnn.bin文件，现在，为了使ncnn框架可以支持SNN模型的推理，我们需要对有框架进行一些小改造。

首先，我们要为ncnn框架添加LIFNode算子的支持，若您需要添加其它自定义算子，请参考 `ncnn文档——如何添加自定义层 <https://github.com/Tencent/ncnn/wiki/add-custom-layer.zh>`_。
我们根据Soul中的定义实现LIFNode算子。

lifnode.h:

.. code-block:: c++

    #ifndef LIFNODE_H
    #define LIFNODE_H

    #include "layer.h"
    namespace ncnn {

    class LIFNODE : public Layer
    {
    public:
        LIFNODE();
        virtual int load_param(const ParamDict& pd);
        virtual int load_model(const ModelBin& mb);
        virtual int load_model(const ModelBin& mb, const Option& opt);
        virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
        virtual int reset();
        virtual int destroy();

    private:
        virtual int single_step_forward_hard_reset_decay_input(Mat& x, const Option& opt) const;
        virtual int single_step_forward_hard_reset_no_decay_input(Mat& x, const Option& opt) const;
        virtual int single_step_forward_soft_reset_decay_input(Mat& x, const Option& opt) const;
        virtual int single_step_forward_soft_reset_no_decay_input(Mat& x, const Option& opt) const;
        virtual int fill(float init_v);

    public:
        float v_init;
        float tau;
        bool decay_input;
        float v_threshold;
        float v_reset;
        int w, h, c;
        mutable Mat v_data;
        bool support_vulkan;
    };
    } // namespace ncnn

    #endif

lifnode.cpp

.. code-block:: c++

    #include "lifnode.h"
    #include <stdio.h>

    namespace ncnn {

    LIFNODE::LIFNODE()
    {
        support_vulkan = false;
        one_blob_only = true;
        support_inplace = true;
        v_data = Mat();
    }

    int LIFNODE::load_param(const ParamDict& pd)
    {
        tau = pd.get(0, 2.0f);
        decay_input = pd.get(1, true);
        v_threshold = pd.get(2, 1.0f);
        v_reset = pd.get(3, 0.0f);
        v_init = pd.get(4, 0.0f);
        w = pd.get(5, 0);
        h = pd.get(6, 0);
        c = pd.get(7, 0);
        return 0;
    }

    int LIFNODE::load_model(const ModelBin& mb)
    {
        size_t elemsize = 4;
        if (v_data.empty())
        {
            v_data.create(w, h, c, elemsize);
            fill(v_init);
        }
        return 0;
    }

    int LIFNODE::load_model(const ModelBin& mb, const Option& opt)
    {
        size_t elemsize = 4;

        if (v_data.empty())
        {
            /*if (std::string(opt.mode).find("old") != std::string::npos)
            {
                v_data.create(w, h, c, elemsize, opt.weight_allocator);
            }
            else
            {
                v_data.create(w, h, c, elemsize, opt.persistence_weight_allocator);
            }*/

            v_data.create(w, h, c, elemsize);

            fill(v_init);
        }
        return 0;
    }

    int LIFNODE::single_step_forward_hard_reset_decay_input(Mat& x, const Option& opt) const
    {
        int size = x.w * x.h;

        //print_float_mat_lif(v_data, "/home/root/flexnn/data/v_data.txt");

    #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < x.c; i++)
        {
            float* x_ptr = x.channel(i);
            float* v_ptr = v_data.channel(i);
            for (int j = 0; j < size; j++)
            {
                v_ptr[j] = v_ptr[j] + (x_ptr[j] - (v_ptr[j] - v_reset)) / tau;
                x_ptr[j] = v_ptr[j] >= v_threshold ? 1.0 : 0.0;
                v_ptr[j] = x_ptr[j] * v_reset + v_ptr[j] * (1.0 - x_ptr[j]);
            }
        }
        return 0;
    }

    int LIFNODE::single_step_forward_hard_reset_no_decay_input(Mat& x, const Option& opt) const
    {
        int size = x.w * x.h;

    #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < x.c; i++)
        {
            float* x_ptr = x.channel(i);
            float* v_ptr = v_data.channel(i);
            for (int j = 0; j < size; j++)
            {
                v_ptr[j] = v_ptr[j] + (v_ptr[j] - v_reset) / tau + x_ptr[j];
                x_ptr[j] = v_ptr[j] >= v_threshold ? 1.0 : 0.0;
                v_ptr[j] = x_ptr[j] * v_reset + v_ptr[j] * (1.0 - x_ptr[j]);
            }
        }
        return 0;
    }

    int LIFNODE::single_step_forward_soft_reset_decay_input(Mat& x, const Option& opt) const
    {
        int size = x.w * x.h;

    #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < x.c; i++)
        {
            float* x_ptr = x.channel(i);
            float* v_ptr = v_data.channel(i);
            for (int j = 0; j < size; j++)
            {
                v_ptr[j] = v_ptr[j] + (x_ptr[j] - v_ptr[j]) / tau;
                x_ptr[j] = v_ptr[j] >= v_threshold ? 1.0 : 0.0;
                v_ptr[j] = v_ptr[j] - x_ptr[j] * v_threshold;
            }
        }
        return 0;
    }

    int LIFNODE::single_step_forward_soft_reset_no_decay_input(Mat& x, const Option& opt) const
    {
        int size = x.w * x.h;

    #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < x.c; i++)
        {
            float* x_ptr = x.channel(i);
            float* v_ptr = v_data.channel(i);
            for (int j = 0; j < size; j++)
            {
                v_ptr[j] = v_ptr[j] * (1.0 - 1.0 / tau) + x_ptr[j];
                x_ptr[j] = v_ptr[j] >= v_threshold ? 1.0 : 0.0;
                v_ptr[j] = v_ptr[j] - x_ptr[j] * v_threshold;
            }
        }
        return 0;
    }

    int LIFNODE::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
    {
        int res;
        if (v_reset < 0)
        {
            if (decay_input)
            {
                res = single_step_forward_soft_reset_decay_input(bottom_top_blob, opt);
            }
            else
            {
                res = single_step_forward_soft_reset_no_decay_input(bottom_top_blob, opt);
            }
        }
        else
        {
            if (decay_input)
            {
                res = single_step_forward_hard_reset_decay_input(bottom_top_blob, opt);
            }
            else
            {
                res = single_step_forward_hard_reset_no_decay_input(bottom_top_blob, opt);
            }
        }
        if (!res)
        {
            return 0;
        }
        else
        {
            return -1;
        }
    }

    int LIFNODE::reset()
    {
        if (v_data.empty())
            return -1;

        fill(v_init);
        //print_float_mat_lif(v_data, "/home/root/flexnn/data/v_data.txt");
        return 0;
    }

    int LIFNODE::fill(float init_v)
    {
        if (v_data.empty())
            return -1;

        int h, w, d, c;
        h = v_data.h;
        w = v_data.w;
        d = v_data.d;
        c = v_data.c;
        for (int i = 0; i < c; i++)
        {
            float* ptr = v_data.channel(i);
            for (int j = 0; j < d; j++)
            {
                for (int k = 0; k < h; k++)
                {
                    for (int l = 0; l < w; l++)
                    {
                        *ptr = init_v;
                        ptr++;
                    }
                }
            }
        }
        return 0;
    }

    int LIFNODE::destroy()
    {
        return 0;
    }

    } // namespace ncnn

在CMakeLists.txt文件中添加：

::

    ncnn_add_layer(LIFNODE)

最后，修改原有推理代码：

.. code-block:: c++

    for (int t = 0; t < TIMESTEP; t++)
    {
        ncnn::Mat in = generate_input(shape);
        ex.input(input_names[0], in);
        ncnn::Mat out;
        ex.extract(output_names[0], out);
        if (mem_v.empty())
        {
            mem_v.create_like(out);
        }

        for (int c = 0; c < 1; c++)
        {
            float* ptr_out = out.channel(c);
            float* ptr_tmp = mem_v.channel(c);

            int size = out.w * out.h;

            for (int s = 0; s < size; s++)
            {
                *ptr_out = *ptr_out + *ptr_tmp;

                ptr_out++;
                ptr_tmp++;
            }
        }
    }
    for (int k = 0; k < lifnodes.size(); k++)
    {
        lifnodes[k]->reset();
    }

    ex.clear();

至此，代码部分全部完成，编译部分请参考 `如何构建 <https://github.com/Tencent/ncnn/blob/master/docs/how-to-build/how-to-build.md>`_
若要开启vulkan支持，可通过编译参数 `-DNCNN_VULKAN=ON` 开启。