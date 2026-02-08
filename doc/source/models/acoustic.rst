Acoustic 模型
======================

以下为Soul工具包支持的声学数据处理模型：

SpikingMLP
-------------------------------------------
引用：
    Jianfei Yang et al. "SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing." Patterns'2023.https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
使用方法：
    设置--model或-m：mlp
参数量：

介绍：
    适用于边缘场景的脉冲多层感知机网络结构，针对声学传感任务进行了定制优化。
特殊配置：
    * hidden_dim:1024 （int）MLP中间层的隐藏维度

SpikingLeNet
-------------------------------------------
引用：
    Jianfei Yang et al. "SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing." Patterns'2023.https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
使用方法：
    设置--model或-m：lenet
参数量：
    
介绍：
    适用于边缘场景的脉冲LeNet网络结构，针对声学传感任务进行了定制优化。
特殊配置：
    * hidden_dim:512 （int）全连接部分中间层的隐藏维度

SpikingRNN
-------------------------------------------
引用：
    Jianfei Yang et al. "SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing." Patterns'2023.https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
使用方法：
    设置--model或-m：rnn
参数量：

介绍：
    适用于边缘场景的脉冲循环神经网络结构，针对声学传感任务进行了定制优化。
特殊配置：
    * hidden_dim:128 （int）内部循环层的隐藏维度
    * num_layers:1 （int）循环层数量
    * last_step:true （bool）以序列的最后一个步长作为输出
    

SpikingConvRNN
-------------------------------------------
引用：
    Jianfei Yang et al. "SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing." Patterns'2023.https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
使用方法：
    设置--model或-m：convrnn
参数量：

介绍：
    适用于边缘场景的脉冲卷积循环神经网络结构，针对声学传感任务进行了定制优化。
特殊配置：
    * hidden_dim:128 （int）内部循环层的隐藏维度
    * num_layers:1 （int）循环层数量
    * last_step:false （bool）采用序列最后一个步骤作为输出

SpikingVGG
-------------------------------------------
引用：
    Di Yu et al., "EC-SNN: Splitting Deep Spiking Neural Networks for Edge Devices", IJCAI'2024.https://github.com/AmazingDD/EC-SNN/
使用方法：
    设置--model或-m：spikingvgg9、spikingvgg16
参数量：

介绍：
    适用于声学分类任务的VGG架构脉冲神经网络实现。
特殊配置：
    * mlp_ratio:1.0 （float）MLP中间层隐藏维度的扩展系数
    * mlp_hidden_dim:2048 （int）VGG网络多层感知机部分中间层的隐藏维度

SEWResNet
-------------------------------------------
引用：
    Wei Fang et al., "Deep residual learning in spiking neural networks", NeurIPS'2021.https://github.com/fangwei123456/Spike-Element-Wise-ResNet
使用方法：
    设置--model或-m：sewresnet18、sewresnet50
参数量：

介绍：
    适用于声学分类任务的ResNet架构网络实现。
特殊配置：
    * connect_function:'ADD' （str）可选的脉冲按元素操作：加法（ADD）、按位与（IAND）、逻辑与（AND）
    * groups:1 （int）分组卷积
    * width_per_group:64 （int）中间层的通道数
    * zero_init_residual:False （bool）是否对每个残差分支的最后一个批归一化层进行零初始化

MSResNet
-------------------------------------------
引用：
    Yifan Hu et al., "Advancing spiking neural networks toward deep residual learning", TNNLS'2024.https://github.com/Ariande1/MS-ResNet
使用方法：
    设置--model或-m：msresnet18、msresnet50
参数量：

介绍：
    适用于声学分类任务的ResNet架构网络实现。
特殊配置：
    * groups:1 （int）分组卷积（仅适用于瓶颈层）
    * base_width:64 （int）中间层的通道数
    

SpikingTCN
-------------------------------------------
使用方法：
    设置--model或-m：spikingtcn
参数量：

介绍：
    适用于声学分类任务的时序卷积网络架构的脉冲神经网络实现。
特殊配置：
    * ksize:7 （int）核大小
    * hidden_channels:[128, 256, 256] （list）中间模型架构

Spikformer
-------------------------------------------
引用：
    Zhaokun Zhou et al., "Spikformer: when spiking neural network meets transformer", ICLR'2023.https://github.com/ZK-Zhou/spikformer
使用方法：
    设置--model或-m：spikformer256、spikformer384、spikformer512
参数量：

介绍：
    适用于声音数据分类任务的 Transformer 架构脉冲神经网络模型实现。
特殊配置：
    * patch_size:16 （int）输入块大小
    * mlp_ratio:4 （float）MLP中间层隐藏维度的扩展系数
    * num_heads:8 （int）注意力输出的头数

MetaSpikeformer
-------------------------------------------
引用：
    Man Yao et al., "Spike-driven Transformer V2: Meta Spiking Neural Network Architecture Inspiring the Design of Next-generation Neuromorphic Chips", ICLR'2024.https://github.com/BICLab/Spike-Driven-Transformer-V2
使用方法：
    设置--model或-m：metaspikeformer256、metaspikeformer384、metaspikeformer512
参数量：

介绍：
    适用于声音数据分类任务的 Transformer 架构脉冲神经网络模型实现。
    需要注意的是，原始的重参数化卷积（RepConv）会导致所有模型无法收敛。
    此外，两个卷积层之间未设置任何神经元直接相连，这在神经形态芯片中属于不合法的计算逻辑。
    因此，我们采用普通的线性层替代重参数化卷积（RepConv）部分，以实现模型的稳定部署。
特殊配置：
    * mlp_ratio:4 （float）MLP中间层隐藏维度的扩展系数
    * num_heads:8 （int）注意力输出的头数
    
SpikingResformer
-------------------------------------------
引用：
    Xinyu Shi et al., "SpikingResformer: Bridging ResNet and Vision Transformer in Spiking Neural Networks", CVPR'2024.https://github.com/xyshi2000/SpikingResformer
使用方法：
    设置--model或-m：spikingresformer192、spikingresformer256、spikingresformer384、spikingresformer512
参数量：

介绍：
    适用于声音数据分类任务的 Transformer 架构脉冲神经网络模型实现。
特殊配置：
    * group_size:64 （int）卷积组
    * mlp_ratio:4 （float）MLP中间层隐藏维度的扩展系数

QKFormer
-------------------------------------------
引用：
    Chenlin Zhou et al., "QKFormer: Hierarchical Spiking Transformer using Q-K Attention", NeurIPS'2024.https://github.com/zhouchenlin2096/QKFormer
使用方法：
    设置--model或-m：qkformer256、qkformer384、qkformer512
参数量：

介绍：
    适用于声音分类任务的 Transformer 架构脉冲神经网络模型实现。
特殊配置：
    * num_heads:8 （int）注意力输出的头数
    * patch_size:16 （int）输入块大小
    * mlp_ratio:4 （float）MLP中间层隐藏维度的扩展系数
    