Motion Model
==================

The following are the motion data processing models supported by the Soul toolkit:

SpikingMLP
-------------------------------------------
Citation:
    Jianfei Yang et al. "SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing." Patterns'2023.https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
Usage:
    Set the --model or -m parameter to: mlp
Number of Parameters:：

Description：
    A spike-based multi-layer perceptron network architecture customized and optimized for motion behavior tasks in edge scenarios.
Special Configurations：
    * hidden_dim:1024 （int）hidden dimension for the intermediate layer of MLP

SpikingLeNet
-------------------------------------------
Citation:
    Jianfei Yang et al. "SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing." Patterns'2023.https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
Usage:
    Set the --model or -m parameter to: lenet
Number of Parameters:：
    
Special Configurations：
    * hidden_dim:512 （int）hidden dimension for the intermediate layers in full connection part

SpikingRNN
-------------------------------------------
Citation:
    Jianfei Yang et al. "SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing." Patterns'2023.https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
Usage:
    Set the --model or -m parameter to: rnn
Number of Parameters:：

Description：
    A spike-based recurrent neural network architecture customized and optimized for motion behavior tasks in edge scenarios.
Special Configurations：
    * hidden_dim:128 （int）hidden dimension for interior recurrent layers
    * num_layers:1 （int）the number of recurrent layers
    * last_step:true （bool）use last sequence step as output

SpikingConvRNN
-------------------------------------------
Citation:
    Jianfei Yang et al. "SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing." Patterns'2023.https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
Usage:
    Set the --model or -m parameter to: convrnn
Number of Parameters:：

Description：
    A spike-based convolutional recurrent neural network architecture customized and optimized for motion behavior tasks in edge scenarios.
Special Configurations：
    * hidden_dim:128 （int）hidden dimension for interior recurrent layers
    * num_layers:1 （int）the number of recurrent layers
    * last_step:false （bool）use last sequence step as output

DCNN
-------------------------------------------
Citation:
    Yang Jianbo et al., "Deep Convolutional Neural Networks On Multichannel Time Series For Human Activity Recognition", IJCAI'2015.https://github.com/gongchenooo/MobiCom24-Delta
Usage:
    Set the --model or -m parameter to: dcnn
Number of Parameters:：

Description：
    Implementation of a spike-wise deep convolutional neural network architecture model for human activity recognition tasks.
Special Configurations：
    * mlp_hidden_dim:400 （int）hidden dimension for intermediate layers in mlp part

SenseHAR
-------------------------------------------
Citation:
    Jeyakumar J.V. et al., "SenseHAR: A Robust Virtual Activity Sensor for Smartphones and Wearables", SenSys'2019.https://github.com/devanshuDesai/SenseHAR
Usage:
    Set the --model or -m parameter to: sensehar
Number of Parameters:：

Description：
    Implementation of a spike-wise SenseHAR model for human activity recognition tasks.
Special Configurations：
    * mlp_hidden_dim:400 （int）dimension for intermediate layers in lstm/linear module of application model
    * k_pool:2 （int）kernel size for K-MaxPooling operation
    * encoder_channels:64 （int）the intermediate out channels for encoder filter

DCL
-------------------------------------------
Citation:
    Ordóñez, F. J. et al., "Deep convolutional and LSTM recurrent neural networks for multimodal wearable activity recognition", Sensors'2016.
    Yuhang Li et al., "Wearable-based Human Activity Recognition with Spatio-Temporal Spiking Neural Networks", Frontiers in Neuroscience'2023.https://github.com/Intelligent-Computing-Lab-Panda/SNN_HAR
    Wei Fang et al., "SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence", Science Advances'2023.https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/rnn.py
Usage:
    Set the --model or -m parameter to: dcl
Number of Parameters:：

Description：
    Implementation of a spike-wise deep convolutional and LSTM recurrent neural network for human activity recognition tasks.
Special Configurations：
    * hidden_dim:128 （int）dimension of the hidden layer for LSTM
    * n_channels:64 （int）number of intermediate channels for convolutional layers

SpikingVGG
-------------------------------------------
Citation:
    Di Yu et al., "EC-SNN: Splitting Deep Spiking Neural Networks for Edge Devices", IJCAI'2024.https://github.com/AmazingDD/EC-SNN/
Usage:
    Set the --model or -m parameter to: spikingvgg9, spikingvgg16
Number of Parameters:：

Description：
    Implementation of a VGG-structured spiking neural network for human activity recognition tasks.
Special Configurations：
    * mlp_ratio:1.0 （float）expand ratio for MLP hidden layers
    * mlp_hidden_dim:2048 （int）hidden dimension for intermediate layers in VGG mlp part

SEWResNet
-------------------------------------------
Citation:
    Wei Fang et al., "Deep residual learning in spiking neural networks", NeurIPS'2021.https://github.com/fangwei123456/Spike-Element-Wise-ResNet
Usage:
    Set the --model or -m parameter to: sewresnet34, sewresnet50
Number of Parameters:：

Description：
    Implementation of a ResNet-structured network for human activity recognition tasks.
Special Configurations：
    * connect_function:'ADD' （str）Optional spike element-wise operation, 'ADD', 'IAND', 'AND'
    * groups:1 （int）grouped convolution
    * width_per_group:64 （int）number of channels for intermediate layers
    * zero_init_residual:False （bool）whether zero-initialize the last BN in each residual branch

MSResNet
-------------------------------------------
Citation:
    Yifan Hu et al., "Advancing spiking neural networks toward deep residual learning", TNNLS'2024.https://github.com/Ariande1/MS-ResNet
Usage:
    Set the --model or -m parameter to: msresnet34, msresnet50
Number of Parameters:：

Description：
    Implementation of a ResNet-structured network for human activity recognition tasks.
Special Configurations：
    * groups:1 （int）grouped convolution (only for bottleneck)
    * base_width:64 （int）number of channels for intermediate layers 

ISpikformer
-------------------------------------------
Citation:
    Changze Lv. et al., "Efficient and Effective Time-Series Forecasting with Spiking Neural Networks", ICML'2024.https://github.com/microsoft/SeqSNN
Usage:
    Set the --model or -m parameter to: ispikformer, spikformer256, spikformer384, spikformer512

Description：
    Implementation of a spike-wise Transformer model for human activity recognition tasks.
Special Configurations：
    * embedding_dim:256 （int）dimension of the embedding
    * num_head:8 （int）number of attention heads
    * depth:4 （int）number of transformer blocks
    * mlp_ratio:4.0 （float）ratio of mlp hidden dimension in transformer

MetaSpikeformer
-------------------------------------------
Citation:
    Man Yao et al., "Spike-driven Transformer V2: Meta Spiking Neural Network Architecture Inspiring the Design of Next-generation Neuromorphic Chips", ICLR'2024.https://github.com/BICLab/Spike-Driven-Transformer-V2
Usage:
    Set the --model or -m parameter to: metaspikeformer256, metaspikeformer384
Number of Parameters:：

Description：
    Implementation of a Transformer-structured spiking neural network model for motion data classification tasks.
    Note that the original Reparameterization Convolution (RepConv) will cause the misconvergence of all models.
    Besides, the two convolution layers are linked directly without any neurons in between, which is not a legal 
    computation logic in neuromorphic chips.Hence, we use a normal linear layer to replace the RepConv part for stable 
    model implementation.
Special Configurations：
    * mlp_ratio:4 （float）expand ratio for hidden dimension in MLP intermediate layers
    * num_heads:8 （int）number of heads for attention output

SpikingResformer
-------------------------------------------
Citation:
    Xinyu Shi et al., "SpikingResformer: Bridging ResNet and Vision Transformer in Spiking Neural Networks", CVPR'2024.https://github.com/xyshi2000/SpikingResformer
Usage:
    Set the --model or -m parameter to: spikingresformer256, spikingresformer384
Number of Parameters:：

Description：
    Implementation of a Transformer-structured spiking neural network model for motion data classification tasks.
Special Configurations：
    * group_size:64 （int）convolutional group
    * mlp_ratio:4 （float）expand ratio for hidden dimension in MLP intermediate layers

QKFormer
-------------------------------------------
Citation:
    Chenlin Zhou et al., "QKFormer: Hierarchical Spiking Transformer using Q-K Attention", NeurIPS'2024.https://github.com/zhouchenlin2096/QKFormer
Usage:
    Set the --model or -m parameter to: qkformer256, qkformer384
Number of Parameters:：

Description：
    Implementation of a Transformer-structured spiking neural network model for motion data classification tasks.
Special Configurations：
    * num_heads:8 （int）number of heads for attention output
    * patch_size:16 （int）patch size for input
    * mlp_ratio:4 （float）expand ratio for hidden dimension in MLP intermediate layers