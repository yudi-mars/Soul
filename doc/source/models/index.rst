模型
====
针对不同模态的数据集（声学模态、动作行为模态、视觉模态、无线传输模态），Soul工具包提供了不同的模型选项，您可以通过 --model，-m 参数进行模型选择，目前Soul支持的模型如下：

* 动作识别任务：
   SpikingMLP、SpikingLeNet、SpikingRNN、SpikingConvRNN、DCNN、SenseHAR、DCL、
   ISpikformer、ISpikformer256、ISpikformer384、ISpikformer512、SpikingVGG9、
   SpikingVGG16、SpikingVGG19、SEWResNet34、SEWResNet50、MSResNet34、
   MSResNet50、Spikformer256、Spikformer384、MetaSpikeformer256、MetaSpikeformer384、
   QKFormer256、QKFormer512
* 视觉图像任务：
   SpikingMLP、SpikingLeNet、SpikingRNN、SpikingConvRNN、SpikingVGG5、SpikingVGG9、
   SpikingVGG11、SpikingVGG13、SpikingVGG16、SpikingVGG19、SEWResNet18、SEWResNet50、
   SEWResNet34、MSResNet18、MSResNet34、MSResNet50、Spikformer256、Spikformer384、
   Spikformer512、MetaSpikeformer256、MetaSpikeformer384、MetaSpikeformer512、
   SpikingResformer192、SpikingResformer256、SpikingResformer384、SpikingResformer512、
   QKFormer256、QKFormer384、QKFormer512
* 声音信号任务：
   SpikingMLP、SpikingLeNet、SpikingRNN、SpikingConvRNN、SpikingVGG9、SpikingVGG16、
   SEWResNet18、SEWResNet50、MSResNet18、MSResNet50、SpikingTCN、Spikformer256、
   Spikformer384、Spikformer512、MetaSpikeformer256、MetaSpikeformer384、MetaSpikeformer512、
   SpikingResformer192、SpikingResformer256、SpikingResformer384、SpikingResformer512、
   QKFormer256、QKFormer384、QKFormer512
* 无线信号任务：
   SpikingMLP、SpikingLeNet、SpikingRNN、SpikingConvRNN、SpikingTCN、SpikingVGG9、
   SpikingVGG16、SEWResNet34、SEWResNet50、MSResNet34、MSResNet50、Spikformer256、
   Spikformer384、MetaSpikeformer256、MetaSpikeformer384、QKFormer256、QKFormer384、
   SpikingResformer256、SpikingResformer384

各个模态模型的详细信息，请点击下方查看。

.. toctree::
   :maxdepth: 2

   acoustic
   motion
   vision
   wireless
