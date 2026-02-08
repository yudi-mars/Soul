Models
====
For datasets of different modalities (acoustic modality, motion & behavior modality, visual modality, wireless transmission modality), the Soul toolkit provides various model options. 
You can select a model using the --model or -m parameter. The models currently supported by Soul are listed below:

* motion sensing：
   SpikingMLP、SpikingLeNet、SpikingRNN、SpikingConvRNN、DCNN、SenseHAR、DCL、
   ISpikformer、ISpikformer256、ISpikformer384、ISpikformer512、SpikingVGG9、
   SpikingVGG16、SpikingVGG19、SEWResNet34、SEWResNet50、MSResNet34、
   MSResNet50、Spikformer256、Spikformer384、MetaSpikeformer256、MetaSpikeformer384、
   QKFormer256、QKFormer512
* vision sensing：
   SpikingMLP、SpikingLeNet、SpikingRNN、SpikingConvRNN、SpikingVGG5、SpikingVGG9、
   SpikingVGG11、SpikingVGG13、SpikingVGG16、SpikingVGG19、SEWResNet18、SEWResNet50、
   SEWResNet34、MSResNet18、MSResNet34、MSResNet50、Spikformer256、Spikformer384、
   Spikformer512、MetaSpikeformer256、MetaSpikeformer384、MetaSpikeformer512、
   SpikingResformer192、SpikingResformer256、SpikingResformer384、SpikingResformer512、
   QKFormer256、QKFormer384、QKFormer512
* acoustic sensing：
   SpikingMLP、SpikingLeNet、SpikingRNN、SpikingConvRNN、SpikingVGG9、SpikingVGG16、
   SEWResNet18、SEWResNet50、MSResNet18、MSResNet50、SpikingTCN、Spikformer256、
   Spikformer384、Spikformer512、MetaSpikeformer256、MetaSpikeformer384、MetaSpikeformer512、
   SpikingResformer192、SpikingResformer256、SpikingResformer384、SpikingResformer512、
   QKFormer256、QKFormer384、QKFormer512
* wireless sensing：
   SpikingMLP、SpikingLeNet、SpikingRNN、SpikingConvRNN、SpikingTCN、SpikingVGG9、
   SpikingVGG16、SEWResNet34、SEWResNet50、MSResNet34、MSResNet50、Spikformer256、
   Spikformer384、MetaSpikeformer256、MetaSpikeformer384、QKFormer256、QKFormer384、
   SpikingResformer256、SpikingResformer384

.. toctree::
   :maxdepth: 2

   acoustic_en
   motion_en
   vision_en
   wireless_en
