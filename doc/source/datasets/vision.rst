Vision 数据集
====================

以下为Soul工具包支持的视觉/图像相关数据集：

CIFAR10/100
-------------------------------------------
引用：
    https://www.cs.utoronto.ca/~kriz/learning-features-2009-TR.pdf
下载地址：
    https://www.cs.toronto.edu/~kriz/cifar.html
详细信息：

    - RGB数据大小： 3 x 32 x 32
    - 类别数： 10\\100
    - 类别： CIFAR10有 plane, car, bird, cat, deer, dog, frog, horse, ship, truck; CIFAR100有 100 个不同类别 
    - 训练集大小： 50000
    - 测试集大小： 10000

Tiny-ImageNet
-------------------------------------------
引用：
    https://ieeexplore.ieee.org/abstract/document/5206848/
下载地址：
    https://www.kaggle.com/c/tiny-imagenet
详细信息：

    - RGB数据大小： 3 x 224 x 224
    - 类别数： 200
    - 类别： goldfish，European fire salamander，bullfrog...
    - 训练集大小： 100000
    - 测试集大小： 10000

CIFAR10-DVS
-------------------------------------------
引用：
    https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00309/full
下载地址：
    https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671
详细信息：

    - DVS数据大小： 2 x receptor_size x receptor_size
    - 类别数： 10
    - 类别： plane, car, bird, cat, deer, dog, frog, horse, ship, truck
    - 训练集大小： 50000
    - 测试集大小： 10000

DVS-Gesture
-------------------------------------------
引用：
    https://ieeexplore.ieee.org/document/8100264
下载地址：
    https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794
详细信息：

    - DVS数据大小： 2 x receptor_size x receptor_size
    - 类别数： 11
    - 类别： 
        Hand Clapping, Right Hand Wave, Left Hand Wave, Right Arm Clockwise,
        Left Arm Clockwise, Arm Roll, Left Arm Counter Clockwise, Right Arm Counter Clockwise,
        Both Arms Clockwise, Both Arms Counter Clockwise, invalid data
    - 训练集大小： 1176
    - 测试集大小： 288

NCaltech101
-------------------------------------------
引用：
    https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2015.00437/full
下载地址：
    https://drive.google.com/drive/folders/1sY91hL_iHnmfRXSTc058bfZ0GQcEC6St
详细信息：

    - DVS数据大小： 2 x receptor_size x receptor_size
    - 类别数： 101
    - 类别： 100 objection 类别 and 1 background class 
    - 训练集大小： 7000
    - 测试集大小： 1200

NMNIST
-------------------------------------------
引用：
    https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2015.00437/full
下载地址：
    https://www.kaggle.com/datasets/surya77/nmnist
详细信息：

    - DVS数据大小： 2 x receptor_size x receptor_size
    - 类别数： 10
    - 类别： 10 numbers 
    - 训练集大小： 60000
    - 测试集大小： 10000

MNIST
-------------------------------------------
引用：
    https://scholar.google.com/citations?view_op=view_citation&hl=fr&user=WLN3QrAAAAAJ&citation_for_view=WLN3QrAAAAAJ:6fs0NoO7GbkC
下载地址：
    http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/
详细信息：

    - GrayScale size： 1 x 32 x 32
    - 类别数： 10
    - 类别： t-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot
    - 训练集大小： 60000
    - 测试集大小： 10000

FashionMNIST
-------------------------------------------
引用：
    https://arxiv.org/abs/1708.07747
下载地址：
    https://www.kaggle.com/datasets/surya77/nmnist
详细信息：

    - DVS数据大小： 2 x receptor_size x receptor_size
    - 类别数： 10
    - 类别： 10 numbers 
    - 训练集大小： 60000
    - 测试集大小： 10000

SVHN
-------------------------------------------
引用：
    https://experimentationground.wordpress.com/2016/09/26/digit-recognition-from-google-street-view-images/
下载地址：
    https://www.kaggle.com/datasets/stanfordu/street-view-house-numbers
详细信息：

    - 灰度图数据大小： 1 x 32 x 32
    - 类别数： 10
    - 类别： 10 numbers 
    - 训练集大小 : 73257
    - 测试集大小 : 26032 
    - 额外数据大小 : 531131