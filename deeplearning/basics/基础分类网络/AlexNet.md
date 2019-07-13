# AlexNet

__paper__ : [link](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

## AlexNet的特点

* AlexNet是在LeNet的基础上加深了网络的结构，学习更丰富更高维的图像特征。AlexNet的特点：

* 更深的网络结构
* 使用层叠的卷积层，即卷积层+卷积层+池化层来提取图像的特征
* 使用Dropout抑制过拟合
* 使用数据增强Data Augmentation抑制过拟合
* 使用Relu替换之前的sigmoid的作为激活函数
* 多GPU训练


## AlexNet参数

__首次提出了分组卷积的操作？？？__



| layer | input size | kernel size, stride | filter number | output size | formular          |      |
| ----- | ---------- | ------------------- | ------------- | ----------- | ----------------- | ---- |
| conv1 | 227x227x3  | 11x11x3, 4          | 96            | 55x55x96    | (227-11)/4+1=55   |      |
| pool1 | 55x55x96   | 3x3, 2              |               | 27x27x96    | (55-3)/2+1=27     |      |
| conv2 | 27x27x96   | 5x5x48, 1           | 256           | 27x27x256   | (27-5+2x2)/1+1=27 |      |
| pool2 | 27x27x256  | 3x3, 2              |               | 13x13x256   | (27-3)/2+1=13     |      |
| conv3 | 13x13x256  | 3x3x256, 1          | 384           | 13x13x384   | (13-3+2x1)/1+1=13 |      |
| conv4 | 13x13x384  | 3x3x192,1           | 384           | 13x13x384   | (13-3+2x1)/1+1=13 |      |
| conv5 | 13x13x384  | 3x3x192,            | 256           | 13x13x256   | (13-1+2x1)/1+1=13 |      |
| pool5 | 13x13x256  | 3x3, 2              |               | 6x6x256     | (13-3)/2+1=6      |      |

如果input size和kernel size的通道数对应不上，那么说明特征图被分成了两组，单独在自己的GPU上进行特征提取操作。
如果input size和kernel size的通道数对应上了，那么说明了两组特征图被合起来，分别送到每个GPU上进行特征提取操作。


之后就会将6x6x256的特征图通过全连接层，得到4096个输出结果。
第6层输出的4096个数据与第7层的4096个神经元进行全连接，然后经由ReLU和Dropout进行处理后生成4096个数据。


## 数据增强


## 参考

[AlexNet详细解读 - 学思行仁的博客 - CSDN博客](https://blog.csdn.net/qq_24695385/article/details/80368618)