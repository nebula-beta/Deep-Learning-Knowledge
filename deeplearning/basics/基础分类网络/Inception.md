[TOC]

# Inception总结





Inception v1的网络，将1x1，3x3，5x5的conv和3x3的pooling，stack在一起，一方面增加了网络的width，另一方面增加了网络对尺度的适应性；



v2的网络在v1的基础上，进行了改进，一方面了加入了BN层，减少了Internal
Covariate Shift（内部neuron的数据分布发生变化），使每一层的输出都规范化到一个N(0, 
1)的高斯，另外一方面学习VGG用2个3x3的conv替代inception模块中的5x5，既降低了参数数量，也加速计算；



v3一个最重要的改进是分解（Factorization），将7x7分解成两个一维的卷积（1x7,7x1），3x3也是一样（1x3,3x1），这样的好处，既可以加速计算（多余的计算能力可以用来加深网络），又可以将1个conv拆成2个conv，使得网络深度进一步增加，增加了网络的非线性，还有值得注意的地方是网络输入从__224x224__变为了__299x299__，更加精细设计了35x35/17x17/8x8的模块；



v4研究了Inception模块结合Residual
Connection能不能有改进？发现ResNet的结构可以极大地加速训练，同时性能也有提升，得到一个Inception-ResNet 
v2网络，同时还设计了一个更深更优化的Inception v4模型，能达到与Inception-ResNet v2相媲美的性能。













## 参考

[谷歌Inception网络中的Inception-V3到Inception-V4具体作了哪些优化？ - 徐亮的回答 - 知乎](https://www.zhihu.com/question/50370954/answer/138938524)



[深度学习家族盘点-inception-v4-和inception-resnet未来走向何方](http://nooverfit.com/wp/inception%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%B6%E6%97%8F%E7%9B%98%E7%82%B9-inception-v4-%E5%92%8Cinception-resnet%E6%9C%AA%E6%9D%A5%E8%B5%B0%E5%90%91%E4%BD%95%E6%96%B9/)

