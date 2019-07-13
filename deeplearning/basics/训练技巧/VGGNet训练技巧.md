# 训练技巧


## VGG训练/测试技巧

VGG规定了图像的输入大小为224x224


### Single-scale Training

让S为训练图像的短边，也就是说将原始图像的短边resize到S，然后长边安装比例进行resize。

然后在resized后的图像上随机crop出224x224的图像用来训练。

应该让S >> 224，这样子crop才会包含图像的一小部分，包含小物体或者物体的一部分，也就让神经网络更加robust，看到物体的一部分就能够识别出来。

Single-scale training表示训练期间S是固定的。论文里用了S=256和S=384这两个尺度。

在测试时，实验表明训练时S=384的效果略微由于S=256的效果。


### Multi-scale Training
训练是， S在区间$[S_{min}, S_{max}]$中随机取值，然后将图像resized到对应的尺度，然后再进行crop。

这个训练技巧在数据增强中叫做scale jittering。


### Single-scale Evaluation

测试时也有一个尺度Q，将图像resized到该尺度，然后进行crop。

对于Single-scale Training的Evaluation，自然是让Q=S
对应Multi-scale Training的Evaluation，则是让$Q=0.5(S_{min} + S_{max})$.


### Multi-scale Evaluation

也即多选几个Q，然后将图像resize到该尺度下，然后进行crop。
在这些crop上进行预测，然后对结果进行融合。


对于Single-scale Training的Evaluation，令Q={S-32, S, S+32}
对于Multi-scale Training的Evaluation，则是令$Q=\{S_{min}, 0.5(S_{min} + S_{max}), S_{max}\}$


### Multi-crop Evaluation

对于每个尺度Q，对其随机裁剪出若干个crop，在这些crop上进行预测，然后对结果进行融合。


### Dense Evaluation
[理解为什么要将全连接层转化为卷积层 - LiuYuZhan - 博客园](https://www.cnblogs.com/liuzhan709/p/9356960.html)



### Multi-crop Evaluation和Dense Evaluation的不同


multi-crops相当于对于dense evaluatio的补充，原因在于，两者在边界的处理方式不同：multi-crop相当于padding补充0值，而dense evaluation相当于padding补充了相邻的像素值，并且增大了感受野


[在VGG网络中dense evaluation 与multi-crop evaluation两种预测方法的区别以及效果 - 小C的博客 - CSDN博客](https://blog.csdn.net/C_chuxin/article/details/82832229)