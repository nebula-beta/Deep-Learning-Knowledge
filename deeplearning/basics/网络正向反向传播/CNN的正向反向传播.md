[TOC]

# CNN的正向反向传播



## CNN的正向传播



#### 卷积层

如果用标准的形式实现卷积，则要用循环实现，依次执行乘法和加法运算。为了加速，可以将卷积操作转化成矩阵乘法实现，以充分利用GPU的并行计算能力。

整个过程分为以下3步：

1. 将待卷积图像、卷积核转换成矩阵
2. 调用通用矩阵乘法GEMM函数对两个矩阵进行乘积
3. 将结果矩阵转换回图像

假设特征图的通道数为$n_{in}$，卷积核的大小为$s \cdot s$，卷积核的数量为$n_{out}$。



将卷积核的每个通道中的$s \cdots s$个权重拉直成__一行__，由于每个卷积核有$n_{in}$个通道，将这$n_{in}$通道拉直的结果放在__一行__中，形成权重矩阵的__一行__。最终的权重矩阵有多少行取决于卷积核的个数有多少个。



将特征图中原本应与卷积核进行卷积的每个patch的像素拉直成__一列__，由于特征图有$n_{in}$个通道，将这$n_{in}$个通道拉直的结果放在__一列__中，形成输入矩阵的__一列__。最终的输入矩阵有多少列取决于特征图中有多少个patch与卷积核进行卷积。
$$
n_{out}
\left\{ \left[ \begin{array}{ccc}

\overbrace{\begin{array}{cccc}k_1^1 &  k_2^1  & \cdots & k_{s\cdot s}^1,\end{array}}^{channel \ 1} & 
\overbrace{\begin{array}{cccc}k_1^2 &  k_2^2  & \cdots & k_{s\cdot s}^2,\end{array}}^{channel \ 2 } &
\overbrace{\begin{array}{cccc}k_1^{n_{in}} &  k_2^1  & \cdots & k_{s\cdot s}^{n_{in}},\end{array}}^{channel \ n_{in}} \\

\vdots & \vdots & \vdots\\
\end{array} \right] \right.


\left[\begin{array}{ccc}
 \left\{
\begin{array}{c}
x_1^1 \\
x_2^1 \\
\vdots \\
x_{s\cdot s}^1 \\
\end{array}
\right. \\

\left\{
\begin{array}{c}
x_1^2 \\
x_2^2 \\
\vdots \\
x_{s\cdot s}^2 \\
\end{array}
\right.  & \cdots & \cdots \\

\vdots \\
\left\{
\begin{array}{c}
x_1^{n_{in}} \\
x_2^{n_{in}} \\
\vdots \\
x_{s\cdot s}^{n_{in}} \\
\end{array}
\right. \\
\end{array} \right]
$$


之后，便可以简写为
$$
KX
$$
也就是说卷积操作可以通过`im2col`操作转变为矩阵乘法，从而保持和全连接神经网络的一致性。

#### 池化层

无论是最大池化还是平均池化，其操作都比较简单。



## CNN的反向传播

#### 卷积核的反向传播

将卷积核的正向传播转换成矩阵乘法之后，那么其反向传播和全连接神经网络没有什么区别，所以也就不推导了。



#### 池化层的反向传播

无论max pooling还是mean pooling，都没有需要学习的参数。因此，在卷积神经网络的训练中，Pooling层需要做的仅仅是将误差项传递到上一层，而没有梯度的计算。

1. max pooling层：对于max pooling，下一层的梯度项的值会原封不动的传递到上一层对应区块中的最大值所对应的神经元，而其他神经元的梯度项的值都是0；
2. mean pooling层：对于mean pooling，下一层的梯度项的值会平均分配到上一层对应区块中的所有神经元。


$$
Z^1 = KX + b \\
A^1 = f(Z^1) \\
P^1 = Pooling(A^1) \\
...
$$


池化层的反向传播就是在求得导数$\frac{\partial L}{\partial P^l}$后，怎么将该导数通过池化层传播到$A_1$。

#### 最大池化反向传播

如果是最大池化（池化的大小为$s\cdot s$），在正向传播时需要记住最大值的位置。在反向传播，下一层的误差项的值会原封不动的传递到上一层对应区块中的最大值所对应的神经元，而其他神经元的误差项的值都是0。也就是说会把一个池化层一个元素的梯度扩充成一个$s\cdot s$大小的块，最大值位置的元素设为$\frac{\partial L}{\partial P_{ij}^l}$，其它位置全部置为0。
$$
\left[\begin{array}{ccc}
0 & 0 & 0\\
0 & \cdots & \cdots \\
\frac{\partial L}{\partial P_{ij}^l} & \cdots & 0\\
\end{array}\right]
$$

#### 平均池化反向传播

对于最大池化层的反向传播，其数学原理可能不明显。但是对于平均池化层的反向传播，其数学原理就比较好理解了。

比如池化层的一次池化操作如下：
$$
P_{1}^{1} = \frac{1}{s \cdot s} \cdot(A_{11}^{1} +  A_{12}^{1} + \cdots A_{ss}^{1})
$$
那么反向传播时有：
$$
\frac{\partial L}{\partial A_{11}^{1}} 
= 
\frac{\partial L}{\partial P_{1}^{1}} 
\cdot 
\frac{\partial P_{1}^{1}}{\partial A_{11}^{1}}
=
\frac{\partial L}{\partial P_{1}^{1}} 
\cdot
\frac{1}{s\cdot s}
$$


也即有：
$$
\left[\begin{array}{ccc}
\frac{\partial L}{\partial P_{1}^{1}} 
\cdot
\frac{1}{s\cdot s} & \frac{\partial L}{\partial P_{1}^{1}} 
\cdot
\frac{1}{s\cdot s} & \frac{\partial L}{\partial P_{1}^{1}} 
\cdot
\frac{1}{s\cdot s}\\
\frac{\partial L}{\partial P_{1}^{1}} 
\cdot
\frac{1}{s\cdot s} & \cdots & \cdots \\
\frac{\partial L}{\partial P_{1}^{1}} 
\cdot
\frac{1}{s\cdot s} & \cdots & \frac{\partial L}{\partial P_{1}^{1}} 
\cdot \frac{1}{s\cdot s}\\
\end{array}\right]
$$


## 参考

[反向传播算法推导-卷积神经网络 - 简书](https://www.jianshu.com/p/8ad58a170fd9)



