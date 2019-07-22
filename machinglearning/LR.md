[TOC]



# LR

LR是线性分类器的一种。主要用于二分类，也可以用于多分类。

![image-20190721210320054](../assets/LR.assert/image-20190721210320054.png)

## 二分类

其模型定义如下：
$$
h_{\theta}(x) = g(\theta^T x) = \frac{1}{1+e^{-\theta^Tx}}
$$




函数的值表示将数据预测为类别1的概率：
$$
p(y=1| x;\theta) = h_{\theta}(x)
$$

$$
p(y=0|x;\theta) = 1 - h_{\theta}(x)
$$

那么对于任意的数据，其预测值为：
$$
p(y|x;\theta) = (h_{\theta}(x))^y \cdot (1-h_{\theta}(x))^{1-y}
$$
那么其似然函数为：
$$
L(\theta) = \prod_{i=1}^{m}p(y^{(i)}|x^{(i)};\theta)  = \prod_{i=1}^{m} (h_{\theta}(x^{(i)}))^{y^{(i)}} \cdot (1-h_{\theta}(x^{(i)}))^{1-y^{(i)}} 
$$
其负对数似然函数为:
$$
-\log L(\theta) = -\sum_{i=1}^{m} y^{(i)} \cdot \log h(x^{(i)}) + (1 - y^{(i)}) \cdot \log (1- h(x^{(i)}))
$$

$$
J(\theta) = -\sum_{i=1}^{m} y^{(i)} \cdot\log g(\theta^T x^{(i)}) + (1 - y^{(i)}) \cdot \log (1 - g(\theta^T x^{(i)}))
$$

省略上标，求其梯度，那么有：
$$
\begin{align}
\nonumber \frac{\partial J(\theta)}{\partial \theta} &= \frac{\partial J(\theta)}{\partial g(\theta^Tx)} \cdot \frac{\partial g(\theta^Tx)}{\partial \theta^T x} \cdot \frac{\partial \theta^Tx}{\partial \theta} \\
&= \left( \frac{-y}{g(\theta^T x)} + \frac{(1-y)}{1-g(\theta^Tx)} \right) \cdot g(\theta^Tx) (1-g(\theta^Tx)) \cdot x \\
&= [-y (1-g(\theta^Tx)) + (1-y) g(\theta^Tx)] \cdot x \\
&= [-y + yg(\theta^Tx) + g(\theta^Tx) - yg(\theta^Tx)] \cdot x \\
&= [g(\theta^Tx) - y]\cdot x
\end{align}
$$
对于m个数据的梯度的总和为：
$$
\frac{\partial J(\theta)}{\partial \theta} = \sum_{i=1}^{m} (g(\theta^Tx^{(i)}) - y^{(i)}) \cdot x^{(i)}
$$
写成矩阵的形式为：
$$
X^T
$$










> LR就是只有一个神经元的神经网络， 且该神经元的激活函数为Sigmoid。



## 多分类

> K分类的LR算法可以看做是具有K个神经元的神经网络，且这些神经元的激活函数为Softmax。



## 参考

