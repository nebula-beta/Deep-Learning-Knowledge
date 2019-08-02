[TOC]

# 特征值分解-奇异值分解-主成分分析

分析了EVD，SVD，PCA之间到底有什么关系

EVD只能用于方阵，并且要求方阵有$N$个线性无关的特征值向量；

SVD则能够用于任意矩阵；

PCA则用来对数据进行去相关性和降维，其求解用到了EVD或SVD。



# 特征值分解





## 普通矩阵的特征值分解



矩阵$A$为$N \times N$的__方阵__，且有$N$个__线性无关__的特征向量，那么有：
$$
A \left[ \begin{array}{cccc} p_1 & p_2 & \cdots & p_N\end{array} \right] 
=  
\left[ \begin{array}{cccc} p_1 & p_2 & \cdots & p_N\end{array} \right] 
\left[ \begin{array}{cccc} \lambda_1 & 0 & \cdots & 0 \\ 0 & \lambda_2 & \cdots & 0 \\  \vdots & \vdots & \cdots & \vdots \\ 0 & 0 & \cdots & \lambda_N \end{array} \right]
$$
所以有：
$$
AP = P \Lambda
$$
因为线性无关，所以矩阵$P$可逆，所以有：
$$
A = P \Lambda P^{-1}
$$
这就是特征值分解，或者也可以叫做是矩阵对角化。



## 实对称矩阵的特征值分解

特别的，若矩阵是实对称矩阵，那么可以找到$N$个正交的特征向量，从而有：
$$
AP = P \Lambda
$$

$$
A = P \Lambda P^T
$$

这就是实对称矩阵的特征值分解，或者也可以叫做__正交对角化__。

那么为什么实对称矩阵有$N$个正交的特征向量呢，那么就要分析一下实对称矩阵的性质了。



### 实对称矩阵的性质

#### 不同特征值对应的特征向量正交

设$p_1, p_2$分别为$\lambda_1, \lambda_2$所对应的特征向量，且$\lambda_1 != \lambda_2$
$$
A p_1 = \lambda_1 p_1
$$

$$
Ap_2 = \lambda_2 p_2
$$

那么有:
$$
(Ap_2)^Tp_1 = (\lambda_2 p_2)^T p_1  = \lambda_2 p_2^T p_1 \\
(Ap_2)^T p_1 = p_2^TA^Tp_1 = p_2^T A p_1 =  p_2^T \lambda_1 p_1 = \lambda_1 p_2^Tp_1
$$
从而有:
$$
\lambda_2p_2^T p_1 = \lambda_1p_2^Tp_1
$$
由于$\lambda_1 \ne \lambda_2$，所以就有$p_2^T p_1 = 0$，从而推出不同特征值所对应的特征向量正交。



#### 相同特征值对应的特征向量线性无关

特征值重数和线性无关特征向量的个数

比如 |A-λE| = (1-λ)^2 (2+λ)^3
特征值是1,-2.则 特征值1的重数为2,特征值-2的重数为3



由于实对称矩阵特征值的重数等于线性无关的特征向量的个数。假设$\lambda_i$的重数大于1（设为$c_i$），那么有：
$$
(A - \lambda_i I)p_i = 0
$$
由于$\lambda_i$有$c_i$个线性无关的特征向量，因此这$c_i$个线性无关的特征向量构成了$A -  \lambda_i I$的零空间。也就是说，这个子空间中的任意一个向量都是矩阵$A$对应特征值$\lambda_i$的特征向量。因此，我们可以取这个子空间的一组正交基作为$A$对应特征值$\lambda_i$的特征向量。



假设$p$为该子空间中的向量，那么有:
$$
p = (a_1 p_{i,1} + a_2p_{i,2)} + \cdots + a_{i,c_i}p_{i,c_i})
$$
从而有：
$$
\begin{align}
\nonumber Ap &= A(a_1 p_{i,1} + a_2 p_{i,2} + \cdots + a_{c_i} p_{i,c_i}) \\
\nonumber  &= (\lambda_i a_1 p_{i,1} + \lambda_i a_2 p_{i,2} + \cdots + \lambda_i a_{c_i} p_{i,c_i}) \\
\nonumber  &= \lambda_i ( a_1 p_{i,1} + a_2 p_{i,2} + \cdots +  a_{c_i} p_{i,c_i}) \\
\nonumber &= \lambda_i p

\end{align}
$$
由于$Ap = \lambda_i p$，所以该子空间中的向量都是矩阵$A$关于特征值$\lambda_i$的特征向量。



[为什么实对称矩阵的特征向量正交化并单位化后仍为原矩阵的特征向量? - george zhou的回答 - 知乎](https://www.zhihu.com/question/61876453/answer/192018392)



## 奇异值分解

特征值分解需要矩阵是方阵，并且有$N$个线性无关的特征向量才能够分解。奇异值则是能够对任意的矩阵进行分解。



具体公式，看下面得链接。

[奇异值分解](./奇异值分解.html)





所以，奇异值分解至少有下面这两个作用:

1. 可以对任意大小的矩阵进行分析，然后利用奇异值的大小对其进行压缩；
2. 若我们要对矩阵$AA^T$或者$A^TA$的特征值和特征值向量，可以通过求$A$或者$A^T$的奇异值分解后，进一步处理后得到$AA^T$或者$A^TA$的特征值和特征值向量。（SVD分解比特征值分解快吗？因此是吧，如果进行特征值分解，需要进行矩阵乘法求出$AA^T$）













## 主成分分析

[PCA的数学原理(转) - 郑申海的文章 - 知乎](https://zhuanlan.zhihu.com/p/21580949)

先看完上面链接的内容，再来看下面自己总结的内容。



PCA有两个作用

1. 去除相关性，也即去除不同特征之间的相关性（线性相关）
2. 降维





### 去相关性



__数据的协方差矩阵__:

假设有$m$条$n$维的数据，我们可以将其排成一个的矩$n$行$m$列的矩阵：
$$
X= \left[ \begin{array}{cccc} 
x_{1,1} & x_{2,1} & \cdots & x_{m,1} \\
x_{1,2} & x_{2,2} & \cdots & x_{m,2} \\
\vdots & \vdots &  \cdots & \vdots\\
x_{1,n} & x_{2,n} & \cdots & x_{m,n} \\
\end{array} \right] \\ \\
$$
其中，每一列都是一条数据。并且，我们可以对数据进行零均值化，也就是对矩阵$X$的每一行（一个特征）求均值，然后让该行的每个值都减去该均值。

现在，我们假设矩阵$X$已经是经过均值化处理了，那么现在
$$
\nonumber \begin{align}
X X^T &=
\left[ \begin{array}{cccc} 
x_{1,1} & x_{2,1} & \cdots & x_{m,1} \\
x_{1,2} & x_{2,2} & \cdots & x_{m,2} \\
\vdots & \vdots &  \cdots & \vdots\\
x_{1,n} & x_{2,n} & \cdots & x_{m,n} \\
\end{array} \right] 
\left[ \begin{array}{cccc} 
x_{1,1} & x_{1,2} & \cdots & x_{1,n} \\
x_{2,1} & x_{2,2} & \cdots & x_{2,n} \\
\vdots & \vdots &  \cdots & \vdots\\
x_{m,1} & x_{m,2} & \cdots & x_{m,n} \\
\end{array} \right] \\  \nonumber \\

\nonumber  &=
\left[ \begin{array}{cccc} 
\sum_{i=1}^{m} x_{i, 1} x_{i, 1}  &\sum_{i=1}^{m} x_{i, 1} x_{i, 2} &\cdots & \sum_{i=1}^{m} x_{i, 1} x_{i, m} \\
\sum_{i=1}^{m} x_{i, 2} x_{i, 1}  &\sum_{i=1}^{m} x_{i, 2} x_{i, 2} &\cdots & \sum_{i=1}^{m} x_{i, 2} x_{i, m} \\
\vdots & \vdots &  \cdots & \vdots\\
\sum_{i=1}^{m} x_{i, m} x_{i, 1}  &\sum_{i=1}^{m} x_{i, m} x_{i, 2} &\cdots & \sum_{i=1}^{m} x_{i, m} x_{i, m} \\
\end{array} \right] \\ \nonumber \\

\end{align}
$$


第一个特征的方差为（特征已经进行了零均值化）：
$$
\begin{align}
\nonumber Var(f_1) = & \frac{1}{m} \sum_{i=1}^{m} (x_{i, 1} - \mu_1) \cdot (x_{i, 1} - \mu_1) \\
\nonumber  &= \frac{1}{m} \sum_{i=1}^{m} x_{i, 1}  \cdot x_{i, 1} \\
\end{align}
$$
第一个特征和第二个特征之间的协方差为（特征已经进行了零均值化）：
$$
\begin{align}
\nonumber  Cov(f_1, f_2) = & \frac{1}{m} \sum_{i=1}^{m} (x_{i, 1} - \mu_1) \cdot (x_{i, 2} - \mu_2) \\
\nonumber  &= \frac{1}{m} \sum_{i=1}^{m} x_{i, 1}  \cdot x_{i, 2} \\
\end{align}
$$
所以矩阵
$$
C= \frac{1}{m} X X^T
$$
就是特征之间的协方差矩阵。









__变换数据后的协方差矩阵__：

我们希望投影（基变后）不同特征之间的相关性（协方差）为0。

假设存在一个投影（基变换）矩阵$P$，对数据矩阵$X$进行变换，使得变换过后的数据矩阵
$$
Y = PX
$$
的不同特征之间的不相关的。那么也就是期望$Y$的协方差矩阵除了对角线元素外，其它元素都是0。
$$
\begin{align}
\nonumber  D &= \frac{1}{m} (PX) (PX)^T \\
\nonumber &= \frac{1}{m} PXX^TP^T \\
\nonumber &= P(\frac{1}{m}XX^T)P^T \\
\nonumber &= P C P^T \\
\end{align}
$$

> $E(a_1 + a_2 + \cdots  ) = E(a_1) + E(a_2) + \cdots = 0$，若随机变量的均值为0，那么其和的均值也是0。
>
> 由于矩阵$X$的每个元素的均值都是0，所以$PX$的每个元素的均值也是0。也就是$Y$也是零均值化的，所以可以使用上面的方法计算协方差矩阵。


$$
D = PCP^T
$$
很容易看出来，这是一个正交对角化的公式，并且由于矩阵$C$是实对称矩阵，所以该公式是成立的。

* 矩阵$P$是协方差矩阵$C=\frac{1}{m}XX^T$的$n$个正交的特征向量所组成的矩阵。
* 矩阵$D$是对角矩阵，其对角线上的元素为协方差矩阵$C= \frac{1}{m} XX^T$的$n$个特征值。



所以，通过寻找协方差矩阵$C = \frac{1}{m}XX^T$的$n$个正交的特征向量，就能够对数据矩阵$X$去相关性，从而得到新的数据矩阵
$$
Y = PX
$$

> 需要注意，对角化的公式为：
> $$
> \Lambda = P^T AP
> $$
> $P$为正交矩阵，其中每一列是一个基向量，任意两列是正交的。
>
> $P^T$为正交矩阵，其中每一行是一个基向量。
>
> 
>
> 而PCA中对角化公式为：
> $$
> D = PCP^T
> $$
> 其中$P$为正交矩阵，其中每一行是一个基向量，也即每一行是一个特征向量。
>
> 
>
> 其实你会发现
> $$
> D^T = (PCP^T)^T = P^TC^TP  = P^TCP
> $$
> 从而有
> $$
> D^T = D = P^TCP = PCP^T
> $$
> 
>
> 也就是说，转置加在哪里无所谓（只对于对称矩阵的正交对角化有这样的性质）。
>
> * 左边的正交矩阵是行向量为基向量（特征向量）;
> * 右边的正交矩阵是列向量为基向量（特征向量）。



### 降维

进一步的，我们可以对数据进行降维。


$$
\begin{align}
\nonumber Y=PX 
&= 
\left[
\begin{array}{c}
p_1 \\
p_2 \\
\cdots \\ 
p_n \\
\end{array}
\right]
\left[
\begin{array}{cccc}
x_1 & x_2 & \cdots & x_m \\
\end{array}
\right] \\ 
\nonumber\\
\nonumber  &=

\left[
\begin{array}{cccc}
p_1^T x_1 & p_1^T x_2 & \cdots & p_1^T x_m\\
p_2^T x_1 & p_2^T x_2 & \cdots & p_2^T x_m\\
\vdots & \vdots & \cdots & \vdots \\
p_n^T x_1 & p_n^T x_2 & \cdots & p_n^T x_m\\
\end{array}
\right]
\end{align}
$$


可以看到，正交矩阵$P$对数据矩阵$X$进行了基变换，得到了新的数据矩阵$Y$，其中每一行都代表这一个新的特征。__但是，要注意的是，这些新的特征的重要性是不相同的，并且是能够看出来的。__
$$
\begin{align}
\nonumber D &= PCP^T \\
\nonumber D &= \frac{1}{m}(PX)(PX)^T \\
\end{align}
$$
矩阵$D$是对角矩阵，其对角线上的元素为协方差矩阵$C= \frac{1}{m} XX^T$的$n$个特征值（这是其如何计算的）。每个特征值的含义为数据矩阵$Y = PX$的每个特征的方差（这是特征值在这里的含义）。因此有：

* 矩阵$D$的第$i$个特征值大，也即$Y=PX$的第$i$个特征（第$i$行）的特征是重要的，含有的信息多；
* 矩阵$D$的第$i$个特征值小，也即$Y=PX$的第$i$个特征（第$i$行）的特征是不重要的，含有的信息少；



因此，我们可以根据矩阵$D$对角线元素的大小，来删除$Y=PX$中那些不重要的特征，也即删除$Y$的某些行，也即删除$P$的某些行。若我们选择了矩阵$D$的前$k$个特征值，那么我们只需选择矩阵$P$的前$k$行作为新的$P$，从而使得$PX$能够将数据的维度从$n$维，降低到$k$维。





### PCA步骤

设有$m$条$n$维的数据。

1. 将原始数据按列排成$n$行$m$列的矩阵$X$；
2. 对矩阵$X$的每一行进行零均值化；
3. 求出特征之间的协方差矩阵$C = \frac{1}{m} XX^T$；
4. 求出协方差矩阵所对应的特征值及特征向量；
5. 将特征向量按对应特征值大小从上到下排成矩阵，取其前$k$行组成矩阵$P$；
6. $Y = PX$，即为降维到$k$维后的数据。



[PCA的本质----特征值分解](https://blog.csdn.net/sunshine_in_moon/article/details/51513880)

# 附录

## 正交矩阵

正交矩阵的行向量之间正交，正交矩阵的列向量之间正交。
$$
AA^T = A^TA = 0
$$

* 由$AA^T$可知行向量正交；
* 由$A^TA$可知列向量正交；



## 齐次方程最小二乘解

[齐次方程最小二乘解](./齐次方程组最小二乘解.html)



