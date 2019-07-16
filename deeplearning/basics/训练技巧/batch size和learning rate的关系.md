[TOC]

# batch size和learning rate的关系





## 增大batch size减小梯度的方差

假设batch size的大小为m，那么对于一个mini-batch，其loss为：
$$
L = \frac{1}{m}\sum_{i=1}^{m}L(x_i, y_i)
$$
其梯度为：
$$
g = \frac{1}{m} \sum_{i=1}^{m}g(x_i, y_i)
$$


那么整个mini-batch的梯度方差为：
$$
Var(g) = Var(\frac{1}{m}\sum_{i=1}^{m} g(x_i, y_i))
$$

$$
Var(g) = \frac{1}{m^2} Var(g(x_1, y_1) + g(x_2, y_2) + \cdots + g(x_m, y_m))
$$

$$
Var(g) = \frac{1}{m} Var(g(x_i, y_i))
$$

由于每个样本都是随机中样本集采样的，满足i.i.d假设，因此每个样本梯度的方差都相等，都为$Var(g(x_i, y_i))$。可以看到，随着mini-batch的样本数量m的增大，mini-batch梯度的方差也会降低。至于mini-batch梯度的期望则是不变的：
$$
E(g) = E(\frac{1}{m} \sum_{i=1}^{m}g(x_i,y_i)) = E(g(x_i, y_i))
$$
所以说，增大batch size会降低梯度的噪声，使得梯度更加稳定。



## 增大batch size的同时增大learning rate

从两个角度来解释为什么增大batch size之后要增大learning rate。



__使得更新量相当：__

对于样本数量为$N$，假设使用mini-batch为$m$时i一个epoch能够更新$k$次，但是将mini-batch调大到$2m$后，一个epoch就只能更新$\frac{k}{2}$次，这使得更新量降低了。解决方法就是：

1. 将max_epoch数调大到原来的两倍，这样就能使得更新量相当。
2. 将learning rate调大原来的两倍，这样也能够使得更新量相当，并且由于每个mini-batch的梯度方差减小了，也急方差更准了，那么在更准的梯度上使得更大的学习率也没什么问题。



__使得梯度的方差不变:__

增大mini-batch的一个好处就是使得BN的统计量计算更加准确，另外一个则是降低了mini-batch梯度的方差，但是方差小并一定是好的，虽然梯度更准了，但是方差（噪声）大的梯度有利于神经网络跳出局部极小值点或者是鞍点。所以，从这个角度来考虑的话，在增大mini-batch时，又要保持梯度的方差和baseline梯度的方差相同，那么只有加大学习率了：
$$
\frac{1}{m}Var(\sqrt{m} \times lr \times g(x_i, y_i)) = Var(lr \times g(x_i, y_i))
$$
也就是说lr可以增加$\sqrt{m}$倍。





增大batch size的同时增大learning rate，但是这只是一个指导策略，具体增大多少倍，还是需要实际进行实验，因为learning rate过分增大的话会引起震荡。



## 为什么令batch size为1

当模型训练到尾声，想更精细化地提高成绩（比如论文实验/比赛到最后），有一个有用的trick，就是设置batch size为1，即做纯SGD，慢慢把error磨低。



因为batch size为1的，梯度的方差（噪声）大，有利于跳出局部极小值点，也就是说期望通过引入噪声的方式从而达到一个更优秀的极小值点。





## 参考

[如何理解深度学习分布式训练中的large batch size与learning rate的关系？](https://www.zhihu.com/question/64134994/answer/216895968)

[batch_size是否越大越好？](https://zhuanlan.zhihu.com/p/54441115)

[为什么batch size增大之后，可以增加学习率而不发散？](https://www.zhihu.com/question/305782724)

