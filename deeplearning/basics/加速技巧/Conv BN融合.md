[TOC]

# Conv BN融合



Conv层：
$$
z = w * x + b
$$
BN层：
$$
y = \frac{x - mean}{\sqrt{var}} \cdot \gamma + \beta
$$


将Conv层的公式带入到BN层的公式中，那么就可以达到融合两层的目的了：
$$
\begin{align}
\nonumber y &= \frac{w*x+b - mean}{\sqrt{var}} \cdot \gamma + \beta \\
\nonumber  &=  \frac{w*x}{\sqrt{var}}\cdot \gamma + \left( \frac{b-mean}{\sqrt{var}} \cdot \gamma + \beta
 \right)
\end{align}
$$


从而有：
$$
w_{new} = \frac{w*x}{\sqrt{var}} \cdot \lambda
$$

$$
b_{new} = \frac{b-mean}{\sqrt{var}} \cdot \gamma + \beta
$$



训练结束之后，我们可以创建一个新的模型，该模型将所有的BN层去掉（融合进了卷积层），而卷基层的参数则有原模型的卷基层+BN层的参数计算得到。

将BN层如何到卷积层中，就相当于对卷积核进行一定的修改，并没有增加卷积层的计算量，同时整个BN层的计算量都省去了。





## 实现

```python
#!/usr/bin/python2.7
#coding:utf-8


import torch
import torch.nn as nn
import torchvision as tv


class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        # print("Dummy, Dummy.")
        return x


def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


def fuse_module(m):
    children = list(m.named_children())
    c = None
    cn = None

    for name, child in children:
        if isinstance(child, nn.BatchNorm2d):
            bc = fuse(c, child)
            # 将原来的conv层替换成conv+bn的融合
            m._modules[cn] = bc
            # 将原来的bn层替换成空
            m._modules[name] = DummyModule()
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
            cn = name
        else:
            fuse_module(child)


def test_net(m):
    p = torch.randn([1, 3, 224, 224])
    import time
    s = time.time()
    o_output = m(p)
    print("Original time: ", time.time() - s)

    fuse_module(m)

    s = time.time()
    f_output = m(p)
    print("Fused time: ", time.time() - s)

    print("Max abs diff: ", (o_output - f_output).abs().max().item())
    assert(o_output.argmax() == f_output.argmax())
    # print(o_output[0][0].item(), f_output[0][0].item())
    print("MSE diff: ", nn.MSELoss()(o_output, f_output).item())


def test_layer():
    p = torch.randn([1, 3, 112, 112])
    conv1 = m.conv1
    bn1 = m.bn1
    o_output = bn1(conv1(p))
    fusion = fuse(conv1, bn1)
    f_output = fusion(p)
    print(o_output[0][0][0][0].item())
    print(f_output[0][0][0][0].item())
    print("Max abs diff: ", (o_output - f_output).abs().max().item())
    print("MSE diff: ", nn.MSELoss()(o_output, f_output).item())


m = tv.models.resnet152(True)
m.eval()
print("Layer level test: ")
test_layer()

print("============================")
print("Module level test: ")
m = tv.models.resnet18(True)
m.eval()
test_net(m)
```



其输出结果为

```shell
Layer level test: 
-2.90015371718e-08
-2.90015353954e-08
('Max abs diff: ', 2.384185791015625e-06)
('MSE diff: ', 2.4009499561563306e-14)
============================
Module level test: 
('Original time: ', 0.1468958854675293)
('Fused time: ', 0.030582904815673828)
('Max abs diff: ', 6.198883056640625e-06)
('MSE diff: ', 2.423213105867683e-12)

```

可以看到，原模型的输出和新模型（融合conv+bn之后的模型）的输出存在差异，这可能是在融合两层时，浮点数误差导致的。进而，这会导致模型精度的下降。







## 参考

[merge_bn 合并bn层到conv或FC层原理介绍及代码实现](https://blog.csdn.net/qq1145520074/article/details/79151948)

[网络inference阶段conv层和BN层的融合](https://zhuanlan.zhihu.com/p/48005099)

[PyTorch 卷积与BatchNorm的融合 - Captain Jack的文章 - 知乎](https://zhuanlan.zhihu.com/p/49329030)

