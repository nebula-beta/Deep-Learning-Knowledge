[TOC]

## SIFT





1. 构建金字塔
2. 构建DoG金字塔
3. [检测尺度空间极值（位置+尺度）](https://www.cnblogs.com/JiePro/p/sift_2.html)
   - 空间极值检测
   - 亚像素插值
   - 删除不稳定的特征点
   - 计算scale和size
4. [计算方向](https://www.cnblogs.com/JiePro/p/sift_3.html)
   - 统计梯度方向直方图
   - 对直方图进行高斯模糊
   - 取直方图的极值方向作为特征描述子的方向，并对方向做[抛物线插值](https://blog.csdn.net/q_z_r_s/article/details/82705653)
5. 计算描述子



### 构建金字塔

### 构建DoG金字塔

### 检测尺度空间极值



### 计算方向



### 计算描述子





## 参考

[1. SIFT算法中一些符号的说明](https://www.cnblogs.com/ronny/p/4028776.html)