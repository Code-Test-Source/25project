# pytorch 入门
强烈建议通过MNIST手写数据集分类或者更复杂的分类数据集来亲手实操一下，不要直接看github别人的代码，遇到困难建议查pytorch官方文档API、网上的博客、教学视频
[0](https://www.runoob.com/pytorch/pytorch-tutorial.html)
[张量](https://zhuanlan.zhihu.com/p/21139808741)
## B站网课
[1](https://www.bilibili.com/video/BV1Y7411d7Ys)
[2](https://www.bilibili.com/video/BV1hE411t7RN)
## 常见网络层原理（network_layer.ipynb）
了解深度学习中常见网络层的原理、它们常用参数、pytorch代码中的定义方式：【卷积（特别是2D卷积，常用于图像）】输入/输出通道数，卷积核大小，stride大小，padding方式，输出矩阵大小如何计算；【全连接】输入/输出向量长度；【池化pooling】；【全卷积】作用与全连接类似；【BN层】
### 卷积
[1](https://zhuanlan.zhihu.com/p/701954213)
[2](https://blog.csdn.net/COINVK/article/details/129239715)

## Tensor与实例化（tensor.ipynb）

 - 了解tensor常用的处理方式：铺平，reshape，扩增维度，若干tensor拼接，添加非线性函数 
 - pytorch模型架构基本操作：用类定义模型，在模型中定义网络层，并在forward函数中定义网络层间的连接方式；将不同的模型进一步封装拼接成为更大的模型 
 - 将模型实例化，清楚一个神经网络模型如何利用损失信号更新参数，以及利用模型进行预测 
 - 如何保存模型，如何加载已有模型，如何仅加载部分模型参数，如何在训练时冻结某些模型参数 
 - 如何使用tensorboard（或其它好用的工具包）记录训练过程中的重要参数（如损失、分类正确率等），如何将记录的文件中的数据可视化

