---
title: Gluon for MXNet 使用与概念梳理（冗长版）
tags:
  - MXNet
  - Gluon
categories:
  - 工具
mathjax: true
toc: true
date: 2018-01-15 11:45:25
---

基本是按照官方 Gluon 中文教程整理。

<!--more-->

## 1. 安装使用

- [AWS 上使用 MXNet 的环境配置](http://dday.top/2017/12/13/AWS%20%E4%B8%8A%E4%BD%BF%E7%94%A8%20MXNet%20%E7%9A%84%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE/)

## 2. 预备知识

### 2.1 NDArray

- 数据处理包括数据读取以及数据已经在内存中时如何处理。
- `NDArray ≈ NumPy + CPU/GPU异步计算 + 自动求导`
- mxnet.ndarray 的基本操作：

```python
from mxnet import ndarray as nd
## 创建矩阵
x = nd.ones((3, 4)); x = nd.zeros((3, 4))
y = nd.random_normal(0, 1, shape=(3, 4)) #服从均值0标准差1的正态分布
x.shape; x.size
## 操作符
z = x + y; z = x * y; z = nd.exp(y)
z = nd.dot(x, y.T)
## 广播
a = nd.arange(3).reshape((3,1))
b = nd.arange(2).reshape((1,2))
c = a + b #c.shape == (3,2)
## 与NumPy的互转
import numpy as np
x = np.ones((2,3))
y = nd.array(x)  # numpy -> mxnet.ndarray
z = y.asnumpy()  # mxnet.ndarray -> numpy
## 需要新开内存的操作
y = y + x
z = nd.zeros_like(x); z[:] = x + y #有临时空间的使用
## 避免内存开销
y += x; y[:] = y + x
z = nd.zeros_like(x); nd.elemwise_add(x, y, out=z)
## 截取 slicing
x = nd.arange(0,9).reshape((3,3))
x[1:3] #截取第1，2行
x[1:2,1:3] # 多维截取，第2行的第2，3列
```

- [NDArray API](https://mxnet.incubator.apache.org/api/python/ndarray.html)

### 2.2 autograd

- `mxnet.autograd` 可以对正常的命令式程序进行求导。它每次在后端实时创建计算图，从而可以立即得到梯度的计算方法。
- `mxnet.autograd` 的基本操作：

```python
import mxnet.ndarray as nd
import mxnet.autograd as ag
x = nd.array([[1, 2], [3, 4]])
## 申请存放导数的空间
x.attach_grad()
## 显示要求记录求导的程序
with ag.record():
    y = x * 2
    z = y * x
## 求导
z.backward() #如果z不是一个标量，那么z.backward()等价于nd.sum(z).backward().
## 查看导数
x.grad
```

- autograd 可以对动态图自动求导。

## 3. 入门使用

### 3.1 线性回归

```python
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
 
## 创建数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)
## 数据读取
batch_size = 10
dataset = gluon.data.ArrayDataset(X, y)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)
## 定义模型
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1)) #需要指定输出节点的个数
net.initialize() #模型初始化
square_loss = gluon.loss.L2Loss() #定义损失函数
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1}) #优化算法，学习率
## 训练
epochs = 5
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter:
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, average loss: %f" % (e, total_loss/num_examples))
## 查看weight和bias
dense = net[0]
dense.weight.data()
dense.bias.data()
```

### 3.2 多类逻辑回归

```python
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
 
## 获取 gluon.data.vision.FashionMNIST 数据集
def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')
mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)
## 数据读取
batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)
## 定义模型
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(10))
net.initialize()
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss() #Softmax和交叉熵的结合
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1}) #优化算法，学习率
## 计算精度
def accuracy(output, label): #将预测概率最高的那个类作为预测的类，比较
    return nd.mean(output.argmax(axis=1)==label).asscalar()
def evaluate_accuracy(data_iterator, net): #评估模型在数据集上的精度
    acc = 0.
    for data, label in data_iterator:
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_iterator)
## 训练
epochs = 5
for epoch in range(epochs):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
 
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label) #计算训练精度
 
    test_acc = evaluate_accuracy(test_data, net) # 每个epoch就在测试集上评估一下模型精度
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
## 预测
data, label = mnist_test[0:9]
predicted_labels = net(data).argmax(axis=1)
```

### 3.3 多层感知机

```python
from mxnet import gluon
## 定义模型
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(256, activation="relu")) #隐含层+ReLU
    net.add(gluon.nn.Dense(10))
net.initialize()
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss() #损失函数
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5}) #优化算法，学习率
```

- [mxnet.ndarray.Activation](https://mxnet.apache.org/api/python/ndarray.html#mxnet.ndarray.Activation)

### 3.4 L2 正则化和 Dropout

- 通过优化算法的`wd`参数（weight decay）实现对模型的正则化。这相当于 L2 范数正则化。

```python
from mxnet import gluon
 
trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': weight_decay})
```

- 在全连接层后添加`gluon.nn.Dropout`层并指定元素丢弃概率。
- 一般情况下，我们推荐把 更靠近输入层的元素丢弃概率设的更小一点.

```python
from mxnet.gluon import nn
 
net = nn.Sequential()
drop_prob1 = 0.2
drop_prob2 = 0.5
 
with net.name_scope():
    net.add(nn.Flatten())
    net.add(nn.Dense(256, activation="relu")) #第一层全连接
    net.add(nn.Dropout(drop_prob1)) #在第一层全连接后添加丢弃层
    net.add(nn.Dense(256, activation="relu")) #第二层全连接
    net.add(nn.Dropout(drop_prob2)) #在第二层全连接后添加丢弃层
    net.add(nn.Dense(10))
net.initialize()
```

### 3.5 批量归一化

- 如果把目标函数 f 根据参数 w 迭代（如 $f(w−η∇f(w))$ ）进行泰勒展开，有关学习率 η 的高阶项的系数可能由于数量级的原因（通常由于层数多）而不容忽略。然而常用的低阶优化算法（如梯度下降）对于不断降低目标函数的有效性通常基于一个基本假设：在以上泰勒展开中把有关学习率的高阶项通通忽略不计。
- 在训练时给定一个批量输入，批量归一化试图对深度学习模型的某一层所使用的激活函数的输入进行归一化：使批量呈标准正态分布（均值为0，标准差为1）。
- 数学推导：

$$\mu_B \leftarrow \frac{1}{m}\sum\limits_{i = 1}^{m}x_i,B = \{x_{1, ..., m}\}$$

$$\sigma_B^2 \leftarrow \frac{1}{m} \sum\limits_{i=1}^{m}(x_i - \mu_B)^2$$

$$\hat{x_i} \leftarrow \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y_i \leftarrow \gamma \hat{x_i} + \beta \equiv \mbox{BN}_{\gamma,\beta}(x_i)$$

- `ndarray` 的实现：

```python
from mxnet import nd
## 全连接层的BN:
X = nd.arange(6).reshape((3,2)) #data: batch_size x features
X = X.mean(axis=0)
variance = ((X-mean)**2).mean(axis=0)
## 二维卷积层的BN
X = nd.arrange(6).reshape((1,1,3,2)) #data: batch_size x channels x height x width
mean = X.mean(axis=(0,2,3), keepdims=True)
variance = ((X-mean)**2).mean(axis=(0,2,3), keepdims=True)
 
eps = 1e-5
gamma = nd.array([1,1])
beta = nd.array([0,0])
X_hat = (X - mean) / nd.sqrt(variance + eps)
y = gamma.reshape(mean.shape) * X_hat + beta.reshape(mean.shape)
```

- 在给训练数据的计算均值和方差时，同时更新全局的均值和方差，用 `moving_momentum=0.9` 来限制平均的度，在测试集上，用训练集的均值和方差来归一化：

```python
moving_mean = moving_mean.reshape(mean.shape)
moving_variance = moving_variance.reshape(mean.shape)
moving_mean[:] = moving_momentum * moving_mean + (1.0 - moving_momentum) * mean
moving_variance[:] = moving_momentum * moving_variance + (1.0 - moving_momentum) * variance
```

- 批量归一化在定义模型的时候通过添加 BN 层来实现：

```python
from mxnet.gluon import nn
 
net = nn.Sequential()
with net.name_scope():
    # 第一层卷积
    net.add(nn.Conv2D(channels=20, kernel_size=5))
    ### 添加了批量归一化层
    net.add(nn.BatchNorm(axis=1))
    net.add(nn.Activation(activation='relu'))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))
    # 第二层卷积
    net.add(nn.Conv2D(channels=50, kernel_size=3))
    ### 添加了批量归一化层
    net.add(nn.BatchNorm(axis=1))
    net.add(nn.Activation(activation='relu'))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))
    net.add(nn.Flatten())
    # 第一层全连接
    net.add(nn.Dense(128, activation="relu"))
    # 第二层全连接
    net.add(nn.Dense(10))
```

## 4. 各种CNN网络结构

### 4.1 卷积神经网络

#### 4.1.1 卷积层

- 当输入数据有多个通道的时候，每个通道会有对应的权重，对每个通道做卷积之后在通道之间求和：

$$conv(data, w, b) = \sum\limits_i conv(data[:,i,:,:], w[:,i,:,:], b)$$

- 当输出需要多通道时，每个输出通道有对应权重，然后每个通道上做卷积：

$$conv(data, w, b)[:,i,:,:] = conv(data, w[i,:,:,:], b[i])$$

- `nd.Convolution` 的实现：

```python
from mxnet import nd
# 输入输出数据格式是 batch x channel x height x width，这里batch和channel都是1
# 权重：output_channels x in_channels x height x width，这里input_filter和output_filter都是1。
w = nd.arange(4).reshape((1,1,2,2)); b = nd.array([1])
data = nd.arange(9).reshape((1,1,3,3))
out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[1]) #输出：[1,1,2,2]
out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[1],
                     stride=(2,2), pad=(1,1)) # 输出：[1,1,2,2]
## 多个输入通道
w = nd.arange(8).reshape((1,2,2,2))
data = nd.arange(18).reshape((1,2,3,3))
out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[0]) #输出：[1,1,2,2]
## 多个输出通道
w = nd.arange(16).reshape((2,2,2,2)); b = nd.array([1,2])
data = nd.arange(18).reshape((1,2,3,3))
out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[0]) #输出：[1,2,2,2]
```

#### 4.2.2 池化层

- 选出窗口里面最大的元素，或者平均元素作为输出:

```python
out = nd.arange(18).reshape((1,2,3,3))
max_pool = nd.Pooling(data=out, pool_type="max", kernel=(2,2))
avg_pool = nd.Pooling(data=out, pool_type="avg", kernel=(2,2))
```

### 4.2 LeNet

```python
from mxnet.gluon import nn
## LeNet
# 两个卷积-池化层，两个全连接层
net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(128, activation="relu"),
        nn.Dense(10)
    )
```

### 4.3 AlexNet

```python
from mxnet.gluon import nn
 
net = nn.Sequential()
with net.name_scope():
    net.add(
        # 第一阶段 kernel_size=11x11, channels=96
        nn.Conv2D(channels=96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # 第二阶段 kernel_size=5x5, channels=256,
        nn.Conv2D(channels=256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # 第三阶段 kernel_size=3x3, channels=[384, 384, 256]
        nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # 第四阶段
        nn.Flatten(),
        nn.Dense(4096, activation="relu"),
        nn.Dropout(.5),
        # 第五阶段
        nn.Dense(4096, activation="relu"),
        nn.Dropout(.5),
        # 第六阶段
        nn.Dense(10)
    )
```

### 4.4 VGG

- 论文：[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

```python
from mxnet.gluon import nn
 
def vgg_block(num_convs, channels):
    out = nn.Sequential()
    for _ in range(num_convs):
        out.add(nn.Conv2D(channels=channels, kernel_size=3, padding=1, activation='relu'))
    out.add(nn.MaxPool2D(pool_size=2, strides=2))
    return out
def vgg_stack(architecture):
    out = nn.Sequential()
    for (num_convs, channels) in architecture:
        out.add(vgg_block(num_convs, channels))
    return out
## VGG-11
num_outputs = 10
architecture = ((1,64), (1,128), (2,256), (2,512), (2,512))
net = nn.Sequential()
with net.name_scope():
    net.add(
        vgg_stack(architecture),
        nn.Flatten(),
        nn.Dense(4096, activation="relu"),
        nn.Dropout(.5),
        nn.Dense(4096, activation="relu"),
        nn.Dropout(.5),
        nn.Dense(num_outputs))
```

### 4.5 NiN

- Network in Network：串联数个卷积层块和全连接层块来构建深度网络。
- NiN 提出只对通道层做全连接并且像素之间共享权重来解决全连接层有过多的参数的问题。就是说，我们使用kernel大小是1×1的卷积。
- 1x1 的卷积：实现跨通道的交互和信息整合，进行卷积核通道数的降维和升维，减少网络参数。

```python
from mxnet.gluon import nn
 
def mlpconv(channels, kernel_size, padding, strides=1, max_pooling=True):
    out = nn.Sequential()
    out.add(
        nn.Conv2D(channels=channels, kernel_size=kernel_size,
                  strides=strides, padding=padding, activation='relu'),
        nn.Conv2D(channels=channels, kernel_size=1,
                  padding=0, strides=1, activation='relu'),
        nn.Conv2D(channels=channels, kernel_size=1,
                  padding=0, strides=1, activation='relu'))
    if max_pooling:
        out.add(nn.MaxPool2D(pool_size=3, strides=2))
    return out
```

- NiN 的卷积层的参数跟 Alexnet 类似，使用三组不同的设定;除了使用了 1×1 卷积外，NiN 在最后不是使用全连接，而是使用通道数为输出类别个数的`mlpconv`，外接一个平均池化层来将每个通道里的数值平均成一个标量：

```python
net = nn.Sequential()
with net.name_scope():
    net.add(
        mlpconv(96, 11, 0, strides=4),
        mlpconv(256, 5, 2),
        mlpconv(384, 3, 1),
        nn.Dropout(.5),
        # 目标类为10类
        mlpconv(10, 3, 1, max_pooling=False),
        # 输入为 batch_size x 10 x 5 x 5, 通过AvgPool2D转成
        # batch_size x 10 x 1 x 1。
        nn.AvgPool2D(pool_size=5),
        # 转成 batch_size x 10
        nn.Flatten()
    )
```

- “一卷卷到底最后”再加一个平均池化层。

### 4.6 GoogLeNet

- Inception：四个并行卷积层的块

```python
from mxnet.gluon import nn
from mxnet import nd
 
class Inception(nn.Block):
    def __init__(self, n1_1, n2_1, n2_3, n3_1, n3_5, n4_1, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # path 1: conv 1x1
        self.p1_conv_1 = nn.Conv2D(n1_1, kernel_size=1, activation='relu')
        # path 2: conv 1x1 conv 3x3
        self.p2_conv_1 = nn.Conv2D(n2_1, kernel_size=1, activation='relu')
        self.p2_conv_3 = nn.Conv2D(n2_3, kernel_size=3, padding=1, activation='relu')
        # path 3: conv 1x1 conv 5x5
        self.p3_conv_1 = nn.Conv2D(n3_1, kernel_size=1, activation='relu')
        self.p3_conv_5 = nn.Conv2D(n3_5, kernel_size=5, padding=2, activation='relu')
        # path 4: pool 3x3 conv 1x1
        self.p4_pool_3 = nn.MaxPool2D(pool_size=3, padding=1, strides=1)
        self.p4_conv_1 = nn.Conv2D(n4_1, kernel_size=1, activation='relu')
    def forward(self, x):
        p1 = self.p1_conv_1(x)
        p2 = self.p2_conv_3(self.p2_conv_1(x))
        p3 = self.p3_conv_5(self.p3_conv_1(x))
        p4 = self.p4_conv_1(self.p4_pool_3(x))
        return nd.concat(p1, p2, p3, p4, dim=1)
```

- GoogLeNet:

```python
class GoogLeNet(nn.Block):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(GoogLeNet, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            # block 1
            b1 = nn.Sequential()
            b1.add(
                nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2)
            )
            # block 2
            b2 = nn.Sequential()
            b2.add(
                nn.Conv2D(64, kernel_size=1),
                nn.Conv2D(192, kernel_size=3, padding=1),
                nn.MaxPool2D(pool_size=3, strides=2)
            )
            # block 3
            b3 = nn.Sequential()
            b3.add(
                Inception(64, 96, 128, 16,32, 32),
                Inception(128, 128, 192, 32, 96, 64),
                nn.MaxPool2D(pool_size=3, strides=2)
            )
            # block 4
            b4 = nn.Sequential()
            b4.add(
                Inception(192, 96, 208, 16, 48, 64),
                Inception(160, 112, 224, 24, 64, 64),
                Inception(128, 128, 256, 24, 64, 64),
                Inception(112, 144, 288, 32, 64, 64),
                Inception(256, 160, 320, 32, 128, 128),
                nn.MaxPool2D(pool_size=3, strides=2)
            )
            # block 5
            b5 = nn.Sequential()
            b5.add(
                Inception(256, 160, 320, 32, 128, 128),
                Inception(384, 192, 384, 48, 128, 128),
                nn.AvgPool2D(pool_size=2)
            )
            # block 6
            b6 = nn.Sequential()
            b6.add(
                nn.Flatten(),
                nn.Dense(num_classes)
            )
            # chain blocks together
            self.net = nn.Sequential()
            self.net.add(b1, b2, b3, b4, b5, b6)
    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out
```

- **Version1**：[Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842)
- **Version2**：[Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/abs/1502.03167) 加入了 Batch Normalization
- **Version3**：[Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567) 调整了 Inception
- **Version4**：[Inception-ResNet and the Impact of Residual Connections on Learning](http://arxiv.org/abs/1602.07261) 加入了 Residual Connections

### 4.7 ResNet

- Residual 块：ResNet 沿用了 VGG 的那种全用 3×3 卷积，但在卷积和池化层之间加入了批量归一层来加速训练，每次跨层连接跨过两层卷积。

```python
from mxnet.gluon import nn
from mxnet import nd
 
class Residual(nn.Block):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1, strides=strides)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm()
        if not same_shape: #如果输入的通道数和输出不一样时，使用一个额外的1×1卷积来做通道变化
            self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=strides)
    def forward(self, x):
        out = nd.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return nd.relu(out + x)
```

- ResNet-18-v1：

```python
class ResNet(nn.Block):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            # block 1: conv 7x7
            b1 = nn.Conv2D(64, kernel_size=7, strides=2)
            # block 2: pool 3x3 + Residual(64) + Residual(64)
            b2 = nn.Sequential()
            b2.add(
                nn.MaxPool2D(pool_size=3, strides=2),
                Residual(64),
                Residual(64)
            )
            # block 3: Residual(128) + Residual(128)
            b3 = nn.Sequential()
            b3.add(
                Residual(128, same_shape=False),
                Residual(128)
            )
            # block 4: Residual(256) + Residual(256)
            b4 = nn.Sequential()
            b4.add(
                Residual(256, same_shape=False),
                Residual(256)
            )
            # block 5: Residual(512) + Residual(512)
            b5 = nn.Sequential()
            b5.add(
                Residual(512, same_shape=False),
                Residual(512)
            )
            # block 6: pool 3x3 + dense
            b6 = nn.Sequential()
            b6.add(
                nn.AvgPool2D(pool_size=3),
                nn.Dense(num_classes)
            )
            # chain all blocks together
            self.net = nn.Sequential()
            self.net.add(b1, b2, b3, b4, b5, b6)
    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out
```

- ResNet_v1：`conv->BN->relu`
- ResNet_v1_bottleneck：
- ResNet_v2：`BN->relu->conv` [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
- ResNet-18/34/50/101/152 一览：[ResNet的理解及其Keras实现](http://lanbing510.info/2017/08/21/ResNet-Keras.html)


### 4.8 DenseNet

- DenseNet 里来自跳层的输出不是通过加法（+）而是拼接（concat）来跟目前层的输出合并。因为是拼接，所以底层的输出会保留的进入上面所有层。
- Dense 块：使用 ResNet 改进版本的 `BN->Relu->Conv` ，每个卷积的输出通道数被称之为 growth_rate，这是因为假设输出为 in_channels，而且有 layers 层，那么输出的通道数就是 `in_channels+growth_rate*layers`。

```python
from mxnet import nd
from mxnet.gluon import nn
 
def conv_block(channels):
    out = nn.Sequential()
    out.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(channels, kernel_size=3, padding=1)
    )
    return out
class DenseBlock(nn.Block):
    def __init__(self, layers, growth_rate, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for i in range(layers):
            self.net.add(conv_block(growth_rate))
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = nd.concat(x, out, dim=1)
        return x
```

- 过渡块（Transition Block）：因为使用拼接的缘故，每经过一次拼接输出通道数可能会激增。为了控制模型复杂度，这里引入一个过渡块，它不仅把输入的长宽减半，同时也使用 1×1 卷积来改变通道数。

```python
def transition_block(channels):
    out = nn.Sequential()
    out.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(channels, kernel_size=1),
        nn.AvgPool2D(pool_size=2, strides=2)
    )
    return out
```

- DenseNet-121：

```python
init_channels = 64
growth_rate = 32
block_layers = [6, 12, 24, 16]
num_classes = 10
def dense_net():
    net = nn.Sequential()
    with net.name_scope():
        # first block: conv 7x7 + pool 3x3
        net.add(
            nn.Conv2D(init_channels, kernel_size=7,
                      strides=2, padding=3),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1)
        )
        # dense blocks: DenseBlock + transition_block
        channels = init_channels
        for i, layers in enumerate(block_layers):
            net.add(DenseBlock(layers, growth_rate))
            channels += layers * growth_rate
            if i != len(block_layers)-1:
                net.add(transition_block(channels//2))
        # last block: pool + dense
        net.add(
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.AvgPool2D(pool_size=1),
            nn.Flatten(),
            nn.Dense(num_classes)
        )
    return net
```

- Desnet 通过将 ResNet 里的 + 替换成 concat 从而获得更稠密的连接。


## 5. Gluon 基础

### 5.1 自定义神经网络

- `nn.Sequential`的主要好处是定义网络起来更加简单。但`nn.Block`可以提供更加灵活的网络定义。

```python
from mxnet import nd
from mxnet.gluon import nn
 
## 简单神经网络的定义
#通过nn.Sequential
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(256, activation="relu"))
    net.add(nn.Dense(10))
#通过继承nn.Block类，定义__init__和forward基本方法
class MLP(nn.Block):
    def __init__(self, **kwargs): #参数可以传入prefix和params
        super(MLP, self).__init__(**kwargs) #调用nn.Block的__init__
        with self.name_scope(): #调用nn.Block提供的name_scope()
            self.dense0 = nn.Dense(256)
            self.dense1 = nn.Dense(10)
    def forward(self, x):
        return self.dense1(nd.relu(self.dense0(x)))
net = MLP()
#nn.Sequential的简单定义
class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)
    def add(self, block):
        self._children.append(block)
    def forward(self, x):
        for block in self._children:
            x = block(x)
        return x
net4 = MySequential()
with net4.name_scope():
    net4.add(nn.Dense(256, activation="relu"))
    net4.add(nn.Dense(10))
#使用nn.Block更灵活的定义
class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense = nn.Dense(256)
            self.weight = nd.random_uniform(shape=(256,20))
    def forward(self, x):
        x = nd.relu(self.dense(x))
        x = nd.relu(nd.dot(x, self.weight)+1)
        x = nd.relu(self.dense(x))
        return x
```

- `nn`下面的类基本都是`nn.Block`的子类，他们可以很方便地嵌套使用。

```python
#nn.Block与nn.Sequential嵌套使用
class RecMLP(nn.Block):
    def __init__(self, **kwargs):
        super(RecMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        with self.name_scope():
            self.net.add(nn.Dense(256, activation="relu"))
            self.net.add(nn.Dense(128, activation="relu"))
            self.dense = nn.Dense(64)
    def forward(self, x):
        return nd.relu(self.dense(self.net(x)))
rec_mlp = nn.Sequential()
rec_mlp.add(RecMLP())
rec_mlp.add(nn.Dense(10))
```

- 设计自定义层

```python
from mxnet import nd
from mxnet.gluon import nn
 
## 不需要维护模型参数的简单自定义层
class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
    def forward(self, x):
        return x - x.mean()
layer = CenteredLayer()
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(128))
    net.add(nn.Dense(10))
    net.add(layer)
## 带模型参数的自定义层
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(in_units, units))
            self.bias = self.params.get('bias', shape=(units,))
    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)
dense = MyDense(5, in_units=10, prefix='o_my_dense_')
dense.params
net = nn.Sequential()
with net.name_scope():
    net.add(MyDense(32, in_units=64))
    net.add(MyDense(2, in_units=32))
```

### 5.2 模型参数访问与初始化

- [mxnet.initializer](https://mxnet.incubator.apache.org/api/python/optimization.html#the-mxnet-initializer-package)
- 延后初始化，为了模型定义的时候不需要指定输入大小。

```python
## 模型参数访问
#通过weight和bias访问Dense的参数
w = net[0].weight
b = net[0].bias
w.data(); b.data() #访问参数
w.grad(); b.grad() #访问梯度
#通过collect_params来访问Block里面所有的参数
params = net.collect_params() #会返回一个名字到对应Parameter的dict
params['sequential0_dense0_bias'].data()
params.get('dense0_weight').data() #使用get不需要填写名字前缀
## 模型参数初始化
from mxnet import init
net.initialize() #把所有权重初始化成在[-0.07, 0.07]之间均匀分布的随机数
params.initialize(init=init.Normal(sigma=0.02), force_reinit=True) #正态分布初始化
## 共享模型参数
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(4, activation="relu"))
    net.add(nn.Dense(4, activation="relu"))
    net.add(nn.Dense(4, activation="relu", params=net[-1].params))
    net.add(nn.Dense(2))
```

- 自定义初始化方法

```python
from mxnet import init
class MyInit(init.Initializer):
    def __init__(self):
        super(MyInit, self).__init__()
        self._verbose = True
    def _init_weight(self, _, arr):
        # 初始化权重，使用out=arr后我们不需指定形状
        print('init weight', arr.shape)
        nd.random.uniform(low=5, high=10, out=arr)
net = get_net()
net.initialize(MyInit())
net(x) #因为延后初始化，所以要调用一次才会初始化
net[0].weight.data()
```

- `net.initialize(ctx=mx.gpu(), init=init.Xavier())`

### 5.3 序列化 - 读写模型

- 读写 NDArray

```python
from mxnet import nd
x = nd.ones(3)
y = nd.zeros(4)
filename = "../data/test1.params"
nd.save(filename, [x, y]) #save写入文件
a, b = nd.load(filename) #load读入文件
```

- 读写Gluon模型的参数

```python
from mxnet.gluon import nn
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(10, activation="relu"))
    net.add(nn.Dense(2))
net.initialize()
x = nd.random.uniform(shape=(2,10))
net(x)
filename = "../data/mlp.params"
net.save_params(filename) #save_params写入模型参数
 
net2 = nn.Sequential()
with net2.name_scope():
    net2.add(nn.Dense(10, activation="relu"))
    net2.add(nn.Dense(2))
net2.load_params(filename) #load_params读入相同模型的参数
```

### 5.4 GPU的使用

- MXNet 使用 Context 来指定使用哪个设备来存储和计算。默认会将数据开在主内存，然后利用 CPU 来计算，这个由 `mx.cpu()` 来表示。
- GPU 则由 `mx.gpu()` 来表示。注意 `mx.cpu()` 表示所有的物理 CPU 和内存，意味着计算上会尽量使用多有的 CPU 核。但 `mx.gpu()` 只代表一块显卡和其对应的显卡内存。
- 如果有多块 GPU，我们用 `mx.gpu(i)` 来表示第 *i* 块 GPU（ *i* 从 0 开始）。
- GPU 上创建内存：

```python
from mxnet import nd
 
x = nd.array([1,2,3])
x.context #查看数据存在哪个设备上，默认在CPU
a = nd.array([1,2,3], ctx=mx.gpu()) #指定在GPU上创建变量
b = nd.zeros((3,2), ctx=mx.gpu())
c = nd.random.uniform(shape=(2,3), ctx=mx.gpu())
y = x.copyto(mx.gpu()) #将x传输到GPU上新建内存y上
z = x.as_in_context(mx.gpu()) #相比较copyto的优点在于，如果设备一致，则不复制
```

- 所有计算要求输入数据在同一个设备上。不一致的时候系统不进行自动复制。
- 默认会复制回 CPU 的操作：如果某个操作需要将 NDArray 里面的内容转出来，例如打印或变成 numpy 格式，如果需要的话系统都会自动将数据 copy 到主内存。
- Gluon 的 GPU 计算：

```python
from mxnet import gluon
 
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))
net.initialize(ctx=mx.gpu()) #模型参数在GPU上初始化
data = nd.random.uniform(shape=[3,2], ctx=mx.gpu()) #GPU上的数据
net(data) #在GPU上计算结果
```

## 6. 优化算法

- 局部最小值：各个方向的梯度都为 0，但不是全局最优点。
- 鞍点：大部分方向的梯度为 0，其他方向仍有明显下降趋势。

### 6.1 SGD

```python
lr = 0.2
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
# 学习率在2个epoch后自我衰减
if epoch > 2:
    trainer.set_learning_rate(trainer.learning_rate * 0.1)
```

### 6.2 Momentum

```python
lr = 0.2; mom = 0.9
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': mom})
# 重设学习率。
if epoch > 2:
  trainer.set_learning_rate(trainer.learning_rate * 0.1)
```

### 6.3 Adagrad

```python
lr = 0.2
trainer = gluon.Trainer(net.collect_params(), 'adagrad', {'learning_rate': lr})
```

### 6.4 RMSProp

```python
lr = 0.03; gamma = 0.999
trainer = gluon.Trainer(net.collect_params(), 'rmsprop', {'learning_rate': lr, 'gamma1': gamma})
```

### 6.5 Adadelta

```python
rho = 0.9999
trainer = gluon.Trainer(net.collect_params(), 'adadelta', {'rho': rho})
```

### 6.6 Adam

```python
lr = 0.1
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
```

## 7. 进阶操作

### 7.1 Hybridize

- 命令式编程：可以拿到所有中间变量值。
- 符号式编程：更加高效而且更容易移植。
  - 定义计算流程
  - 编译成可执行的程序
  - 给定输入调用编译好的程序
- 用户应该用纯命令式的方法来使用 Gluon 进行开发和调试。但当需要产品级别的性能和部署的时候，我们可以将代码，至少大部分，转换成符号式来运行。
- 通过使用 HybridBlock 或者 HybridSequential 来构建神经网络。默认他们跟 Block 和 Sequential 一样使用命令式执行。当我们调用`.hybridize()`后，系统会转换成符号式来执行。事实上，所有 Gluon 里定义的层全是 HybridBlock，这个意味着大部分的神经网络都可以享受符号式执行的优势。
- `nn.HybridSequential`

```python
from mxnet.gluon import nn
from mxnet import nd
 
net = nn.HybridSequential()
with net.name_scope():
    net.add(
        nn.Dense(256, activation="relu"),
        nn.Dense(128, activation="relu"),
        nn.Dense(2)
    )
net.initialize()
net.hybridize() # 2倍速度
```

- `nn.HybridBlock`

```python
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        with self.name_scope():
            self.fc1 = nn.Dense(10)
            self.fc2 = nn.Dense(2)
    def hybrid_forward(self, F, x):
        print(F)
        x = F.relu(self.fc1(x))
        return self.fc2(x) 
net = HybridNet()
net.initialize()
x = nd.random.normal(shape=(1, 4))
y = net(x) #module 'mxnet.ndarray'
 
net.hybridize()
y = net(x)
#module 'mxnet.symbol',另外，输入和中间输出都变成了symbol
y = net(x)
#无输出，因为将输入替换成Symbol来构建符号式的程序。
#再运行的时候系统将不再访问Python的代码，而是直接在C++后端执行这个符号式程序。
```

- MXNet 有一个符号式的 API (symbol) 和命令式的 API (ndarray)。这两个接口里面的函数基本是一致的。系统会根据输入来决定 F 是使用 symbol 还是 ndarray 。

### 7.2 延后执行

- 延后执行使得系统有更多空间来做性能优化。
- 推荐每个批量里至少有一个同步函数，例如对损失函数进行评估，来避免将过多任务同时丢进后端系统。
- 同步函数：
  - `print`
  - `nd.NDArray.wait_to_read()`
  - `nd.waitall()`

### 7.3 自动并行

- MXNet 能够自动并行执行没有数据依赖关系的任务从而提升系统性能。

### 7.4 多 CPU 训练

- 多设备的初始化：

```python
ctx = [gpu(0), gpu(1)]
net.initialize(ctx=ctx)
```

- 将数据分割并返回各个设备上的复制：

```python
x = nd.random.uniform(shape=(4, 1, 28, 28))
x_list = gluon.utils.split_and_load(x, ctx)
```

- `gluon.Trainer` 里面会被默认执行在多GPU之间复制梯度求和并广播。

## 8. 计算机视觉领域的常用操作

### 8.1 图片增广

- 常用方法以及注意事项：

```python
import matplotlib.pyplot as plt
from mxnet import image
 
img = image.imdecode(open('../img/cat1.jpg', 'rb').read())
plt.imshow(img.asnumpy())
 
def apply(img, aug, n=3):
    # 转成float，一是因为aug需要float类型数据来方便做变化。
    # 二是这里会有一次copy操作，因为有些aug直接通过改写输入（而不是新建输出）获取性能的提升
    X = [aug(img.astype('float32')) for _ in range(n*n)]
    # 有些aug不保证输入是合法值，所以做一次clip
    # 显示浮点图片时imshow要求输入在[0,1]之间
    Y = nd.stack(*X).clip(0,255)/255
    utils.show_images(Y, n, n, figsize=(8,8))
 
## 以.5的概率做翻转
aug = image.HorizontalFlipAug(.5); apply(img, aug)
## 随机裁剪一个块 200 x 200 的区域
aug = image.RandomCropAug([200,200]); apply(img, aug)
## 随机裁剪一块随机大小的区域
aug = image.RandomSizedCropAug((200,200), .1, (.5,2)); apply(img, aug)
## 随机将亮度增加或者减小在0-50%间的一个量
aug = image.BrightnessJitterAug(.5); apply(img, aug)
## 随机色调变化
aug = image.HueJitterAug(.5); apply(img, aug)
```

- CIFAR10 使用示例：

```python
from mxnet import gluon
from mxnet import nd
 
def apply_aug_list(img, augs):
    for f in augs: img = f(img)
    return img
def get_transform(augs):
    def transform(data, label):
        # data: sample x height x width x channel label: sample
        data = data.astype('float32')
        if augs is not None:
            data = nd.stack(*[apply_aug_list(d, augs) for d in data])
        data = nd.transpose(data, (0,3,1,2)) #sample x channel x height x width
        return data, label.astype('float32')
    return transform
def get_data(batch_size, train_augs, test_augs=None):
    cifar10_train = gluon.data.vision.CIFAR10(train=True, transform=get_transform(train_augs))
    cifar10_test = gluon.data.vision.CIFAR10(train=False, transform=get_transform(test_augs))
    train_data = utils.DataLoader(cifar10_train, batch_size, shuffle=True)
    test_data = utils.DataLoader(cifar10_test, batch_size, shuffle=False)
    return (train_data, test_data)
```

### 8.2 Fine-tuning

- Hotdog 使用示例：

```python
from mxnet import gluon
import zipfile
## 获取数据集
data_dir = '../data'
fname = gluon.utils.download(
    'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/hotdog.zip',
    path=data_dir, sha1_hash='fba480ffa8aa7e0febbb511d181409f899b9baa5')
 
with zipfile.ZipFile(fname, 'r') as f:
    f.extractall(data_dir)
## 图片增强
from mxnet import nd
from mxnet import image
from mxnet import gluon
train_augs = [image.HorizontalFlipAug(.5), image.RandomCropAug((224,224))]
test_augs = [image.CenterCropAug((224,224))]
def transform(data, label, augs):
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)
    data = nd.transpose(data, (2,0,1))
    return data, nd.array([label]).asscalar().astype('float32')
train_imgs = gluon.data.vision.ImageFolderDataset(data_dir+'/hotdog/train',
    transform=lambda X, y: transform(X, y, train_augs))
test_imgs = gluon.data.vision.ImageFolderDataset(data_dir+'/hotdog/test',
    transform=lambda X, y: transform(X, y, test_augs))
## 读取数据
data = gluon.data.DataLoader(train_imgs, 32, shuffle=True)
## 获取预训练模型
from mxnet.gluon.model_zoo import vision as models
pretrained_net = models.resnet18_v2(pretrained=True) #pretrained_net = features + output
## 模型定义
from mxnet import init
finetune_net = models.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features #features与预训练模型一致
finetune_net.output.initialize(init.Xavier()) #仅初始化output部分
## 训练
def train(net, ctx, batch_size=64, epochs=10, learning_rate=0.01, wd=0.001):
    train_data = gluon.data.DataLoader(train_imgs, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(test_imgs, batch_size)
    net.collect_params().reset_ctx(ctx) #确保net的初始化在ctx上
    net.hybridize()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': wd})
    utils.train(train_data, test_data, net, loss, trainer, ctx, epochs)
```

- 附录：utils.train() 的定义

```python
def train(train_data, test_data, net, loss, trainer, ctx, num_epochs, print_batches=None):
    """Train a network"""
    print("Start training on ", ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        train_loss, train_acc, n, m = 0.0, 0.0, 0.0, 0.0
        if isinstance(train_data, mx.io.MXDataIter):
            train_data.reset()
        start = time()
        for i, batch in enumerate(train_data):
            data, label, batch_size = _get_batch(batch, ctx)
            losses = []
            with autograd.record():
                outputs = [net(X) for X in data]
                losses = [loss(yhat, y) for yhat, y in zip(outputs, label)]
            for l in losses:
                l.backward()
            train_acc += sum([(yhat.argmax(axis=1)==y).sum().asscalar()
                              for yhat, y in zip(outputs, label)])
            train_loss += sum([l.sum().asscalar() for l in losses])
            trainer.step(batch_size)
            n += batch_size
            m += sum([y.size for y in label])
            if print_batches and (i+1) % print_batches == 0:
                print("Batch %d. Loss: %f, Train acc %f" % (n, train_loss/n, train_acc/m))
 
        test_acc = evaluate_accuracy(test_data, net, ctx)
        print("Epoch %d. Loss: %.3f, Train acc %.2f, Test acc %.2f, Time %.1f sec" % (
            epoch, train_loss/n, train_acc/m, test_acc, time() - start))
```

## 参考文章

- [动手学深度学习](http://zh.gluon.ai/)
- [MXNet / Gluon 论坛](https://discuss.gluon.ai/)