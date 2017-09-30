---
title: AndrewNg的DeepLearning课程知识整理(2)
tags:
  - Andrew Ng
  - Deep Learning
date: 2017-09-30 19:44:49
categories:
  - Deep Learning
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>

<script type="text/javascript"
   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

本系列的目标是将 Andrew Ng 的 Deep Learning 课程整体梳理一遍，读薄课程中的基础知识和关键概念，便于回顾。文章内容大部分来自我的手写笔记，中英文混杂还请见谅。这个系列一共有五门课程，本文是本系列的第二课：**Improving Deep Neural Network: Hyperparameter tuning, Regularization, Optimization**

<!--more-->

大家好，我是 Day。听过很多道理，却依然过不好这一生。看过很多书和视频，却与进阶知识眉来眼去不敢向前。前段时间读了一个非常好的个人博客，[小土刀](http://wdxtub.com/)，受益匪浅，他将看过的书都整理了下来，即所谓的"读薄"，沉淀下来总是最珍贵的。

本系列的目标是将 Andrew Ng 的 Deep Learning 课程整体梳理一遍，读薄课程中的基础知识和关键概念，便于回顾。文章内容大部分来自我的手写笔记，中英文混杂还请见谅。这个系列一共有五门课程，本文是本系列的第二课：**Improving Deep Neural Network: Hyperparameter tuning, Regularization, Optimization**

------

### Week 1. Setting up your ML Application

#### Train / Dev / Test Sets

- Train Set：用训练集对算法或模型进行训练过程；
- Development Set：利用验证集或者又称为简单交叉验证集（Hold-out Cross Validation Set）进行交叉验证，选择出最好的模型；
- Test Set：最后利用测试集对模型进行测试，获取模型运行的无偏估计。

**小数据集**：如 100, 1000, 10000 的数据量大小，可以将数据集做以下划分：

- 无验证集的情况：70% / 30%；
- 有验证集的情况：60% / 20% / 20%；

**大数据集**：百万级别的数据量，可以选择 1000 条数据就足以评估单个模型效果，可以将数据集做以下划分：

- 100万数据量：98% / 1% / 1%；
- 超百万数据量：99.5% / 0.25% / 0.25%（or 99.5% / 0.4% / 0.1%）

**建议：**

- 验证集要和训练集来自于同一个分布（Shuffle一下），可以使得机器学习算法变得更快；
- 如果不需要用无偏估计来评估模型的性能，则可以不需要测试集。

#### Bias / Variance Tradeoff

The base error is quite small, e.g Human-level ≈ 0%

|  Train Set Error   |      1%       |    15%    |   15%   |  0.5%   |
| :----------------: | :-----------: | :-------: | :-----: | :-----: |
| **Dev Set Error**  |    **11%**    |  **16%**  | **30%** | **1%**  |
| **Bias / Vaiance** | High Variance | High Bias | HB & HV | LB & LV |

#### Basic "Recipe" for ML

**是否 High Bias（欠拟合）？**

- Bigger Network
- Train Longer
- (NN Architecture Search)

**是否 High Variance（过拟合）？**

- More Data
- Regularization
- (NN Architecture Search)

#### Regularization

**直观理解**：正则化因子 lambda 设置的足够大的情况下，为了使代价函数最小化，权重矩阵 W 就会被设置为接近于0的值。则相当于消除了很多神经元的影响，那么图中的大的神经网络就会变成一个较小的网络。实际上隐藏层的神经元依然存在，但是他们的影响变小了，便不会导致过拟合。

**L2 ** (=Weight Decay)

**Logistic Regression**:

$$J(w,b)=\cfrac{1}{m}\sum^m_{i=1}L(\hat{y}^{(i)},y^{(i)})+\cfrac{\lambda}{2m}||w||^2_2$$

- L2 Regularization: $$\cfrac{\lambda}{2m}||w||^2_2=\cfrac{\lambda}{2m}\sum^{n_x}_{j=1}w^2_j=\cfrac{\lambda}{2m}w^Tw$$
- L1 Regularization: $$\cfrac{\lambda}{2m}||w||_1=\cfrac{\lambda}{2m}\sum^{n_x}_{j=1}|w_j|$$

**Neural Network**:

$$J(w^{[1]},b^{[1]},...,w^{[L]},b^{[L]},)=\cfrac{1}{m}\sum^m_{i=1}L(\hat{y}^{(i)},y^{(i)})+\cfrac{\lambda}{2m}\sum^L_{l=1}||w^{l}||^2_F$$

$$w.shape = (n^{[l]},n^{[l-1]})$$, F is "Frobenius norm"

**Weight Decay**:

$$dW^{[l]}=(from\,backprop)+\cfrac{\lambda}{m}W^{[l]}$$

$$W^{[l]}:=W^{[l]}-\alpha dW^{[l]}=(1-\cfrac{\alpha\lambda}{m})W^{[l]}-\alpha(from\,backprop)$$

$$Notes: (1-\cfrac{\alpha\lambda}{m})<1$$

**数学原理**：假设神经元中使用的激活函数为 $g(z) = tanh(z)$ 在加入正则化项后：当 $\lambda$ 增大，导致 $W^{[l]}$ 减小，$Z^{[l]}=W^{[l]}a^{[l-1]}+b^{[l]}$ 便会减小，当 z  处于数值较小的区域里，$tanh(z)$ 函数近似线性，所以每层的函数就近似线性函数，整个网络就成为一个简单的近似线性的网络，从而不会发生过拟合。

**Dropout** - "Inverted dropout"

```
# layer L = 3, keep_prob = 0.8
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob
a3 = np.multiply(a3, d3)
a3 /= keep_prob  # 为了不影响Z[4] = W[4]*a[3]+b[4]的期望值
```

**注意**：在测试阶段不要用dropout，因为那样会使得预测结果变得随机。

**直观理解**：通过 dropout，使得网络不会依赖于任何一个特征（可能会被丢弃），从而 shrink  the weights。

**缺点**：使得 Cost function 不再能被明确的定义。

**使用**：关闭dropout功能，即设置 keep_prob = 1.0；运行代码，确保 Cost Function 单调递减；再打开 dropout。

**Data Augmentation**

数据扩增，通过图片的一些变换，得到更多的训练集和验证集。

**Early Stopping**

在交叉验证集的误差上升之前的点停止迭代，避免过拟合。这种方法的缺点是无法同时解决 bias 和 variance 之间的最优。

#### Normalizing Inputs

$$\mu=\cfrac{1}{m}\sum^{[m]}_{i=1}x^{(i)}$$

$$\sigma^{2}=\cfrac{1}{m}\sum^m_{i=1}x^{(i)^2}$$

$$x:=x-\mu,x=x/\sigma^2$$

在不使用归一化的代价函数中，如果我们设置一个较小的学习率，那么很可能我们需要很多次迭代才能到达代价函数全局最优解；如果使用了归一化，那么无论从哪个位置开始迭代，我们都能以相对很少的迭代次数找到全局最优解。

#### Vanishing / Exploding Gradients 

在梯度函数上出现的以指数级递增或者递减的情况就分别称为梯度爆炸或者梯度消失。

**减缓梯度爆炸或梯度消失**：

当输入的 x 的维度较大时，我们希望每个 wi 都小一点，这样得到的和 z 也较小。

```
# Xavier Initialization
WL = np.random.randn(WL.shape[0],WL.shape[1])* np.sqrt(1/n)
# 激活函数使用 ReLU 的话，使用 np.sqrt(2/n)
# 激活函数使用 tanh 的话，使用 np.sqrt(1/n)
```

这么做是因为，如果激活函数的输入 xi 近似设置成均值为0，标准方差1的情况，输出 z 也会调整到相似的范围内。虽然没有解决梯度消失和爆炸的问题，但其在一定程度上确实减缓了梯度消失和爆炸的速度。

#### Gradient Check

使用双边误差的方法逼近导数，误差为 $O(\epsilon^2)$

因为我们的神经网络中含有大量的参数: W[1],b[1],⋯,W[L],b[L]，为了做梯度检验，需要将这些参数全部连接起来，reshape 成一个大的向量 θ 。同时对 dW[1],db[1],⋯,dW[L],db[L] 执行同样的操作。 

$$d\theta_{approx}[i]=\cfrac{J(\theta_1,...,\theta_i+\epsilon,...)-J(\theta_1,...,\theta_i-\epsilon,...)}{2\epsilon}$$

由，$$\cfrac{||d\theta_{approx}−d\theta||_2}{||d\theta_{approx}||_2+||dθ\theta||_2}$$

判断是否 $d\theta_{approx}≈d\theta$

- 不要在训练过程中使用梯度检验，只在 debug 的时候使用，使用完毕关闭梯度检验的功能；
- 如果算法的梯度检验出现了错误，要检查每一项，找出错误，也就是说要找出哪个$d\theta_{approx}[i]$与$d\theta$的值相差比较大；
- 不要忘记了正则化项；
- 梯度检验不能与dropout同时使用。因为每次迭代的过程中，dropout会随机消除隐层单元的不同神经元，这时是难以计算dropout在梯度下降上的代价函数J；
- 在随机初始化的时候运行梯度检验，或许在训练几次后再进行。

------

### Week 2. Optimization Algorithms

#### Mini-batch Gradient Descent

- If mini-batch size = m: Batch Gradient Descent
- If mini-batch size = 1: Stochastic Gradient Descent
- In-Between: fastest
  - Vectorization 
  -  Make progress without processing the entire set.
- If small training set (m≤2000): use BGD
- Typical mini-batch size: 64, 128, 256, 512
- Make sure mini-batch fit in CPU / GPU memory

#### Exponentially Weighted Averages

$V_t=\beta V_{t-1}+(1-\beta)\theta_t$,此时 $V_t$ 相当于之前 $\cfrac{1}{1-\beta}$ 个数据的平均。

Bias Correction: 因为冷启动的问题，所以可以用 $\cfrac{V_t}{1-\beta^t}=\cfrac{\beta V_t+(1-\beta)\theta_t}{1-\beta^t}$ 进行偏差修正。

#### Momentum

**直观理解**：选择了一条更直接的路径，减缓梯度的下降幅度。

```
# Hyperparameters: alpha, beta
# beta = 0.9, 相当于前10次平均，无需bias correction
VdW = 0; Vdb = 0
on iteration t:
	compute dW, db on current mini-batch
	VdW = beta*VdW + (1-beta)*dW
	Vdb = beta*Vdb + (1-beta)*db
	W := W - alpha*VdW
	b := b - alpha*db
```

#### RMSprop

控制 W, b 的梯度下降速度。

```
# Hyperparameters: alpha, beta
SdW = 0; Sdb = 0
on iteration t:
	compute dW, db on current mini-batch
	SdW = beta*SdW + (1-beta)*sqrt(dW,2)
	Sdb = beta*Sdb + (1-beta)*sqrt(db,2)
	W := W - alpha*dW/sqrt(SdW,0.5)  # SdW越小，W变化就越大
	b := b - alpha*db/sqrt(SdW,0.5)  # Sdb越大，b变化就越大
```

#### Adam

```
# Hyperparameters: alpha, beta1, beta2, epsilon
VdW = 0; Vdb = 0; SdW = 0; Sdb = 0
on iteration t:
	compute dW, db on current mini-batch
	
	VdW = beta1*VdW + (1-beta1)*dW
	Vdb = beta1*Vdb + (1-beta1)*db
	SdW = beta2*SdW + (1-beta2)*sqrt(dW,2)
	Sdb = beta2*Sdb + (1-beta2)*sqrt(db,2)
	
	VdW_correction = VdW / (1-sqrt(beta1,t))
	Vdb_correction = Vdb/ (1-sqrt(beta1,t))
	SdW_correction = SdW / (1-sqrt(beta2,t))
	Sdb_correction = Sdb/ (1-sqrt(beta2,t))
	
	W := W - alpha*(VdW_correction/(sqrt(SdW_correction,0.5)+epsilon))
	b := b - alpha*(Vdb_correction/(sqrt(Sdb_correction,0.5)+epsilon))
```

#### Learning Rate Decay

$$\alpha=\cfrac{1}{1+decay_rate*epoch_num}*\alpha_0$$

#### The Problem of Local Optima

- Unlikely to get stuck in a bad local optima, usually in saddle point.
- The plateaus can make learning slow.

------

### Week 3. Hyper Parameter Tuning

#### Tuning Process

在一定范围内随机取参数，不要使用网格记录，要由粗到细。

- alpha
- beta(=0.9), num of hidden units, mini-batch size
- num of layers, learning rate decay
- beta1(=0.9), beta2(=0.999), epsilon(=10^-8)

#### Appropriate Scale to Pick

不能线性取值，因为不同区段敏感度不同。

如果要选 alpha = 0.001 ~ 0.1, 那么就随机在 -4 到 -1选一个r，则 alpha = 10^r。

如果要选 beta  = 0.9 ~ 0.999, 那么就转换为 1 - beta 的及进行选择。 

要用鱼子酱的模式去试，即 Training many models in parallel。

#### Batch Normalization

#### Softmax Regression

