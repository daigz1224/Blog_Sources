---
title: AndrewNg - Deep Learning (3)
tags:
  - Andrew Ng
  - Deep Learning
date: 2017-09-30 21:44:49
categories:
  - Deep Learning
toc: true
mathjax: true
---

本系列的目标是将 Andrew Ng 的 Deep Learning 课程整体梳理一遍，读薄课程中的基础知识和关键概念，便于回顾。文章内容大部分来自我的手写笔记，中英文混杂还请见谅。这个系列一共有五门课程，本文是本系列的第三课：**Structuring your ML Project**

<!--more-->

大家好，我是 Day。听过很多道理，却依然过不好这一生。看过很多书和视频，却与进阶知识眉来眼去不敢向前。前段时间读了一个非常好的个人博客，[小土刀](http://wdxtub.com/)，受益匪浅，他将看过的书都整理了下来，即所谓的"读薄"，沉淀下来总是最珍贵的。

------

### Week 1. ML Strategy (1)

**Ideas to Modify your Model**:

- Collect more data
- Collect more diverse training set
- Train algorithm longer with gradient descent
- Try Adam instead og gradient descent
- Try bigger network
- Try smaller network
- Try dropout
- Add L2 regularization
- Network architecture: Activation function, num of hidden units

#### Orthogoalization

**Chain of assumptions in ML**:

- Fit training set well on  cost function (≈ human-level) : Bigger network, Adam...
- Fit dev set well on cost function: Regularization, Bigger training set...
- Fit test set well on cost function: Bigger dev set...
- Performs well in real world: Change dev set or cost function...

#### Single Number Evaluation Metric

**F1 Score = Precision + Recall**, like an average of percision and recall.

$$F_1=\cfrac{2}{\cfrac{1}{P}+\cfrac{1}{R}}$$

$P=\cfrac{TP}{TP+FP}$, $R=\cfrac{TP}{TP+TN}$

#### Satisficing and Optimizing Metric

优化的目标是为了精度，但是也要注意网络的复杂度，过于复杂的网络在使用时不能达到实时性的要求。

通常优化的目标只有一个，其他均为需要满足的条件。

#### Human-level Performance and Avoidable Bias

机器学习的目标要接近甚至高于 Human-level，但不可能达到 Bayes Optimal。

- Get labeled data from humans
- Gain insight from manual error analysis
- Better analysis of bias / variance.

Avoidable Bias 指的是 Training error 到 Human-level error 的距离。

|                   |  Scenario A   |    Scenario B     |
| :---------------: | :-----------: | :---------------: |
| Human-level Error |      1%       |       7.5%        |
|  Training Error   |      8%       |        8%         |
| Development Error |      10%      |        10%        |
|    Next Target    | Focus on bias | Focus on variance |

**Surpassing human-level performance**: expecially with structured data

- Online Advertising
- Produce Recommendations
- Logistics (like predicting transit time)
- Loan Approvals

### Week 2. ML Strategy (2)

#### Error Analysis

- Look at dev examples to evaluate ideas
- Evaluate multiple ideas in parallel

No need to clean up incorrectly labeled data, cause DL algorithms are quite robust to **random** errors in the training set.

#### Training and Testing on Different Distributions

如果训练集和测试集不得不来自不同分布，那么要保证验证集和测试集一定要保持一致，而且目标明确。

可以设置一个**训练-验证集**，与训练集的分布一致，但不用于训练，只用于验证模型是否匹配训练集。

对于这种 **Data Mismatch** 的问题，可以通过手工误差分析去理解训练集和验证测试集的区别。也可以通过让训练集更像测试集方向去预处理，比如加入噪声等人工拟合。需要注意的是，**不要从所有可能的噪声空间中只选一部分去模拟，这样会导致对噪声的过拟合。**

#### Transfer Learning

通过大数据集的预训练学到易写基本的特征，重新初始化最后一层用于特定的小数据集的训练。

什么时候迁移学习最有效？

- 任务 A 和任务 B 有同样的输入 x
- 相比较任务 B，任务 A 有大量数据集来训练。
- 任务 A 可以学到一些低层次的，有利于任务 B 的特征。

所以当你的训练数据不够多的时候，可以找相关问题的模型去预训练。

#### Multi-task Learning

Softmax Regression: 输出的列向量只有一个 1。

Multi-task Learning: 输出的列向量可以有很多 1。

#### End-to-End Learning

**Speech Recognition**:

**Audio** -MFCC-> **Feature** -ML-> **Phonemes** --> **Words** --> **Transcript**

**End-to-End Deep Learning**:

**Audio** ------------------------------------------------------------------> **Transcript**

**Pros**:

- Let the data speak.
- Less hand-designing of components needed.

**Cons**:

- May need large amount of data.
- Excludes potentially useful hand-designed components.