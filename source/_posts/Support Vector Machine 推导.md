---
title: Support Vector Machine 推导
tags:
  - 机器学习
  - 理论
categories:
  - 机器学习
mathjax: true
toc: true
date: 2017-10-1 10:51:48
---

手推 SVM，涉及到拉格朗日乘子法、KKT 条件、对偶问题、岭回归、核函数等问题。

<!--more-->

## 1. 基本概念

- 通俗来讲，它是一种二类分类模型，其基本模型定义为特征空间上的间隔最大的线性分类器，其学习策略便是间隔最大化，最终可转化为一个凸二次规划问题的求解。
- 设想一个多维平面上散落着正样本和负样本，如果能找到一个超平面，恰好能将正负样本分开，那么这个超平面就可以用来对样本进行分类。
- SVM 要解决的两个核心问题：
  - 如果超平面确实存在，那么如何找到它？
  - 如果超平面不存在，那么如何找一个尽可能将正负样本分开的超平面？

## 2. 线性 SVM

### 2.1 硬间隔最大化

- 分割超平面的表达式：$wx+b=0$
- 两边离分割超平面最近的点即支撑向量（support vector）。
- 平行于分割超平面的支撑平面：$wx+b=1,wx+b=-1$ （因为w, b可以等比例缩放，所以右边可以等于1/-1方便计算）
- 对于所有的训练数据，要满足 $wx_++b\geq 1,wx_-+b\leq -1$ ，即：$y(wx+b)\geq1$
- 两个支撑平面的间隔：$width=(x_+-x_-)\frac{w}{||w||}=\frac{wx_+}{||w||}-\frac{wx_-}{||w||}=\frac{2}{||w||}$
- 最大化间隔问题转换为一个带不等式约束的凸优化问题：

$$min\frac{1}{2}||w||^2$$

$$s.t. y_i(wx_i+b)\geq 1, i=1,2,...,m.$$

- 结合 KKT 条件，拉格朗日函数可写为：

$$L(w,b,α)=\frac{1}{2}||w||^2-\sum\limits_{i=1}^Nα_iy_i(wx_i+b)+\sum\limits_{i=1}^Nα_i$$

- 问题转换为 $\min\limits_{w,b}\max\limits_{\alpha,\alpha>0}L(w,b,\alpha)$ , 由于满足 Slater 定理，所以可以构造拉格朗日对偶问题：

$$\min\limits_{w,b}\max\limits_{\alpha,\alpha>0}L(w,b,\alpha) =\max\limits_{\alpha,\alpha>0} \min\limits_{w,b}L(w,b,\alpha)$$

- 对 w, b 分别求导等于0，可以得到：

$$w=\sum\limits_i^Nα_iy_ix_i$$

$$\sum\limits_i^Nα_iy_i=0$$

- 代入到拉格朗日函数中可得：

$$\max\limits_{\alpha}L(α)=-\frac{1}{2}\sum\limits_i^N\sum\limits_j^Nα_iα_jy_iy_j(x_i\cdot x_j)+\sum\limits_i^Nα_i$$

$$s.t. \sum\limits_i^N\alpha_iy_i=0,\alpha_i\geq 0$$

- SMO 算法的基本思路：每次选择两个变量 $\alpha_{i}​$ 和 $\alpha_{j}​$ ，选取的两个变量所对应的样本之间间隔要尽可能大，因为这样更新会带给目标函数值更大的变化。
- SMO 算法之所以高效，是因为仅优化两个参数的过程实际上仅有一个约束条件，其中一个可由另一个表示，这样的二次规划问题具有闭式解。

### 2.2 软间隔最大化

- 由于现实中正负类样本相互交缠，不太可能严格满足线性可分。
- 可以认为存在一些特异点（outlier），于是引入一个松弛变量（slack variables） $ξ_i\geq 0$ ，约束条件变为：

$$y_i(wx_i+b)\geq 1-ξ_i$$

- 再引入惩罚参数 $C>0$ ，目标函数变成：

$$min\frac{1}{2}||w||^2+C\sum\limits_i^Nξ_i$$

$$s.t. y_i(wx_i+b)≥1-ξ_i, ξ_i\geq 0$$

- 松弛变量在使用时表示的是一种损失函数，譬如岭回归（Hinge）的损失函数 $l(z)=max(0,1-z)$

## 3. 非线性 SVM

- 线性 SVM 最后推导的式子可以看出，目标函数只使用了数据集两两点乘。
- 可以用核函数代替这个点乘，实现空间的非线性映射，使得样本在高维的特征空间内线性可分。
- 线性核函数可以降低线性 SVM 计算的复杂度，非线性核函数将特征映射到高维空间甚至无限维空间以寻找超平面。
- 常用核函数：
  - 线性核（Linear kernel function）：$K(x,z)=x^Tz$
  - 多项式核函数（polynomial kernel function）：$K(x,z) = (x\cdot z)^p$
  - 高斯核函数（Gaussian kernel function）：$K(x,z) = exp(-\frac{||x-z||^2}{2σ^2})$
- 目标函数变为：

$$\min L(α) = \frac{1}{2}\sum\limits_i^N\sum\limits_j^Nα_iα_jy_iy_jK(x_i,x_j)-\sum\limits_i^Nα_i$$

$$s.t. \sum\limits_i^Nα_iy_i=0, 0 \leq α_i \leq C​$$

- 分割的超平面为：$\sum\limits_i^Nα_iy_iK(x,x_i)+b=0$

## 参考文章

- [如何优雅地手推SVM](https://applenob.github.io/svm.html)
- [手推SVM](https://zhuanlan.zhihu.com/p/31271919)
- [直观详解】支持向量机SVM](https://charlesliuyx.github.io/2017/09/19/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BASVM%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/)