---
title: Ubuntu 下 pytorch 的环境配置
tags:
  - PyTorch
  - Ubuntu
categories:
  - Ubuntu
  - PyTorch
mathjax: false
toc: true
date: 2017-11-08 21:07:01
---

从 Ubuntu16.04.3 的安装到 shadowsocks-qt5, Anaconda, pytorch, torchvision, visdom, fire 等软件的安装和简单配置。

<!--more-->

## 安装 Ubuntu16.04.3

1. PC 双系统安装步骤参考文章 [Win10+Ubuntu16.04 双系统 UEFI+GPT, SDD+HDD 解决方案](http://dday.top/2017/09/14/Win10+Ubuntu16.04%20%E5%8F%8C%E7%B3%BB%E7%BB%9F%20UEFI+GPT,%20SDD+HDD%20%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88/)
2. AWS 上申请 p2.xlarge 实例参考文章 [在AWS上运行教程](http://zh.gluon.ai/chapter_preface/aws.html) 或 [CS231N AWS Tutorial](http://cs231n.github.io/aws-tutorial/)
3. Google Cloud 上申请实例参考文章 [CS231N Google Cloud Tutorial](http://cs231n.github.io/gce-tutorial/) 或 [CS231N Google Cloud Tutorial Part 2 (with GPUs)](http://cs231n.github.io/gce-tutorial-gpus/)
4. 安装好之后一些简单的优化配置参考文章 [Ubuntu 初始配置以及简洁优雅美化指南](http://dday.top/2017/09/26/Ubuntu%20%E5%88%9D%E5%A7%8B%E9%85%8D%E7%BD%AE%E4%BB%A5%E5%8F%8A%E7%AE%80%E6%B4%81%E4%BC%98%E9%9B%85%E7%BE%8E%E5%8C%96%E6%8C%87%E5%8D%97/)

## 安装 shadowsocks5-qt5

```shell
# 通过PPA源安装，仅支持Ubuntu 14.04或更高版本
sudo add-apt-repository ppa:hzwhuang/ss-qt5
sudo apt-get update
sudo apt-get install shadowsocks-qt5
```

通过购买 Vultr VPS 搭建 ipv4/ipv6 梯子的[教程](http://blog.csdn.net/QingHeShiJiYuan/article/details/53330581) 。通过配置 shadowsocks5-qt5 以完成 Socks5 模式下的科学上网。并设置让终端走代理，不建议永久保存，可以使用 alias 或 proxychains。参考文章 [让终端走代理的几种方法](https://blog.fazero.me/2015/09/15/%E8%AE%A9%E7%BB%88%E7%AB%AF%E8%B5%B0%E4%BB%A3%E7%90%86%E7%9A%84%E5%87%A0%E7%A7%8D%E6%96%B9%E6%B3%95/) 。

## 安装 Anaconda

从 Tuna 的镜像源下载 [Anaconda](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)

```shell
#1. 下载Anaconda, https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/
bash Anaconda3-**.Linux-x86_64.sh
#2. 或者下载安装miniconda
# wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash
# 更换仓库源为Tuna的镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

使用 conda 创建一个 pytorch 的虚拟环境：

```
conda create --name pytorch python=3.6
source activate pytorch  # 进入 pytorch 环境
source deactivate  # 离开

# 常用 conda 命令:
# conda info --envs
# conda list
# conda remove --name XXX --all
```



## 安装 pytorch, torchvison

参考 [官网](http://pytorch.org/) 

建议选择 conda 路线，因为国内下载 pytorch 太慢，所以要打开代理。

如果没有代理，到 GitHub 上参考作者的 [文档](https://github.com/pytorch/pytorch#from-source) 。



## 安装一些实用工具[可选]

在项目中新建 requirements.txt

```
git+https://github.com/pytorch/tnt.git@master
fire
numpy
torchvision
visdom
ipdb
```

激活 pytorch 虚拟环境后：

```shell
pip install -r requirements.txt
```

工具介绍参考 [fire repo](https://github.com/google/python-fire), [tnt repo](https://github.com/pytorch/tnt), [visdom](https://github.com/facebookresearch/visdom) etc.

附录：visdom

由于众所周知的原因，当你启用 visdom: 

```
python -m visdom.server
```

 需要下载的的 js 文件可能打不开，导致 connection timed out。打开 localhost:8097 也是一片虚无。

当启用 visdom 时需要下载的文件和存放位置：

```
download from https://unpkg.com/jquery@3.1.1/dist/jquery.min.js 
   and put at visdom/static/js/jquery.min.js
download from https://unpkg.com/bootstrap@3.3.7/dist/js/bootstrap.min.js 
   and put at visdom/static/js/bootstrap.min.js
download from https://unpkg.com/react-resizable@1.4.6/css/styles.css 
   and put at visdom/static/css/react-resizable-styles.css
download from https://unpkg.com/react-grid-layout@0.14.0/css/styles.css
   and put at visdom/static/css/react-grid-layout-styles.css
download from https://unpkg.com/react@15.6.1/dist/react.min.js
   and put at visdom/static/js/react-react.min.js
download from https://unpkg.com/react-dom@15.6.1/dist/react-dom.min.js 
   and put at visdom/static/js/react-dom.min.js
download from https://unpkg.com/classnames@2.2.5 
   and put at visdom/static/fonts/classnames
download from https://unpkg.com/layout-bin-packer@1.2.2 
   and put at visdom/static/fonts/layout_bin_packer
download from https://cdn.rawgit.com/STRML/react-grid-layout/0.14.0/dist/react-grid-layout.min.js 
   and put at visdom/static/js/react-grid-layout.min.js
download from https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_SVG 
   and put at visdom/static/js/mathjax-MathJax.js
download from https://cdn.rawgit.com/plotly/plotly.js/master/dist/plotly.min.js 
   and put at visdom/static/js/plotly-plotly.min.js
download from https://unpkg.com/bootstrap@3.3.7/dist/css/bootstrap.min.css 
   and put at visdom/static/css/bootstrap.min.css
download from https://unpkg.com/bootstrap@3.3.7/dist/fonts/glyphicons-halflings-regular.eot 
   and put at visdom/static/fonts/glyphicons-halflings-regular.eot
download from https://unpkg.com/bootstrap@3.3.7/dist/fonts/glyphicons-halflings-regular.eot?#iefix 
   and put at visdom/static/fonts/glyphicons-halflings-regular.eot?#iefix
download from https://unpkg.com/bootstrap@3.3.7/dist/fonts/glyphicons-halflings-regular.woff2 
   and put at visdom/static/fonts/glyphicons-halflings-regular.woff2
download from https://unpkg.com/bootstrap@3.3.7/dist/fonts/glyphicons-halflings-regular.woff 
   and put at visdom/static/fonts/glyphicons-halflings-regular.woff
download from https://unpkg.com/bootstrap@3.3.7/dist/fonts/glyphicons-halflings-regular.ttf 
   and put at visdom/static/fonts/glyphicons-halflings-regular.ttf
download from https://unpkg.com/bootstrap@3.3.7/dist/fonts/glyphicons-halflings-regular.svg#glyphicons_halflingsregular 
   and put at visdom/static/fonts/glyphicons-halflings-regular.svg#glyphicons_halflingsregular
```

目前发现主要两个文件需要自己下载存放：[react-grid-layout.min.js](https://cdn.rawgit.com/STRML/react-grid-layout/0.14.0/dist/react-grid-layout.min.js)  和 [plotly-plotly.min.js](https://cdn.rawgit.com/plotly/plotly.js/master/dist/plotly.min.js )

将这两个文件翻墙下载下来放到 `visdom/static/js/` 中即可。翻不了墙的可以通过百度云下载：

链接: https://pan.baidu.com/s/1bo25jrD 

密码: 8uct