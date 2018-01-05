---
title: Ubuntu 配置及美化指南
tags:
  - Linux
  - Ubuntu
date: 2017-09-26 16:41:58
categories:
  - Linux
toc: true
---

本文的目标是在整理一下安装 Ubuntu16.04 后的必要操作，包括一些必要的配置，删除不用的软件，安装比较美观的主题，以及和基本使用的软件。

<!--more-->

大家好，我是 Day。最近的文章都可以在我的个人博客看到，一些更新和纠正也会在博客上同，欢迎大家把我的博客添加到收藏夹，我也会得到督促和认可~最近买了个域名，直接浏览器输入 dday.top 就可访问我的个人博客了，欢迎来玩~

![桌面截图.png](http://upload-images.jianshu.io/upload_images/4086548-419ea594ac4d8256.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 初始配置

### 配置镜像软件源

浏览器打开 清华大学开源软件镜像站中 Ubuntu 镜像[使用帮助](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/)

Ctrl+Shift+T 打开命令行

```shell
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak  # 备份软件源配置文件
sudo gedit /etc/apt/sources.list
# 复制粘贴使用帮助中对应 Ubuntu 版本的镜像源。
sudo apt update  # 更新软件列表
```

Tip：这里说一下 apt 与 apt-get 的区别。简单来说，**apt = apt-get、apt-cache 和 apt-config 中最常用命令选项的集合**。apt 命令更易用，有进度条，克服了 apt-get  的一些错误。尽量习惯使用 apt。

### 安装 Chromium 浏览器

由于 Chromium 在官方源中直接存在，不需要添加 ppa 源，直接 install 就好。

```shell
sudo apt install chromium-browser
sudo apt remove firefox  # 卸载 FireFox，视自己喜好而定。
chromium-browser  # 打开 Chromium 浏览器
```

设为默认浏览器，更改默认搜索引擎。

## 系统清理

```shell
sudo apt upgrade
sudo apt remove libreoffice-common  # 卸载 LibreOffice
sudo apt remove unity-webapps-common  # 删除 Amazon 链接

#删除不常用的软件
sudo apt remove thunderbird totem rhythmbox empathy brasero simple-scan gnome-mahjongg aisleriot
sudo apt remove gnome-mines cheese transmission-common gnome-orca webbrowser-app
sudo apt remove webbrowser-app gnome-sudoku  landscape-client-ui-install onboard deja-dup
```

## 安装新主题

```shell
sudo apt install unity-tweak-tool  # Unity 图形化管理工具
sudo add-apt-repository ppa:noobslab/themes  # 添加主题 ppa 源
sudo add-apt-repository ppa:noobslab/icons  # 添加图标 ppa 源
sudo apt update
sudo apt install flatabulous-theme  # 安装 Flatabulous 主题
sudo apt install ultra-flat-icons  # 安装 Ultra 图标
unity-tweak-tool  # 打开图形化管理工具，在 Theme 和 Icons 里应用主题和图标
```

## 美化终端

*你需要知道你在做什么再执行命令*

```shell
sudo apt install zsh  # 安装 zsh
sudo apt install git  # 安装 Git 工具
# 下载 GitHub 上的 oh-my-zsh 脚本
wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh
sudo sh install.sh  # 安装脚本
sudo gedit /etc/passwd  # 修改最后一行，将 sh 改为 zsh
```

**注销后重新登录**，打开命令行，右键 Profiles -> Profiles references，修改终端透明度。

## 基本软件安装

### 搜狗输入法

搜索 "sougou linux"，在搜狗官网点击下载对应版本的 deb 文件，浏览器会提示要不要保存，记得 Keep。

打开系统设置，找到 Language Support。Ubuntu 是英文版安装的话这里需要先安装下中文语言，在 Install/Remove Languages 找到 Chinese(simplified)，右边打勾，点击 Apply。

```shell
cd Downloads
sudo dpkg -i sogoupinyin_***.deb  # 使用 dpkg 安装
# 会遇到错误，是缺少依赖的问题，使用下面命令自动安装缺少的依赖
sudo apt -f install
sudo dpkg -i sogoupinyin_***.deb  # 再次使用 dpkg 安装
```

安装成功后，打开系统设置，Language Support，把最下面 Keyboard input method system 改为 fcitx。

**注销后重新登录**，右上角 Keyboard 打开配置，点击 "+"，取消勾选 Only Show Current Language，找到点击Sougou Pinyin。**再次注销重新登录**就可以使用搜狗啦，可以设置自己习惯的快捷键切换，默认为 Shift。

### 安装 WPS

搜索 "wps linux"，在 WPS 官网下载对应版本的 deb 文件。也可以复制目标 deb 的链接地址，打开终端到 Downloads 目录下使用 wget 命令下载。

```shell
cd Downloads
sudo dpkg -i wps-office_***.deb
```

打开试用，更新和字体缺失的问题直接忽略就好。

### 安装 Atom 编辑器

```shell
sudo add-apt-repository ppa:webupd8team/atom
sudo apt update
sudo apt install atom
atom .  # 在当前目录下打开 Atom
```

冒似官网也有 deb 文件了，官网下载后像上面两个软件那样用 dpkg 安装也可以。

-----

以前用 Lantern 翻墙的日子，每月的流量都要省着点儿用，省着的后果就是除非必要，否则也不开。

上周买了 SS 服务之后，后台 PAC 模式常开，搜索引擎默认为 Google，视频上油管上找，简直畅快。

![小丑背影.jpg](http://upload-images.jianshu.io/upload_images/4086548-8f293d3826da64f1.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
