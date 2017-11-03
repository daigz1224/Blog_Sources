---
title: Hexo+GitHub Pages 搭建简洁优雅的个人博客
date: 2017-09-18 14:57:11
tags:
- GitHub
- Hexo
categories: Hexo
toc: true
---

本文的目标是使用 Hexo 搭建一个自己的博客，并配上非常优雅的 NexT 主题，并把博客托管到 GitHub 上，实现传说中的第三阶段。
 <!-- more -->

大家好，我是 Day。上周突然意识到个人博客的重要性，不仅是写技术相关、兴趣相关亦或是简单的碎碎念，有一个博客对于给简历增加亮点，引发自己思考甚至是提升表达能力都有很大的作用。大家通常都说网上写博客有三个阶段：


> 第一阶段，刚接触 Blog，觉得很新鲜，试着选择一个免费空间来写。

> 第二阶段，发现免费空间限制太多，就自己购买域名和空间，搭建独立博客。

> 第三阶段，觉得独立博客的管理太麻烦，最好在保留控制权的前提下，让别人来管，自己只负责写文章。
>

## 【几个概念】

**什么是 Hexo，什么是 NexT 主题？**
Hexo 是一个基于 Node.js 的静态站点生成框架，快速、简洁且高效。Hexo 主要使用 Markdown 解析文章，在几秒内，即可利用靓丽的主题生成静态网页。NexT 是其一个非常简洁优雅的主题。

**什么是 Markdown？**
Markdown 用过一些简单的标记，让你的文字实现精致的排版，实现易读易写，无需考虑美化，专注文字本身。

**什么是 GitHub ?**
GitHub 是基于 Git 技术的社交编程及代码托管网站。你可以用它对你的项目进行版本控制，也可以浏览学习、参与开发别人的开源项目，甚至可以交友。

**如何把这些名词串联起来？**

- 安装必要的软件后，使用 Hexo 创建你的本地博客，生成静态页面；
- 将静态页面托管到 GitHub 上，这样别人就可以通过公开网址访问你的博客了；
- 下载 NexT 主题美化你的博客样式。

------

## 【使用 Hexo 创建本地博客】

### 安装 Git

- Windows：下载后一路安装 [git-for-windows](https://github.com/waylau/git-for-win)（国内下载点）.
- Mac：使用 [Homebrew](http://mxcl.github.com/homebrew/), [MacPorts](http://www.macports.org/)；或下载 [安装程序](http://sourceforge.net/projects/git-osx-installer/) 安装。
- Linux (Ubuntu, Debian)：`$ sudo apt-get install git-core`
- Linux (Fedora, Red Hat, CentOS)：`$ sudo yum install git-core`


### 安装 Node.js

- Mac / Linux：使用 curl 或 wget 下载安装

```shell
$ curl https://raw.github.com/creationix/nvm/master/install.sh | sh
or
$ wget -qO- https://raw.github.com/creationix/nvm/master/install.sh | sh
# 重启终端
$ nvm install stable
```


- Windows：下载后一路安装 [Node.js](https://nodejs.org/en/)


### 安装 Hexo

- 先创建自己博客文件夹，比如 `D:/hexo`

```shell
$ cd your-hexo-site # 切换到你的本地博客文件夹
# Windows 用户直接在该文件夹下右键打开 Git Bash Here
$ npm install -g hexo-cli
```


### 初始化自己的博客静态网站

```shell
$ hexo init  # 生成一些必要的初始文件
$ npm install  # 安装依赖包
$ npm install hexo-deployer-git --save  # 为了可以将网站部署到 GitHub 上
$ hexo g  # 生成静态页面
$ hexo s  # 打开测试服务器
```
- 浏览器打开网址：localhost:4000，就是最初的样子。

-----

## 【配置 GitHub】

- 在 [GitHub 官网](https://github.com/) 注册一个账号，记得注册邮箱和用户名，建议用户名和邮箱的用户名一样，建议此时学习一些 Git 和 GitHub 的基本知识。
- 本地打开命令行或 Git Bash，配置本地需要连接的账号：

```powershell
$ git config --global user.name "your_name"  # 设置用户名
$ git config --global user.email "your_email@youremail.com"  # 设置邮箱
```

- 配置 SSH，建立本地与 GitHub 账号之间的连接密钥：

```shell
$ ssh-keygen -t rsa -C your_email@youremail.com
```

- 网页登录GitHub，点击头像 -- Settings -- SSH and GPG keys -- New SSH key，将密钥复制粘贴至此。
  *p.s. Mac / Linux 下密钥在 ~/.ssh/id_rsa.pub；Windows下密钥在 C:\Users\用户名\.ssh\id_rsa.pub*
- 验证是否成功：`$ ssh -T git@github.com`
- GitHub 上新建仓库，命名为 {your_name}.github.io，其中{your_name} 必须与你的用户名一样，这是 GitHub Pages 的特殊命名规范。
- 仓库的 Settings 里，下拉到 GitHub Pages 里，点击 source 中 None ，修改其为 master 分支，也就是作为部署 GitHub Pages 的分支，然后点击 save。

-----

## 【配置、美化本地博客】

### 安装 NexT 主题

```shell
  $ cd your-hexo-site # 切换到你的本地博客文件夹
  $ git clone https://github.com/iissnan/hexo-theme-next themes/next
```


### 配置站点_config.yml文件

- 以 D:/hexo 为例，【站点_config.yml】 即为 `D:/hexo/_config.yml`，如果你的 Windows 电脑里没有代码文本编辑器的话，这个后缀的文本应该是打不开的，我使用的是 Atom 编辑器，你也可以使用 notepad++，VS code等，其他操作系统默认都有 vi 之类的编辑器可以打开。
- 找到 language 字段：可修改值为 zh-CN
- 找到 theme 字段，修改其值为 next
- 找到最后 deploy 字段，修改如下：

```
type: git
repository: https://github.com/{your_name}/{your_name}.github.io
branch: master
```


### 部署到 GitHub

```shell
$ hexo clean  # 删除原来的静态页面
$ hexo g  # generate 生成新的静态页面
$ hexo d  # deploy 将页面部署到 GitHub 上
```

- 访问 {your_name}.github.io 就可以看到你的博客网站了！**一般部署之后需要等待一段时间博客网站才会刷新，如果你想一边配置一边看效果，可以再打开一个   Git Bash，输入`$ hexo s  # 打开本地测试服务器`，浏览器打开localhost:4000 即时查看变化。**

-----

### 自定义配置、写文章

- 可根据 [Hexo 文档](https://hexo.io/zh-cn/docs/) 和 [NexT 文档](http://theme-next.iissnan.com/getting-started.html)，自由配置；
- 主要需注意两个文件，**站点配置文件** (`D:/hexo/_config.yml`) 和**主题配置文件**(`D:/hexo/thems/next/_config.ym`) 。

### 写文章的一般流程

- 创建文章：

`$ hexo new post <title> `在 `source\_posts` 下生成 post 布局的<title>.md 文档

- 使用支持 Markdown 的编辑器打开写文章。
- 文章写完后，执行：

```shell
$ hexo clean  # 清理 public 文件夹
$ hexo g  # generate 生成静态网页
$ hexo d  # deploy 部署更新文章到 GitHub Pages
```

![卓别林睡觉.jpg](http://upload-images.jianshu.io/upload_images/4086548-d3412eef2692475c.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
