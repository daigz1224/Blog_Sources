---
title: Hexo+NexT优化部署、个性化配置、主题美化流程
tags:
  - Hexo
  - NexT
date: 2017-09-19 12:34:33
categories: Hexo
---

本文实现的目标是在 Hexo 的博客搭建之后，对于优化部署，源代码托管，域名绑定，搜索引擎优化，个性化配置及主题美化的一些操作。

<!--more-->

大家好，我是 Day。这几天折腾博客也是够呛，不过当明白了做的每一步是什么、为什么、以及怎么实现的之后，自己还是受益匪浅的。这篇文章主要整理一下自己搜集到的方法，方便以后浏览查阅。

**本文实现的目标是在 Hexo 的博客搭建之后，对于优化部署，源代码托管，域名绑定，搜索引擎优化，个性化配置及主题美化的一些操作。**

------

## 【优化部署】

### 注册 GitHub 和 Coding 并分别创建 Pages。

那为什么要注册两个网站呢？因为 Github 是国外的服务器，访问速度比较慢，而 Coding 是国内的，速度相对来说比较快，在后面 **DNS解析** 的时候可以把国内的解析到 Coding，国外的解析到 Github，完美。

具体请查看 *[Coding Page](https://coding.net/help/doc/pages/index.html)* 、 *[Github Page](https://pages.github.com/)*

1. 注册 GitHub 注册好之后，可以使用 GitHub 的账号登陆 Coding。
2. 在 GitHub 创建 Repository，名字的格式为 {your_name}.github.io，这是 GitHub Pages 的特殊命名规范，在项目的 Setting 中确认 GitHub Pages 正常启用，显示绿框 Your site is published at ...
3. 在 Coding 中创建项目，项目名为 {your_name}.coding.me，这是 Coding Pages 的特殊命名规范。点击项目 -> 代码 -> Pages 服务，若没有开启则点开启。

### 配置 SSH 与 Git：绑定个人电脑

```
ssh-keygen -t rsa -C your_email@youremail.com  # 生成 SSH Key
```
找到 `.ssh/id_rsa.pub` 文件，复制密钥，粘贴到 GitHub 和 Coding 中。

**GitHub**：右上角 头像 -> Settings -> SSH nd GPG keys -> New SSH key 。把公钥粘贴到 key 中，填好 title 并点击 Add SSH key。
**Coding**：点击 账户 -> SSH公钥 -> 输入key再点击 添加

### 验证是否配对以及配置账户

```
ssh -T git@github.com  # 验证 GitHub
ssh -T git@git.coding.net  # 验证 Coding
git config --global user.name your name  # 绑定用户名
git config --global user.email your_email@youremail.com  # 绑定邮箱
```

### 部署到 Github Pages 与 Coding Pages

安装 Git 部署插件：

`npm install hexo-deployer-git --save`

打开【站点配置文件】`hexo_blog\_config.yml`，在底部修改部署配置：

```
# Deployment
## Docs: https://hexo.io/docs/deployment.html
deploy:
  type: git
  repo:
    github: git@github.com:{your_name}/{your_name}.github.io.git,master
    coding: git@git.coding.net:{your_name}/{your_name}.git,master
```

------

## 【源代码托管】

源代码用于生成静态页面，一直放在本地的话，用其他电脑就不方便修改了。所以可以源代码托管到 GitHub 上。网上将源代码托管到博客的 Repository 的新建分支中会出现主题文件夹无法同步的问题。这里采用新建 Repository 的方法实现源代码托管。

### 上传源代码到 GitHub

确认源代码下 .gitignore 内容：

```
.DS_Store
Thumbs.db
db.json
*.log
node_modules/
public/
.deploy*/
```

GitHub 上新建 hexo_blog 仓库，复制 Clone or download 下的在线地址。

```
git init
git remote add origin <server>  # hexo_blog 仓库的在线地址
git add . #添加 blog 目录下所有文件，注意有个'.' ( .gitignore 里面声明的文件不在此内)
git commit -m "hexo source first add"  # 添加更新说明
git push -u origin master  # 推送更新到 git 上
```

### 下载源代码到另一台电脑

配置好 Git，SSH，初始化 Hexo 环境后，新建博客的本地文件夹：

```
git init  # 将目录添加到版本控制系统中
git remote add origin <server>  # 同上
git fetch --all  # 将 git 上所有文件拉取到本地
git reset --hard origin/master  # 强制将本地内容指向刚刚同步 git 云端内容
```

reset 对所拉取的文件不做任何处理，此处不用 pull 是因为本地尚有许多文件，使用 pull 会有一些版本冲突，解决起来也麻烦，而本地的文件都是初始化生成的文件，较拉取的库里面的文件而言基本无用，所以直接丢弃。

### 从 GitHub 仓库中更新源代码

```
git pull  # 更新源代码
```

------

## 【域名绑定】

### 购买域名

**[万网](https://wanwang.aliyun.com/)**或其他平台。

### 域名解析

1. 进入工作台，点击域名，然后解析。

2. 第一次可能需要填写个人信息，填完了，点击上面的域名解析 -> 解析设置 -> 添加解析，记录类型选 A 或 CNAME， A 记录的记录值就是 ip 地址， GitHub 提供了两个 IP 地址， 192.30.252.153 和 192.30.252.154，随便填一个就行，解析记录设置两个 www 和不填，线路就默认就行了，CNAME 记录值填你的 Coding 的博客网址。
3. 如果选择 A 记录，就要在网站根目录新建 CNAME 文件，里面填写注册的域名，之后修改站点配置文件，把站点地址更新成新的绑定的域名即可。

------

## 【搜索引擎优化】

### 网站验证

- **[百度提交入口](http://zhanzhang.baidu.com/linksubmit/url)**
- **[Google提交入口](https://www.google.com/webmasters/tools/home?hl=zh-CN)**
- **[360提交入口](http://info.so.360.cn/site_submit.html)**

### 添加并提交 sitemap

安装 Hexo 的 sitemap 网站地图生成插件:

```
npm install hexo-generator-sitemap --save
npm install hexo-generator-baidu-sitemap --save
```

在【站点配置文件】`hexo_blog\_config.yml` 中添加如下代码。

```
# hexo sitemap
sitemap:
  path: sitemap.xml
baidusitemap:
  path: baidusitemap.xml
```

配置成功后，会生成 sitemap.xml 和 baidusitemap.xml，前者适合提交给谷歌搜素引擎，后者适合提交百度搜索引擎。

### 主动推送

安装主动推送插件：

```
﻿npm install hexo-baidu-url-submit --save
```

在根目录下，把以下内容配置到站点配置文件中：

```
baidu_url_submit:
  count: 3 ## 比如3，代表提交最新的三个链接
  host: www.henvyluk.com ## 在百度站长平台中注册的域名
  token: your_token ## 请注意这是您的秘钥，请不要发布在公众仓库里!
  path: baidu_urls.txt ## 文本文档的地址，新链接会保存在此文本文档里
```

your_token 可在百度站长的接口调用地址找到。

查看【站点配置文件】中 url 的值， 必须包含是百度站长平台注册的域名（一般有 www）， 比如：

```
url: http://www.ookamiantd.top
root: /
permalink: :year/:month/:day/:title/
```

接下来添加一个新的 deploy 的类型：

```
# Deployment
## Docs: https://hexo.io/docs/deployment.html
deploy:
- type: baidu_url_submitter
- type: git
  repo:
    github: git@github.com:masteranthoneyd/masteranthoneyd.github.io.git,master
    coding: git@git.coding.net:ookamiantd/ookamiantd.git,master
```

执行hexo deploy的时候，新的连接就会被推送了。这里讲一下原理：

- 新链接的产生，hexo generate 会产生一个文本文件，里面包含最新的链接
- 新链接的提交，hexo deploy 会从上述文件中读取链接，提交至百度搜索引擎

### 自动推送

把 next 主题配置文件中的 baidu_push 设置为 true，就可以了。

------

## 【个性化配置】
- 参考 Hexo 官网配置说明文档：[Hexo 文档](https://hexo.io/zh-cn/docs/)：主要是【站点配置文件】
 `hexo_blog\_config.yml `的参数设置。

- 参考 NexT 官网配置说明文档：[NexT 主题配置](http://theme-next.iissnan.com/theme-settings.html#blogroll)：主要是【主题配置文件】`hexo_blog\themes\next\_config.yml` 的参数设置。

------

## 【主题美化】

### 把侧边栏头像变成圆形，并且鼠标停留在上面发生旋转效果

修改 `themes\next\source\css\_common\components\sidebar\sidebar-author.styl`：

```
.site-author-image {
  display: block;
  margin: 0 auto;
  padding: $site-author-image-padding;
  max-width: $site-author-image-width;
  height: $site-author-image-height;
  border: site-author-image-border-color;
  /* start*/
  border-radius: 50%
  webkit-transition: 1.4s all;
  moz-transition: 1.4s all;
  ms-transition: 1.4s all;
  transition: 1.4s all;
  /* end */
}
/* start */
.site-author-image:hover {
  background-color: #55DAE1;
  webkit-transform: rotate(360deg) scale(1.1);
  moz-transform: rotate(360deg) scale(1.1);
  ms-transform: rotate(360deg) scale(1.1);
  transform: rotate(360deg) scale(1.1);
}
/* end */
```

### 为 NexT 主题的主页文章添加阴影效果

打开 `themes/next/source/css/_custom/custom.styl` 文件添加：

```
.post {
  margin-top: 60px;
  margin-bottom: 60px;
  padding: 25px;
  -webkit-box-shadow: 0 0 5px rgba(202, 203, 203, .5);
  -moz-box-shadow: 0 0 5px rgba(202, 203, 204, .5);
 }
```

### 为 NexT 主题添加 canvas_nest 背景特效

打开 `/next/_config.yml`，修改 canvas_nest 参数：

```
# Canvas-nest
canvas_nest: true

# three_waves
three_waves: false

# canvas_lines
canvas_lines: false

# canvas_sphere
canvas_sphere: false

# Only fit scheme Pisces
# Canvas-ribbon
canvas_ribbon: false
```

###  设置网站的图标Favicon

在 [EasyIcon](http://www.easyicon.net/) 中找一张（32*32）的 ico 图标,或者去别的网站下载或者制作，并将图标名称改为 favicon.ico，然后把图标放在 `/themes/next/source/images` 里，并且修改【主题配置文件】`hexo_blog\themes\next\_config.yml`：

```
# Put your favicon.ico into `hexo-site/source/` directory.
favicon: /favicon.ico
```

### 修改字体大小

打开 `\themes\next\source\css\ _variables\base.styl` 文件，将 `$font-size-base` 改成 `16px`，如下所示：

```
$font-size-base =16px
```

### 为NexT主题添加文章阅读量统计功能

 [为NexT主题添加文章阅读量统计功能](https://notes.wanghao.work/2015-10-21-%E4%B8%BANexT%E4%B8%BB%E9%A2%98%E6%B7%BB%E5%8A%A0%E6%96%87%E7%AB%A0%E9%98%85%E8%AF%BB%E9%87%8F%E7%BB%9F%E8%AE%A1%E5%8A%9F%E8%83%BD.html#%E9%85%8D%E7%BD%AELeanCloud)


![摩根弗里曼.jpg](http://upload-images.jianshu.io/upload_images/4086548-92044ec5a048818e.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

捯饬下来倒是不难，但是需要细心一点，关于评论功能大家可以试试 [gitment](https://imsun.net/posts/gitment-introduction/) 项目，想法很好，效果也很棒，期待 NexT 主题作者将其收入库中。
