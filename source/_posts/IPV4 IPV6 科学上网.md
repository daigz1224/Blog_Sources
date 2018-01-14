---
title: IPV4 IPV6 科学上网
tags:
  - Linux
  - 科学上网
  - Shadowsocks
categories:
  - Linux
mathjax: false
toc: true
date: 2018-01-14 13:25:51
---

欠费了，正好换个便宜的服务器，简单记录下流程。

<!--more-->

## 1. 购买服务器

- [官网](https://www.vultr.com/)
- 地点无所谓，选还有最便宜套餐的地区。
- 操作系统：Ubuntu 14.04 LTS 64bit
- Additional Features 勾选 Enable IPV6
- 搞定，开始部署。

## 2. IPV4配置

- Git Bash：

```shell
#以root用户身份访问服务器
ssh root@你的服务器地址
#输入密码，Vlutr网站的Servers信息里有
```

- 软件安装

```shell
apt-get install build-essential
apt-get install python-pip
pip install shadowsocks
```

- 新建 Shadowsocks 配置

```shell
vi /etc/shadowsocks.json
```

- 配置内容：注意修改 server_port 和 password 。

```
{
　　"server":"::",
　　"server_port": 1234,
　　"local_port": 1080,
　　"timeout":300,
　　"password": "yourpassword",
　　"method":"aes-256-cfb",
　　"fast_open":true,
　　"workers":1
}
```

- 运行 Shadowsocks

```shell
ssserver -c /etc/shadowsocks.json -d start #运行
ssserver -c /etc/shadowsocks.json -d stop #停止
```

## 3. IPV6配置

- 查看服务器的 Setting 界面的 networking configuration
- 找到 Ubuntu 12.xx - 15.xx 的脚本内容
- 修改 interfaces 的配置：

```shell
vi /etc/network/interfaces
```

```
auto lo
iface lo inet loopback
 
auto eth0
iface eth0 inet static
    address 207.246.115.50
    netmask 255.255.254.0
    gateway 207.246.114.1
    dns-nameservers 108.61.10.10
    post-up ip route add 169.254.0.0/16 dev eth0
 
iface eth0 inet6 static
    address 2001:19f0:9002:1af1:5400:01ff:fe55:f142
    netmask 64
    dns-nameservers 2001:19f0:300:1704::6
```

- 重启 Shadowsocks 服务

## 4. 客户端使用

- Windows：[shadowsocks-windows](https://github.com/shadowsocks/shadowsocks-windows)
- Android：[shadowsocks-android](https://github.com/shadowsocks/shadowsocks-android)
- Ubuntu：[shadowsocks-qt5](https://github.com/shadowsocks/shadowsocks-qt5)