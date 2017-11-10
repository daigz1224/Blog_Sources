---
title: Win10+Ubuntu16.04 双系统 UEFI+GPT, SDD+HDD 解决方案
date: 2017-09-14 22:42:28
tags:
- Ubuntu
- Windows
categories: Linux
toc: true
---
本文的目标是在 SSD 中安装 Win10，并分出 20 G 用来放 Ubuntu 的系统， HDD 中分出 80 G 给 Ubuntu 的日常文件使用， Ubuntu 直接使用 Win10 的 efi 分区启动。
 <!-- more -->
大家好，我是 Day , 昨天再次给自己的电脑装了双系统之后就想整理一下流程留待以后参考，省得自己重复折腾。关于双系统的安装网上很多资料都很老了，很多时候在不知道你的电脑是不是 Legacy Bios 的时候就推荐用 EasyBCD 什么的修改引导，还有硬盘 HDD 与固态 SSD 的分区表是传统的 MBR 还是新的 GPT 等，都需要确认一下，第二个系统 Ubuntu 在安装的时候，怎样分区比较好，怎样使用引导比较好，这些都要根据自己的情况量身定制。

## 2017/11/10 更新：

**删除 ubuntu 后还是会出现 grub 的问题，解决了之后再重装又回到了 grub 的问题。所以我现在的最新双系统策略是两块盘两个系统，SSD 上装Ubuntu，HDD 上装 Win10。**



## 一些概念

- **Legacy Bios 和 UEFI ?**
  具体细节不用考虑，只需要知道 Legacy Bios 是传统的 Bios，后来被 EFI 取代，再后来更名为 UEFI。近几年的电脑一般都支持 UEFI 引导。

- **如何查看自己的电脑是否支持 UEFI？**
  快捷键 Win + R，输入 msinfo32，右边可以看到 Bios 模式。

- **MBR 和 GPT ?**
  硬盘分区表，随着磁盘容量越来越大，传统的 MBR 已经不能满足大容量磁盘的需求了。GPT 意为 GUID
   分区表，这是一个正逐渐取代 MBR 的新标准。GPT 必须使用 UEFI 的主板引导。

- **如何查看自己的硬盘是否为 GPT？**
  右键开始图标，进入磁盘管理，右键磁盘（SSD 和 HDD 都看一下），查看属性。

- **如何将自己的 MBR 动态磁盘转换为 GPT 磁盘？**
  刚才在磁盘管理里右键磁盘的时候有转换成 GPT 磁盘的选项；
  重装系统时可以用 Shift + F10 调出 CMD 命令行，使用 [DiskPart](https://technet.microsoft.com/zh-cn/library/cc766465(v=ws.10).aspx) 工具可以进行转换。
  **注意**：分区表转换有风险，为了确保安全建议用户将分区表和硬盘数据备份。*

总而言之，**UEFI + GPT 是最好的方案**，重装系统的时候最好将自己的主板和硬盘转换成 UEFI + GPT.

我的老电脑是一块 128 G 固态硬盘 SSD，一块 1 T 的普通硬盘 HDD，都支持 UEFI+GPT， HDD 中有大量文件不方便备份转移，不想潇洒的全盘格式化。本文的目标是在 SSD 中安装 Win10，并分出 20 G 用来放 Ubuntu 的系统， HDD 中分出 80 G 给 Ubuntu 的日常文件使用， Ubuntu 直接使用 Win10 的 efi 分区启动。

## Win10 安装

1. 准备 win10 x64 的 ISO 镜像，备份清空 U 盘，备份清理 C 盘，主要是桌面的文件。
2. 查看主板 Bios 和 硬盘（SSD + HDD）的格式。
3. 使用 [Rufus](http://rufus.akeo.ie/?locale=zh_CN) 快速创建启动盘，创建时注意分区方案和目标系统类型，另外文件系统建议 NTFS 。
4. 重启，F12 选择 U 盘启动，通过 [DiskPart](https://technet.microsoft.com/zh-cn/library/cc766465(v=ws.10).aspx)  工具将 SSD 盘格式化，将 Win10 系统在 SSD 中。
## Ubuntu 16.04 安装

1. Win10 下打开磁盘管理，将 SSD 磁盘压缩卷，分出 20 G （20*1024=20480 MB），用来放 Ubuntu 系统；再将 HDD 压缩卷，分出 80 G （81920 MB，有点大了，可根据自己的情况，40G 足矣），用来放Ubuntu 日常文件和虚拟内存空间。
2. 使用 [Rufus](http://rufus.akeo.ie/?locale=zh_CN) 制作启动盘，创建时注意分区方案和目标系统类型，另外文件系统建议 NTFS 。（提示两种写入镜像的方法，推荐的那个出现引导问题就试试第二个 DD 镜像写入。)
3. 打开电源选项，选择电源按钮的功能，更改当前不可用的设置，关机设置里取消快速启动。
4. 重启 F2 去 Bios 里面取消 secure boot（我的联想 y480 没有这个）
5. 重启 F12 选择 U 盘启动，选择 Install Ubuntu。
6. 到 Installation type（安装类型）的时候选择最后一个 Something else，手动分区。
7. 将 SSD 的 20 G （即 sda 里的 free space 全部分给根目录，即 Ext4 jopurnaling file system，Mount point 设为 “ / ”。
8. 将 HDD 上的 free space，把 两倍于内存的空间 分给 swap 交换分区（休眠时会把内存中内容
   dump 到交换分区中），我的内存 16 G，考虑到可能会使用大内存，故将 32 G （32768 MB） Use as swap。
9. HDD 上剩余的 free space 全部分给 /home，即 Ext4 jopurnaling file system， Mount point 设为 “ /home ”。
10. **重要：**下面的 Device for boot loader installation 选择 Windows Boot Manager，即与windows共用一个 efi 分区，故不需要 /boot。
11. Install Now！
## 卸载 Ubuntu

把 Ubuntu 卸载掉之所以要单独拿出来提，主要是因为我在大四折腾时，看了百度上好多老的攻略，用 EasyBCD 之类的改引导项然后删除 Ubuntu 后出现各种问题最后还是重装了 Win10。这里需要说明的是，EasyBCD 是用在 Legacy Bios 上的，在 UEFI 上我们需要用的是 EasyUEFI。

1. Win10 下载安装 [EasyUEFI](http://www.easyuefi.com/index-cn.html)
2. 打开 EasyUEFI，管理 EFI 启动项，删除 ubuntu 的启动项。
3. 删除 SSD 和 HDD 中分给 Ubuntu 的那两个分区即可。
-------------------
折腾是件好事情，这些年来每次装系统都有不同的收 cuo 获 wu，但最后找到一个较优的方案解决了之后还是比较爽的，顺便锻炼了自己查找资料的能力。不由得吐槽一下，搜索引擎最好还是用 Google，我总能在前几个条目找到自己想要的内容。

![这个杀手不太冷](http://upload-images.jianshu.io/upload_images/4086548-b0ff68db4f836176.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
