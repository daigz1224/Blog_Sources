---
title: AWS 上使用 MXNet 的环境配置
tags:
  - AWS
  - MXNet
  - Ubuntu
categories:
  - Linux 
mathjax: false
toc: true
date: 2017-12-13 16:10:09
---

简单记录下 AWS 竞价实例的使用，CUDA，Miniconda，MXNet 以及配置 gluon 虚拟环境。

<!--more-->

## AWS 竞价实例

1. 创建实例

2. 选择 Ubuntu Server 16.04 LTS 64-bit 的 AMI 或自己保存的 AMI

3. 选择 p2.xlarge 或其他 GPU 计算的实例类型

4. 存储大小可以填大一些，比如 40 GiB

5. 使用已有的 key pair 或创建一个新的

6. 启动完成后，回到 EC2 Dashboard，右键 -> 连接

7. 本地打开带有 ssh 功能的终端：

   ```shell
   # 如果 ***.pem 未设为不公开可见
   chmod 400 ***.pem
   # 使用 key pair 建立 ssh 连接
   ssh -i "***.pem" ubuntu@***.amazonaws.com
   # 更新并安装编译需要的包
   sudo apt-get update && sudo apt-get install -y build-essential git libgfortran3
   ```

## 安装 CUDA

```shell
# 安装 CUDA 8.0,如果 mxnet-cu90 出来后，可以考虑安装 CUDA 9.0 版本
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
# 安装
sudo sh cuda_8.0.61_375.26_linux-run
# 读 license 的时候，太长不看，按 q
```

回答问题：

```
accept/decline/quit: accept
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 375.26?
(y)es/(n)o/(q)uit: y
Do you want to install the OpenGL libraries?
(y)es/(n)o/(q)uit [ default is yes ]: y
Do you want to run nvidia-xconfig?
(y)es/(n)o/(q)uit [ default is no ]: n
Install the CUDA 8.0 Toolkit?
(y)es/(n)o/(q)uit: y
Enter Toolkit Location
 [ default is /usr/local/cuda-8.0 ]:
Do you want to install a symbolic link at /usr/local/cuda?
(y)es/(n)o/(q)uit: y
Install the CUDA 8.0 Samples?
(y)es/(n)o/(q)uit: n
```

```shell
# 检查安装是否成功
nvidia-smi
# 将 CUDA 加入 Library path 方便之后安装的库找到
echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/cuda-8.0/lib64" >>.bashrc
# 生效
bash
```

## 安装 MXNet 及配置虚拟环境

```shell
# 下载安装 Miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

回答问题：

```
Do you accept the license terms? [yes|no]
[no] >>> yes
Do you wish the installer to prepend the Miniconda3 install location
to PATH in your /home/ubuntu/.bashrc ? [yes|no]
[no] >>> yes
```

```shell
# 生效
bash
```

根据教程，从 `environment.yml` 文件中建立虚拟环境，文件内容如下：

```
name: gluon
dependencies:
- python
- jupyter
- matplotlib
- pandas
- pip:
  - requests
  - mxnet>=0.11.1b20171106
```

```shell
# 创建 gluon 的虚拟环境
conda env create -f environment.yml
# 启动 gluon 虚拟环境=
source activate gluon
```

文件中默认 MXNet 是 CPU 老版本，卸载更新为 GPU 新版本：

```shell
pip uninstall -y mxnet
pip install --pre mxnet-cu80
```

运行 Jupyter notebook 并映射到本地：

```shell
# 可选：安装 notedown 插件
pip install https://github.com/mli/notedown/tarball/master
jupyter notebook --generate-config
echo "c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'" >>~/.jupyter/jupyter_notebook_config.py
# 云服务器运行 jupyter notebook
jupyter notebook
# 1. 本地通过8888端口打开
ssh -i "***.pem" -L8888:localhost:8888 ubuntu@***.amazonaws.com
# 然后将云端 jupyter log 的 URL 复制到本地浏览器
# 2. 本地通过其他端口（8889）打开
ssh -i "***.pem" -N -f -L localhost:8889:localhost:8888 ubuntu@***.amazonaws.com
# 然后本地打开 localhost:8889,填写 jupyter log 的 token 值
```



参考文章：

[AWS 上运行 gluon 教程](http://zh.gluon.ai/chapter_preface/aws.html#)