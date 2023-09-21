# 环境配置

## 涉及的安装包

### VSCode

校内加速链接：[VSCode](https://share.319.ccsn.dev/d/安装包/VSCodeUserSetup-x64-1.77.1.exe) [https://share.319.ccsn.dev/d/安装包/VSCodeUserSetup-x64-1.77.1.exe](https://share.319.ccsn.dev/d/安装包/VSCodeUserSetup-x64-1.77.1.exe)

直接下载链接：<https://code.visualstudio.com/sha/download?build=stable&os=win32-x64-user>

### Micromamba

校内镜像：[micromamba](https://share.319.ccsn.dev/d/%E5%AE%89%E8%A3%85%E5%8C%85/micromamba.exe) <https://share.319.ccsn.dev/d/%E5%AE%89%E8%A3%85%E5%8C%85/micromamba.exe>
校外镜像：[micromamba](https://share.ccsn.dev/d/%E5%AE%89%E8%A3%85%E5%8C%85/micromamba.exe) <https://share.ccsn.dev/d/%E5%AE%89%E8%A3%85%E5%8C%85/micromamba.exe>
不用安装，直接复制到与本文件同目录下即可

### Git

校内加速连接 [Git](https://share.319.ccsn.dev/d/%E5%AE%89%E8%A3%85%E5%8C%85/Git-2.40.1-64-bit.exe) （2.40.1）

## 其他的一些配置

### 在Shell里完成

把代码块里的东西全部复制，再粘贴进shell。

#### 设置源为清华源，配置如下

```bash
cp .condarc ~/.condarc
```

#### 创建环境、安装依赖

```bash
.\micromamba.exe  env create --file environment.yml
```

### 在 VS Code 里完成

安装 Git, Jupyter Notebook

选择环境
