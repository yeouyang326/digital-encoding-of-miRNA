# 环境配置

## 涉及的安装包

### VSCode
校内加速链接：[VSCode](https://share.319.ccsn.dev/d/安装包/VSCodeUserSetup-x64-1.77.1.exe) [https://share.319.ccsn.dev/d/安装包/VSCodeUserSetup-x64-1.77.1.exe](https://share.319.ccsn.dev/d/安装包/VSCodeUserSetup-x64-1.77.1.exe)

直接下载链接：https://code.visualstudio.com/sha/download?build=stable&os=win32-x64-user

### MiniConda

清华镜像：[Miniconda](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe) https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe

## 其他的一些配置

### 在Shell里完成

把代码块里的东西全部复制，再粘贴进shell。

#### 在 AnaConda PowerShell Prompt 的用户配置中初始化 conda

AnaConda PowerShell Prompt 可以在开始菜单里找找

```powershell
conda init powershell
```

然后关闭AnaConda PowerShell Prompt.

#### 设置源为清华源，配置如下

```bash
conda config --set show_channel_urls yes
```

再通过vscode，打开 ~/.condarc 文件，添加如下内容

```bash
auto_activate_base: false
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

#### 创建环境、安装依赖

```bash
conda env create --file environment.yml
```

#### 激活依赖

```bash
conda activate igem
```

### 在 VS Code 里完成

安装 Git, Jupyter Notebook

选择环境
