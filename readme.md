# 环境配置

## 涉及的安装包

### VSCode

校内加速链接：[VSCode](https://share.319.ccsn.dev/d/安装包/VSCodeUserSetup-x64-1.77.1.exe) [https://share.319.ccsn.dev/d/安装包/VSCodeUserSetup-x64-1.77.1.exe](https://share.319.ccsn.dev/d/安装包/VSCodeUserSetup-x64-1.77.1.exe)

直接下载链接：<https://code.visualstudio.com/sha/download?build=stable&os=win32-x64-user>

### Micromamba

校内镜像：[micromamba](https://share.319.ccsn.dev/d/%E5%AE%89%E8%A3%85%E5%8C%85/micromamba.exe) <https://share.319.ccsn.dev/d/%E5%AE%89%E8%A3%85%E5%8C%85/micromamba.exe>
校外镜像：[micromamba](https://share.ccsn.dev/d/%E5%AE%89%E8%A3%85%E5%8C%85/micromamba.exe) <https://share.ccsn.dev/d/%E5%AE%89%E8%A3%85%E5%8C%85/micromamba.exe>
不用安装，直接复制到与本文件同目录下即可

微软会弹病毒，进入windows安全中心-病毒和威胁防护-管理设置-添加排除项目，将其加入排除项

### Git

校内加速连接 [Git](https://share.319.ccsn.dev/d/%E5%AE%89%E8%A3%85%E5%8C%85/Git-2.40.1-64-bit.exe) （2.40.1）

## 其他的一些配置

### 打开VSCODE

页面最下端往上拉，选择终端输入下列代码

#### 设置源为清华源，配置如下

```bash
cp .condarc ~/.condarc
```

#### 创建环境、安装依赖

```bash
.\micromamba.exe  env create --file environment.yml
```

如果报错xxx is missing，关杀毒软件特别是360后重试。

#### 数据库选择

来源：[UCSC Xena (xenabrowser.net)](https://xenabrowser.net/datapages/)，只勾选右侧TCGA Hub，使用其中含有exon expression RNAseq的数据，将数据库右侧括号内名字放入dataset_name，可以使用两个数据库，method一般使用为normal，对比两个数据库差异时为diff

```sh
dataset_name = ['LUAD']
method = 'normal'
```

#### 数值调整

绘图代码中可更改该行代码右侧['n'] <= 的数值来控制筛选出靶标miRNA的最大数量

```
select = results_sorted_found_coef[results_sorted_found_coef['reproduce'] == True][results_sorted_found_coef['n'] <= 6].copy().iloc[0]
```

训练模型中可以更改max_run_num的值来控制运行次数，次数越大越准确但耗时更长

```
max_run_num = 600
```

#### 运行程序

点击左侧tcga，选择内核igem(Python 3.8.16),点击上侧全部运行

最下方结果图下侧会出现靶标以及对应的权重
