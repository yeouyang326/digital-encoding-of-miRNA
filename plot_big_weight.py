#plot:big weights
#%%
# 设置数据集名称
dataset_name = ['LUAD', 'LUSC']
# 设置俩： ['LUAD', 'LUSC'
# 单一个： ['LUAD'] 
method = 'normal' # diff是上面俩数据集的差异，normal是和正常样本的差异
# method = 'diff'
import requests
import os
import gzip
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from sklearn import datasets
import warnings
import GEOparse
from typing import Tuple, List
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score,average_precision_score
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
from multiprocessing import Pool, cpu_count
import ast
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LinearRegression, RidgeClassifier, Lasso, LassoCV, LogisticRegression, LogisticRegressionCV, ElasticNet,ElasticNetCV,SGDClassifier,RidgeClassifierCV
from sklearn.svm import SVC, LinearSVC, NuSVC
import matplotlib.pyplot as plt
import squarify
warnings.filterwarnings('ignore')
def download_and_parse_data(dataset_name):
    pkl_file_path = './datasets/' + 'TCGA-' + dataset_name + '.mirna_transposed.pkl'
    if os.path.exists(pkl_file_path):
        df_t = pd.read_pickle(pkl_file_path)
    df_t = df_t.apply(pd.to_numeric)
    return df_t
def find_healthy_samples(df):
    healthy_samples_df = df[df['Status'] == 0]
    return healthy_samples_df
if method == "diff":
    dataA = download_and_parse_data(dataset_name[0])
    dataB = download_and_parse_data(dataset_name[1])

    A_up = dataA[dataA['Status'] == 1]
    B_up = dataB[dataB['Status'] == 1]

    B_up['Status'] = 0
    df_transposed = pd.concat([A_up, B_up], ignore_index=True)
elif method == "normal":
    df_transposed = pd.DataFrame()
    for name in dataset_name:
        df_transposed = df_transposed.append(download_and_parse_data(name))
train_data = df_transposed.apply(pd.to_numeric)
X_train, X_test, y_train, y_test = train_test_split(train_data.drop(['Status'], axis=1), train_data['Status'], test_size=0.2, random_state=42)
# # combine test data = train_data
trainging_data = pd.concat([X_train, y_train], axis=1)
validation_data = pd.concat([X_test, y_test], axis=1)
# feature = ['hsa-mir-182', 'hsa-mir-203a', 'hsa-mir-21', 'hsa-mir-148a', 'hsa-mir-143']
# feature_target = ['hsa-mir-182', 'hsa-mir-203a', 'hsa-mir-21', 'hsa-mir-148a', 'hsa-mir-143', 'status']
# weight = [[5.0, 4.0, 25.0, 25.0, -1.0]]
# threshold = 0
# feature = ['hsa-mir-182', 'hsa-mir-203a', 'hsa-mir-21', 'hsa-mir-148a']
# #feature_target = ['hsa-mir-182', 'hsa-mir-203a', 'hsa-mir-21', 'hsa-mir-148a', 'hsa-mir-143', 'status']
# weight = [9.0, 4.0, 15.0, 5.0]
# threshold = 3718495.65365987

feature = ['hsa-mir-182', 'hsa-mir-21', 'hsa-mir-148a', 'hsa-let-7b', 'hsa-let-7a-3', 'hsa-mir-143', 'hsa-mir-30a']
# weight = [8, 1.0, 2.0, 8.0, -4.0, -1.0, 5.0]

weight = [16.0, 4.0, 10.0, -1.0, -25.0, -2.0, -15.0]#0：99.91
threshold = 0
#%%
#validation
##((testdata_mean['pos_sum'] - testdata_mean['neg_sum'] - threshold > 0) -> 没有患病的threshold的定义
# 仅选择特征和最终结果. 输出末2
#thereshold 是traindata_mean['sum']
ID = feature.copy()
ID = np.array(ID).reshape(-1)
# 加('Status')
ID = np.append(ID, ['Status'])
ID.astype('str')
testdata = validation_data[ID] #train_data测试集 -> testdata 
# 把testdata的columns的名字改成feature
testdata.columns = feature + ["target"]
testdata.tail(2)
weight = np.array(weight).reshape(-1)
weight = np.append(weight, [1])# weight 的最后一位是target，不需要计算，设置为1
# 计算每一种基因表达量的平均值,给定的权重weight，算每一种基因testdata_mean = testdata  * weight
testdata_mean = testdata  * weight
weight = weight.reshape(-1)
weight = weight[:-1]
# 计算每一行的带权重（每行每一个数据已经经过权重处理）testdata_mean['sum'] 和，要去掉最后一列
testdata_mean['sum'] = (testdata_mean.sum(axis=1) - testdata_mean['target']) 
zipped = zip(weight, feature)
# select weight > 0，positive mirna，为feature_pos
feature_pos = [x for x in zipped if x[0] > 0]
feature_pos = [x[1] for x in feature_pos]
testdata_mean['pos_sum'] = testdata_mean[feature_pos].sum(axis=1)
# neg
testdata_mean['neg_sum'] = testdata_mean['pos_sum'] - testdata_mean['sum']
# error
testdata_mean['error_indices'] = ((testdata_mean['pos_sum'] - testdata_mean['neg_sum'] - threshold > 0) & (testdata_mean['target'] == 0)) | ((testdata_mean['pos_sum'] - testdata_mean['neg_sum'] - threshold < 0) & (testdata_mean['target'] == 1))

#plot #plot #plot #plot #plot #plot #plot #plot #plot #plot 
#plot #plot #plot #plot #plot #plot #plot #plot #plot #plot 

# %%
#plot 
#1.threhold
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 3      # 边框粗细
plt.rcParams['axes.labelweight'] = 'bold'  # 轴标签字体粗细
plt.rcParams['xtick.major.width'] = 3   # x轴刻度粗细
plt.rcParams['ytick.major.width'] = 3   # y轴刻度粗细
fig, axs = plt.subplots(1, 1, figsize=(10, 10))
positive = len(testdata_mean['pos_sum'])
negative =  len(testdata_mean['neg_sum'])

axs.scatter(testdata_mean['pos_sum'], testdata_mean['neg_sum'], c=['gray' if target == 0 else 'skyblue' for target in testdata_mean['target']], s=130, alpha=1)
#axs.scatter(testdata_mean['pos_sum'], testdata_mean['neg_sum'], c=testdata_mean['target'], cmap='Spectral', s=130, alpha=0.6)#c=testdata_mean['target'], cmap='coolwarm', alpha=0.8
axs.set_xlabel('Number of positive expression genes', fontsize=25)
axs.set_ylabel('Number of negative expression genes', fontsize=25)
x_vals = np.array(axs.get_xlim())
y_vals = x_vals - threshold
axs.plot(x_vals, y_vals, linestyle='--', color='lightgray')
axs.set_xlim(0, axs.get_xlim()[1])
axs.set_ylim(0, axs.get_ylim()[1])
axs.tick_params(axis='x', labelsize=30)  # 设置 x 轴刻度的字体大小和旋转角度
axs.tick_params(axis='y', labelsize=30)
axs.fill_between(x_vals+100000, y_vals+100000, color='lightskyblue', alpha=0.1)
axs.fill_betweenx(y_vals+100000,x_vals+100000, 0, color='coral', alpha=0.1)
num_errors = len(testdata_mean['pos_sum'][testdata_mean['error_indices']])
print(positive,negative,num_errors)

axs.scatter(testdata_mean['pos_sum'][testdata_mean['error_indices']], testdata_mean['neg_sum'][testdata_mean['error_indices']], color='black', marker='x', s=100)
# legend = axs.legend(frameon=False)
# for text in legend.get_texts():
#     text.set_fontsize(100)
plt.savefig('/Users/ye/Documents/py/threhold_train1.png')
plt.show()
# %%
#2. AUROC
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 3      # 边框粗细
plt.rcParams['axes.labelweight'] = 'bold'  # 轴标签字体粗细
plt.rcParams['xtick.major.width'] = 3   # x轴刻度粗细
plt.rcParams['ytick.major.width'] = 3   # y轴刻度粗细
axs.tick_params(axis='x', labelsize=25)  # 设置 x 轴刻度的字体大小和旋转角度
axs.tick_params(axis='y', labelsize=25)
fig, axs = plt.subplots(1, 1, figsize=(10, 10))


fpr, tpr, thresholds = roc_curve(testdata_mean['target'], testdata_mean['pos_sum'] - testdata_mean['neg_sum'])
auc_pr = average_precision_score(testdata_mean['target'], testdata_mean['pos_sum'] - testdata_mean['neg_sum'])
print(auc_pr)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.plot(fpr, tpr, linewidth=8, color='lightskyblue', markersize=12)
plt.fill_between(fpr, tpr, color='lightskyblue', alpha=0.1)
axs.set_xlabel('1-Sensitivity', fontsize=30)
axs.set_ylabel('Specificity', fontsize=30)
plt.savefig('/Users/ye/Documents/py/AUROC_validation.png')
plt.show()
#%%
# 创建 DataFrame 存储数据
roc_data = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr, 'Thresholds': thresholds})
roc_data.to_csv('/Users/ye/Documents/py/roc_data_5.csv', index=False)
plt.savefig('/Users/ye/Documents/py/AUROC_validation.png')
plt.show()

#%%
# 4. confusion_matrix：自己拉表格画
fig, axs = plt.subplots(1, 1, figsize=(10, 10))
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
axs.set_xlim(0, axs.get_xlim()[1])
axs.set_ylim(0, axs.get_ylim()[1])
predictions = (testdata_mean['pos_sum'] - testdata_mean['neg_sum'] > threshold).astype(int)
confusion = confusion_matrix(testdata_mean['target'], predictions)
annot_kws = {'size': 30}
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=axs, cbar= False, annot_kws=annot_kws)
axs.set_xlabel('Predicted label')
axs.set_ylabel('True label')
# axs.set_title('Confusion Matrix')
axs.legend()
plt.show()

# %%
#distribution
mean = train_data.groupby("Status").mean()
# mean only keep which in feature
#mean = mean[feature]
times = mean.iloc[0] / mean.iloc[1]
times = pd.concat([times, mean.T], axis=1)
times.columns = ['times', '0', '1']
# if 0 and 1 are both 0, then remove the row
times = times[times['times'] > 0]
# and not infinity
times = times[times['times'] < 100000]
times = times[times['0'] > 1]
times = times[times['1'] > 1]
# sort by times
times = times.sort_values(by=['times'], ascending=False)
times = pd.concat([times.head(30), times.tail(30)])
#%%
#data = train_data[feature_target]
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 3      # 边框粗细
plt.rcParams['axes.labelweight'] = 'bold'  # 轴标签字体粗细
plt.rcParams['xtick.major.width'] = 3   # x轴刻度粗细
plt.rcParams['ytick.major.width'] = 3   # y轴刻度粗细
fig, axs = plt.subplots(1, 1, figsize=(20, 10))
id=times.index
df_1 = pd.DataFrame({'id':id,'Average Expression':times['1'],'label':'1'})
df_0 = pd.DataFrame({'id':id,'Average Expression':times['0'],'label':'0'})
df = pd.concat([df_1,df_0])
colors = {'1': 'coral', '0': 'peachpuff'}
ax=sns.barplot(data=df, x='id', y='Average Expression', hue='label', capsize=0.1, width=0.5, palette=colors, ci=None)
ax.set_xlabel('')#xlabel不显示
ax.set_ylabel('Average Expression', fontsize=30)#调整ylabel字体大小
ax.get_legend().set_visible(False)
plt.xticks(rotation=90)
#ax.tick_params(axis='x', labelsize=25)#x axis label大小
ax.tick_params(axis='y', labelsize=30)#y axis label大小
plt.savefig('/Users/ye/Documents/py/distribution_validation.png')
plt.show()
# %%
#treemap
# Sample data
sizes = [16.0, 4.0, 10.0, 1.0, 25.0, 2.0, 15.0]
labels = ['', '', '', '', '', '', '']
colors = ["coral", "peachpuff", "peachpuff", "lightcyan","paleturquoise","lightcyan","lightcyan"]
plt.figure(figsize=(5,3))
squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.5, pad=True)
# Customize the plot
plt.axis('off')  # Turn off axis labels
#plt.title("Treemap with Padding")
plt.savefig('/Users/ye/Documents/py/treemap_validation.png')
plt.show()







#%%
#%%
#train
##((testdata_mean['pos_sum'] - testdata_mean['neg_sum'] - threshold > 0) -> 没有患病的threshold的定义
# 仅选择特征和最终结果. 输出末2
#thereshold 是traindata_mean['sum']
ID = feature.copy()
ID = np.array(ID).reshape(-1)
# 加('Status')
ID = np.append(ID, ['Status'])
ID.astype('str')
testdata = trainging_data[ID] #train_data测试集 -> testdata 
# 把testdata的columns的名字改成feature
testdata.columns = feature + ["target"]
testdata.tail(2)
weight = np.array(weight).reshape(-1)
weight = np.append(weight, [1])# weight 的最后一位是target，不需要计算，设置为1
# 计算每一种基因表达量的平均值,给定的权重weight，算每一种基因testdata_mean = testdata  * weight
testdata_mean = testdata  * weight
weight = weight.reshape(-1)
weight = weight[:-1]
# 计算每一行的带权重（每行每一个数据已经经过权重处理）testdata_mean['sum'] 和，要去掉最后一列
testdata_mean['sum'] = (testdata_mean.sum(axis=1) - testdata_mean['target']) 
zipped = zip(weight, feature)
# select weight > 0，positive mirna，为feature_pos
feature_pos = [x for x in zipped if x[0] > 0]
feature_pos = [x[1] for x in feature_pos]
testdata_mean['pos_sum'] = testdata_mean[feature_pos].sum(axis=1)
# neg
testdata_mean['neg_sum'] = testdata_mean['pos_sum'] - testdata_mean['sum']
# error
testdata_mean['error_indices'] = ((testdata_mean['pos_sum'] - testdata_mean['neg_sum'] - threshold > 0) & (testdata_mean['target'] == 0)) | ((testdata_mean['pos_sum'] - testdata_mean['neg_sum'] - threshold < 0) & (testdata_mean['target'] == 1))

#plot #plot #plot #plot #plot #plot #plot #plot #plot #plot 
#plot #plot #plot #plot #plot #plot #plot #plot #plot #plot 

# %%
#plot 
#1.threhold
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 3      # 边框粗细
plt.rcParams['axes.labelweight'] = 'bold'  # 轴标签字体粗细
plt.rcParams['xtick.major.width'] = 3   # x轴刻度粗细
plt.rcParams['ytick.major.width'] = 3   # y轴刻度粗细
fig, axs = plt.subplots(1, 1, figsize=(10, 10))
positive = len(testdata_mean['pos_sum'])
negative =  len(testdata_mean['neg_sum'])
axs.scatter(testdata_mean['pos_sum'], testdata_mean['neg_sum'], c=['orange' if target == 0 else 'coral' for target in testdata_mean['target']], s=130, alpha=1)
#axs.scatter(testdata_mean['pos_sum'], testdata_mean['neg_sum'], c=testdata_mean['target'], cmap='Spectral', s=130, alpha=0.6)#c=testdata_mean['target'], cmap='coolwarm', alpha=0.8
axs.set_xlabel('Number of positive expression genes', fontsize=25)
axs.set_ylabel('Number of negative expression genes', fontsize=25)
x_vals = np.array(axs.get_xlim())
y_vals = x_vals - threshold
axs.plot(x_vals, y_vals, linestyle='--', color='lightgray')
axs.set_xlim(0, axs.get_xlim()[1])
axs.set_ylim(0, axs.get_ylim()[1])
axs.tick_params(axis='x', labelsize=30)  # 设置 x 轴刻度的字体大小和旋转角度
axs.tick_params(axis='y', labelsize=30)
axs.fill_between(x_vals+200000, y_vals+200000, color='peachpuff', alpha=0.3)
axs.fill_betweenx(y_vals+200000,x_vals+200000, 0, color='gold', alpha=0.1)
num_errors = len(testdata_mean['pos_sum'][testdata_mean['error_indices']])
axs.scatter(testdata_mean['pos_sum'][testdata_mean['error_indices']], testdata_mean['neg_sum'][testdata_mean['error_indices']], color='black', marker='x', s=100)
# legend = axs.legend(frameon=False)
# for text in legend.get_texts():
#     text.set_fontsize(100)
plt.savefig('/Users/ye/Documents/py/threhold_train.png')
plt.show()
# %%
#2. AUROC
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 3      # 边框粗细
plt.rcParams['axes.labelweight'] = 'bold'  # 轴标签字体粗细
plt.rcParams['xtick.major.width'] = 3   # x轴刻度粗细
plt.rcParams['ytick.major.width'] = 3   # y轴刻度粗细
fig, axs = plt.subplots(1, 1, figsize=(10, 10))

axs.tick_params(axis='x', labelsize=25)  # 设置 x 轴刻度的字体大小和旋转角度
axs.tick_params(axis='y', labelsize=25)
fpr, tpr, thresholds = roc_curve(testdata_mean['target'], testdata_mean['pos_sum'] - testdata_mean['neg_sum'])
# roc_auc = metrics.auc(fpr, tpr)
plt.plot([0, 1], [0, 1], color='darkorange', linestyle='--')
plt.plot(fpr, tpr, linewidth=8, color='orange', markersize=12)
plt.fill_between(fpr, tpr, color='coral', alpha=0.1)
axs.set_xlabel('1-Sensitivity', fontsize=30)
axs.set_ylabel('Specificity', fontsize=30)
auc_pr = average_precision_score(testdata_mean['target'], testdata_mean['pos_sum'] - testdata_mean['neg_sum'])
print(auc_pr)
plt.savefig('/Users/ye/Documents/py/AUROC_train.png')
plt.show()
#%%
# 4. confusion_matrix：自己拉表格画
fig, axs = plt.subplots(1, 1, figsize=(10, 10))
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
axs.set_xlim(0, axs.get_xlim()[1])
axs.set_ylim(0, axs.get_ylim()[1])
predictions = (testdata_mean['pos_sum'] - testdata_mean['neg_sum'] > threshold).astype(int)
confusion = confusion_matrix(testdata_mean['target'], predictions)
annot_kws = {'size': 30}
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=axs, cbar= False, annot_kws=annot_kws)
axs.set_xlabel('Predicted label')
axs.set_ylabel('True label')
# axs.set_title('Confusion Matrix')
axs.legend()
plt.show()

