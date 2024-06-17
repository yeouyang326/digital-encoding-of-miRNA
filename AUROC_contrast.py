#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, average_precision_score

# 从两个不同的 CSV 文件加载数据
roc_data_0 = pd.read_csv('/Users/ye/Documents/py/roc_data_0.csv')
roc_data_1 = pd.read_csv('/Users/ye/Documents/py/roc_data_1.csv')
roc_data_2 = pd.read_csv('/Users/ye/Documents/py/roc_data_2.csv')
roc_data_3 = pd.read_csv('/Users/ye/Documents/py/roc_data_3.csv')
roc_data_4 = pd.read_csv('/Users/ye/Documents/py/roc_data_4.csv')
roc_data_5 = pd.read_csv('/Users/ye/Documents/py/roc_data_5.csv')

fpr_0, tpr_0, thresholds_0 = roc_data_0['False Positive Rate'], roc_data_0['True Positive Rate'], roc_data_0['Thresholds']
fpr_1, tpr_1, thresholds_1 = roc_data_1['False Positive Rate'], roc_data_1['True Positive Rate'], roc_data_1['Thresholds']
fpr_2, tpr_2, thresholds_2 = roc_data_2['False Positive Rate'], roc_data_2['True Positive Rate'], roc_data_2['Thresholds']
fpr_3, tpr_3, thresholds_3 = roc_data_3['False Positive Rate'], roc_data_3['True Positive Rate'], roc_data_3['Thresholds']
fpr_4, tpr_4, thresholds_4 = roc_data_4['False Positive Rate'], roc_data_4['True Positive Rate'], roc_data_4['Thresholds']
fpr_5, tpr_5, thresholds_5 = roc_data_5['False Positive Rate'], roc_data_5['True Positive Rate'], roc_data_5['Thresholds']
#%%
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 3      # 边框粗细
plt.rcParams['axes.labelweight'] = 'bold'  # 轴标签字体粗细
plt.rcParams['xtick.major.width'] = 3   # x轴刻度粗细
plt.rcParams['ytick.major.width'] = 3   # y轴刻度粗细
fig, axs = plt.subplots(1, 1, figsize=(10, 10))
axs.tick_params(axis='x', labelsize=25)  # 设置 x 轴刻度的字体大小和旋转角度
axs.tick_params(axis='y', labelsize=25)

plt.fill_between(fpr_2, tpr_2, color='paleturquoise', alpha=0.1)

plt.plot(fpr_1, tpr_1, color='orangered',linewidth=6)
plt.plot(fpr_0, tpr_0, color='yellow',linewidth=3)

plt.plot(fpr_2, tpr_2,color='skyblue', linewidth=3)
plt.plot(fpr_3, tpr_3, color='limegreen',linewidth=8)
plt.plot(fpr_4, tpr_4, color='violet',linewidth=7)
plt.plot(fpr_5, tpr_5, color='blue',linewidth=2)
# 绘制对角线
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=2)


axs.set_xlabel('1-Sensitivity', fontsize=30)
axs.set_ylabel('Specificity', fontsize=30)


plt.legend()
plt.show()

# %%
