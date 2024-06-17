#%%
feature = ['hsa-mir-182', 'hsa-mir-21', 'hsa-mir-148a', 'hsa-let-7b', 'hsa-let-7a-3', 'hsa-mir-143', 'hsa-mir-30a']
weight1 = [16.0, 4.0, 10.0, -1.0, -25.0, -2.0, -15.0]#0：99.91
weight2 = [1.0, 4.0, 1.0, 1.0, 2.0, 2.0, 1.0]#2:0.9832
weight3 = [-1.0, -4.0, -1.0, 1.0, 2.0, 2.0, 1.0]#3:0.771651699939615 
weight4=[4.0, -2.0, -1.0, 1.0, 6.0, 2.0, 4.0]#4:0.77055667621045724:
weight5=[3.0, -2.0, -1.0, 1.0, 4.0, 2.0, 3.0]#5:0.770421620066699 
weight6 = [16.0, 4.0, 8.0, -1.0, -20.0, -2.0, -10.0] #1:0.9991559753460473
#%%
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 3      # 边框粗细
plt.rcParams['axes.labelweight'] = 'bold'  # 轴标签字体粗细
plt.rcParams['xtick.major.width'] = 3   # x轴刻度粗细
plt.rcParams['ytick.major.width'] = 3   # y轴刻度粗细
fig, axs = plt.subplots(1, 1, figsize=(20, 10))
axs.tick_params(axis='x', labelsize=25)  # 设置 x 轴刻度的字体大小和旋转角度
axs.tick_params(axis='y', labelsize=25)

feature = ['hsa-mir-182', 'hsa-mir-21', 'hsa-mir-148a', 'hsa-let-7b', 'hsa-let-7a-3', 'hsa-mir-143', 'hsa-mir-30a']
weight1 = [16.0, 4.0, 10.0, -1.0, -25.0, -2.0, -15.0]
weight2 = [1.0, 4.0, 1.0, 1.0, 2.0, 2.0, 1.0]
weight3 = [-1.0, -4.0, -1.0, 1.0, 2.0, 2.0, 1.0]
weight4 = [4.0, -2.0, -1.0, 1.0, 6.0, 2.0, 4.0]
weight5 = [3.0, -2.0, -1.0, 1.0, 4.0, 2.0, 3.0]
weight6 = [16.0, 4.0, 8.0, -1.0, -20.0, -2.0, -10.0]


plt.fill_between(feature, 8, 17, color='peachpuff', alpha=0.2)
plt.fill_between(feature, 8, -8, color='paleturquoise', alpha=0.2)
plt.fill_between(feature, -8, -26, color='peachpuff', alpha=0.2)
# 绘制散点连线图
plt.plot(feature, weight1, marker='o',color='orangered',linewidth=10,alpha=0.5)
plt.plot(feature, weight2, marker='o',color='peachpuff',linewidth=8)
plt.plot(feature, weight3, marker='o',color='skyblue',linewidth=3)
plt.plot(feature, weight4, marker='o',color='limegreen',linewidth=6)
plt.plot(feature, weight5, marker='o',color='violet',linewidth=4)
plt.plot(feature, weight6, marker='o',color='skyblue',linewidth=7)

# 设置标题和标签
axs.set_xlabel('miRNA', fontsize=30)
axs.set_ylabel('weights', fontsize=30)

# 显示图例
plt.legend()

# # 自动调整横坐标标签
# plt.xticks(rotation=45, ha='right')

# 显示图形
plt.tight_layout()
plt.show()

# %%
