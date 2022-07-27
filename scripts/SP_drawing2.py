import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import palettable  #python颜色库
import numpy as np


sheet_name = 'score_b&c'
df1 = pd.read_excel('../results/data.xlsx', sheet_name=sheet_name)
object_list = df1.columns[1:]


matrix = np.zeros(shape=(75,1))
for object in object_list:
    object_seq = np.array(df1[object])
    matrix = np.column_stack((matrix,object_seq))
matrix = np.delete(matrix, 0, axis=1)


df = pd.DataFrame(matrix,
                  index=[str(i) for i in range(0, 75)],#DataFrame的行标签设置为大写字母
                  columns=object_list)#设置DataFrame的列标签
plt.figure(dpi=200, figsize=(8, 10))
sns.heatmap(data=df,
            vmin=-1,vmax=1,
            cbar=True,
            cmap='RdBu',
            # cmap=sns.diverging_palette(10, 220, sep=10, n=11),#区分度显著色盘：sns.diverging_palette()使用
            # annot=True, fmt=".2f", annot_kws={'size':5,'weight':'normal'},
           )
plt.title("SP(score by BoxSize/CenterOffset)")
plt.xlabel('Object class')
plt.ylabel('Time series(frame)')
plt.show()