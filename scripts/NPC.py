import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


df = pd.read_excel('../results/data.xlsx', sheet_name='Voting')
NPC_mean = df.loc[3].values[1:]
NPC_std = df.loc[4].values[1:]
IP_mean = df.loc[5].values[1:]
IP_std = df.loc[6].values[1:]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.errorbar(x=[1, 5, 10, 15, 20, 25, 30, 35, 40], y=NPC_mean, yerr=NPC_std, label='NPC',
             fmt='k-^', lw=2, color='darkorange', elinewidth=1, ms=7, capsize=3)

ax2.errorbar(x=[1, 5, 10, 15, 20, 25, 30, 35, 40], y=IP_mean, yerr=IP_std, label='IP',
             fmt='k-s', lw=2, color='steelblue', elinewidth=1, ms=7, capsize=3)


ax1.set_ylim((3, 12))
ax2.set_ylim((0, 1))
ax1.set_xlabel('step to voting')
ax1.set_ylabel('Number of prediction changes(NPC)')
ax2.set_xlabel('step to voting')
ax2.set_ylabel('Probability(IP)')
ax1.legend(loc='lower left')
ax2.legend(loc='lower right')
plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
plt.show()