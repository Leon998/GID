import matplotlib.pyplot as plt


def draw_box(data, colors, position, interval=0.3, width=0.2):
    # 绘制箱型图
    # patch_artist=True-->箱型可以更换颜色，positions-->将同一组的三个箱间隔设置为interval，widths-->每个箱宽度
    bplot = plt.boxplot(data, patch_artist=True, labels=labels, positions=(position, position + interval), widths=width,
                        medianprops=dict(color='r', linewidth=1),
                        showmeans=True, meanline=True, meanprops=dict(color='black', linestyle='--', linewidth=1)
                        )
    # 将三个箱分别上色
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    return bplot

data1 = [
    [0.783784, 0.911765, 0.961538, 0.9375, 0.903226, 0.875, 0.857143, 0.806452, 0.666667, 0.677419, 0.833333, 1, 0.916667, 0.555556, 0.846154, 0.96875, ],
    [0.756757, 0.852941, 0.865385, 0.78125, 0.741935, 0.625, 0.828571, 0.709677, 0.583333, 0.645161, 0.833333, 0.970588, 0.722222, 0.555556, 0.730769, 0.78125, ]]
# data2 是F1 score中三个箱型图的参数
data2 = [
    [0.6, 0.4, 0.5, 0.4, 0.634, 0.3245, 0.6, 0.666667, 0.5, 0.75, 0.654, 0.578, 0.7654, 0.666667, 0.74563, 0.436, ],
    [0.4, 0.325, 0.675, 0.234, 0.765, 0.26, 0.5, 0.3, 0.5, 0.3765, 0.654, 0.3, 0.5, 0.666667, 0.643, 0.24, ]]
# data3 是IoU中三个箱型图的参数
data3 = [
    [0.23, 0.3, 0.432, 0.45, 0.64, 0.75, 0.256, 0.27, 0.5, 0.543, 0.333333, 0.67, 0.5, 0.7, 0.2, 0.14, ],
    [0, 0.4, 0.5, 0.3, 0.2, 0.5, 0.333333, 0, 0.5, 0.5, 0.333333, 0.4, 0.6, 0, 0, 0, ]]
# 箱型图名称
labels = ["IP", "GP"]
# 箱型图的颜色 RGB （均为0~1的数据）
colors = [(255/255., 190/255., 122/255.), (130/255., 176/255., 210/255.)]

interval = 0.25
width = 0.2
bplot1 = draw_box(data1, colors, position=1, interval=interval, width=width)
bplot2 = draw_box(data2, colors, position=2, interval=interval, width=width)
bplot3 = draw_box(data3, colors, position=3, interval=interval, width=width)


x_position = [1, 2, 3]
x_position_fmt = ['location1', 'location2', 'location3']
plt.xticks([i + interval / 2 for i in x_position], x_position_fmt)

plt.xlabel('Camera mounting location')
plt.ylabel('Probability')
plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
plt.legend(bplot1['boxes'], labels, loc='lower left')  # 绘制表示框，右下角绘制
plt.savefig(fname="pic.png", figsize=[10, 10])
plt.show()

