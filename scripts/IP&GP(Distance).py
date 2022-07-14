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
    [0.891892, 1, 0.846154, 0.875, 0.935484, 0.916667, 0.8, 0.612903, 0.388889, 0.483871, 0.916667, 1, 0.944444, 0.37037, 0.884615, 0.9375],
    [0.837838, 0.794118, 0.538462, 0.6875, 0.709677, 0.666667, 0.771429, 0.483871, 0.305556, 0.483871, 0.861111, 0.764706, 0.555556, 0.203704, 0.653846, 0.5625, ]]
# data2 是F1 score中三个箱型图的参数
data2 = [
    [0.972973, 1, 0.826923, 0.84375, 0.935484, 0.916667, 0.857143, 0.612903, 0.388889, 0.774194, 0.944444, 1, 0.944444, 0.37037, 0.807692, 0.84375, ],
    [0.756757, 0.794118, 0.538462, 0.625, 0.709677, 0.625, 0.771429, 0.483871, 0.305556, 0.548387, 0.722222, 0.764706, 0.555556, 0.203704, 0.5, 0.25, ]]
# data3 是IoU中三个箱型图的参数
data3 = [
    [0.918919, 1, 0.807692, 0.96875, 0.935484, 0.875, 0.857143, 0.774194, 0.611111, 0.709677, 0.972222, 0.970588, 0.944444, 0.537037, 0.961538, 0.5625, ],
    [0.648649, 0.794118, 0.519231, 0.6875, 0.709677, 0.375, 0.742857, 0.548387, 0.5, 0.451613, 0.694444, 0.735294, 0.5, 0.314815, 0.5, 0.3125, ]]
data4 = [
    [0.918919, 0.970588, 0.826923, 0.9375, 0.870968, 0.833333, 0.942857, 0.741935, 0.611111, 0.677419, 0.861111, 0.970588, 0.944444, 0.537037, 0.923077, 0.46875, ],
    [0.648649, 0.764706, 0.557692, 0.65625, 0.612903, 0.291667, 0.714286, 0.483871, 0.5, 0.387097, 0.583333, 0.705882, 0.472222, 0.314815, 0.423077, 0.3125, ]]
data5 = [
    [0.783784, 0.823529, 0.884615, 0.8125, 0.677419, 0.791667, 0.885714, 0.741935, 0.583333, 0.612903, 0.777778, 0.823529, 0.833333, 0.611111, 0.807692, 0.4375, ],
    [0.486486, 0.617647, 0.653846, 0.5625, 0.451613, 0.291667, 0.657143, 0.451613, 0.416667, 0.387097, 0.5, 0.588235, 0.361111, 0.37037, 0.307692, 0.3125, ]]
data6 = [
    [0.72973, 0.705882, 0.846154, 0.625, 0.612903, 0.75, 0.885714, 0.645161, 0.444444, 0.645161, 0.611111, 0.705882, 0.694444, 0.518519, 0.615385, 0.5, ],
    [0.378378, 0.558824, 0.615385, 0.40625, 0.354839, 0.291667, 0.6, 0.387097, 0.333333, 0.322581, 0.361111, 0.411765, 0.222222, 0.296296, 0.230769, 0.40625, ]]
# 箱型图名称
labels = ["IP", "GP"]
# 箱型图的颜色 RGB （均为0~1的数据）
colors = [(255/255., 190/255., 122/255.), (130/255., 176/255., 210/255.)]

interval = 0.25
width = 0.2
bplot1 = draw_box(data1, colors, position=1, interval=interval, width=width)
bplot2 = draw_box(data2, colors, position=2, interval=interval, width=width)
bplot3 = draw_box(data3, colors, position=3, interval=interval, width=width)
bplot4 = draw_box(data4, colors, position=4, interval=interval, width=width)
bplot5 = draw_box(data5, colors, position=5, interval=interval, width=width)
bplot6 = draw_box(data6, colors, position=6, interval=interval, width=width)

x_position = [1, 2, 3, 4, 5, 6]
x_position_fmt = ['15', '20', '25', '30', '35', '40']
plt.xticks([i + interval / 2 for i in x_position], x_position_fmt)

plt.xlabel('Distance-paired-threshold')
plt.ylabel('Probability')
plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
plt.legend(bplot1['boxes'], labels, loc='lower left')  # 绘制表示框，右下角绘制
plt.savefig(fname="pic.png", figsize=[10, 10])
plt.show()

