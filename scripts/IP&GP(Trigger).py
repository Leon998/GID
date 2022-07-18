import matplotlib.pyplot as plt

fig, ax = plt.subplots()  # 子图


area1 = [0.702703, 0.852941, 0.865385, 0.75, 0.677419, 0.625, 0.714286, 0.741935, 0.555556, 0.580645, 0.75, 0.911765, 0.666667, 0.555556, 0.730769, 0.78125, ]
area2 = [0.756757, 0.852941, 0.865385, 0.78125, 0.741935, 0.625, 0.828571, 0.709677, 0.583333, 0.645161, 0.833333, 0.970588, 0.722222, 0.555556, 0.730769, 0.78125, ]
area3 = [0.756757, 0.911765, 0.865385, 0.8125, 0.741935, 0.625, 0.8, 0.741935, 0.555556, 0.580645, 0.916667, 0.970588, 0.666667, 0.518519, 0.730769, 0.75, ]
area4 = [0.783784, 0.882353, 0.846154, 0.78125, 0.709677, 0.625, 0.8, 0.677419, 0.583333, 0.612903, 0.888889, 0.882353, 0.666667, 0.407407, 0.769231, 0.75, ]
area5 = [0.675676, 0.794118, 0.730769, 0.71875, 0.645161, 0.625, 0.8, 0.548387, 0.472222, 0.580645, 0.833333, 0.764706, 0.527778, 0.333333, 0.576923, 0.65625, ]
area6 = [0.675676, 0.588235, 0.403846, 0.46875, 0.516129, 0.541667, 0.628571, 0.451613, 0.361111, 0.322581, 0.583333, 0.5, 0.277778, 0.277778, 0.423077, 0.34375, ]
data = [area1, area2, area3, area4, area5, area6]

ax.boxplot(data, showmeans=True)
plt.xlabel('Triggering area')
plt.ylabel('GP(Probability)')
plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
ax.set_xticklabels(["A1", "A2", "A3", "A4", "A5", "A6"])
plt.show()