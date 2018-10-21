import matplotlib.pyplot as plt
import numpy as np
# 朴素贝叶斯：1.X独立    2.方差相同
loc0 = 12
loc1 = 17
scale = 5
m = 2
num = 10


# 当X维度为2时，用于画出分类图像
def draw(X0, X1, Y, W):  # 画函数图像
    plt.title("Logistic function")
    plt.scatter(X0.transpose()[1], X0.transpose()[2], label="Y = 0")
    plt.scatter(X1.transpose()[1], X1.transpose()[2], label="Y = 1")
    plt.plot(X.transpose()[1], (W[0]-W[1]*X.transpose()[1])/W[2])
    plt.legend()
    plt.show()


def logistic(X, Y):
    return [1, 1, 1]


# 生成X数据集
def birthX(lable, loc, scale):
    X = np.linspace(1, 1, num)
    for i in range(m):
        X = np.vstack((X, np.random.normal(loc, scale, num)))
    return X.transpose()


X0 = birthX(0, loc0, scale)
X1 = birthX(1, loc1, scale)
X = np.vstack((X0, X1))
print(X0.transpose()[2])
Y = np.hstack((np.linspace(0, 0, num), np.linspace(1, 1, num)))
W = logistic(X, Y)
draw(X0, X1, Y, W)
print(Y)
