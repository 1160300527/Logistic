import matplotlib.pyplot as plt
import numpy as np
import math


# 朴素贝叶斯：1.X独立    2.方差相同
loc0 = 0  # Y=0时X的均值
loc1 = 50  # Y=1时X的均值
scale = 20  # 方差
m = 2  # X属性个数
num = 200  # 数据集大小
step = 0.01  # 梯度下降步长
max_iter = 20000  # 梯度下降最大迭代次数
stop_value = 0.000001  # 梯度下降退出条件
L = math.e**(-8)


# 当X维度为2时，用于画出分类图像
def draw(X,X0, X1, Y, W):  # 画函数图像
    plt.title("Logistic function")
    plt.scatter(X0.transpose().tolist()[1], X0.transpose().tolist()[2], label="Y = 0")
    plt.scatter(X1.transpose().tolist()[1], X1.transpose().tolist()[2], label="Y = 1")
    plt.text(-0.05, -1.2,"loss="+str("%.2f"%loss(X,Y,W))+"accuracy="+str("%.2f"%accuracy(X,Y,W)),fontsize = 8)
    X = np.linspace(-100,100,100)
    W = W.tolist()[0]
    plt.plot(X, -(W[0]+W[1]*X)/W[2])
    plt.legend()
    plt.show()


# 梯度下降法
def gradient(X, Y):
    W = np.mat(np.linspace(0,0,m+1))#1*(m+1)
    iter = 0
    g = gra(X, Y, W)
    while((iter < max_iter)):
        g = gra(X, Y, W)
        W = W - step*g #+ L*W
        if (g.dot(g.T) < stop_value):
            break
        iter += 1
    return W


# 求梯度大小
def gra(X, Y, W):
    P = Y-possibility(X,W)  #1*num
    update = X.T.dot(P.T)#(m+1)*1
    return update.T



# 生成X数据集
def birthX(lable, loc, scale):
    X = np.linspace(1, 1, num)
    for i in range(m):
        X = np.vstack((X, np.random.normal(loc, scale, num)))
    return X.transpose()


#X:num*(m+1)  W:1*(m+1) return:1*num
def possibility(X, W):
    Z = X.dot(W.T)  #num*1
    sum = Z.T
    temp = np.exp(sum)
    p = 1 /(1+temp)
    return p


def loss(X,Y,W):
    return np.sum(np.fabs(Y-possibility(X,W)))/num


def accuracy(X,Y,W):
    sum = 0
    Y = Y.tolist()[0]
    for i in range(X.shape[0]):
        t = W.dot(X[i].T)
        if ((t>0)&(Y[i]==0))|((t<0)&(Y[i]==1)):
            sum+=1
    return sum/(2*num)

X0 = np.mat(birthX(0, loc0, scale))
X1 = np.mat(birthX(1, loc1, scale))
X = np.vstack((X0, X1))
Y = np.mat(np.hstack((np.linspace(0, 0, num), np.linspace(1, 1, num))))
W = gradient(X, Y)
draw(X,X0, X1, Y, W)
print(W)