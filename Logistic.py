import matplotlib.pyplot as plt
import numpy as np
import math


# 朴素贝叶斯：1.X独立    2.方差相同
loc0 = 0  # Y=0时X的均值
loc1 = 50  # Y=1时X的均值
scale0 = 20  # 方差
scale1 = 20
m = 2  # X属性个数
num0 = 100  # 训练集中Y=0类大小
num1 = 100
test_num0 = 200  # 测试集中Y=0类大小
test_num1 = 200
step = 0.01  # 梯度下降步长
max_iter = 20000  # 梯度下降最大迭代次数
stop_value = 0.00000001  # 梯度下降退出条件
L = math.e**(-8)    #正则项系数


# 当X维度为2时，用于画出分类图像
def draw(X0,X1,Y, W,choose):  # 画函数图像
    X = np.vstack((X0, X1))
    plt.title("Logistic function for test"+choose+"\nAccuracy="+str("%.2f" % accuracy(X, Y, W))+"    Precision=" +
              str("%.2f" % precision(X, Y, W))+"    Recall="+str("%.2f" % recall(X, Y, W)))
    plt.scatter(X0.transpose().tolist()[
                1], X0.transpose().tolist()[2], label="Y = 0")
    plt.scatter(X1.transpose().tolist()[
                1], X1.transpose().tolist()[2], label="Y = 1")
    X = np.linspace(loc0-3*scale0, loc1+3*scale1, 2)
    W = W.tolist()[0]
    plt.plot(X, -(W[0]+W[1]*X)/W[2])
    plt.legend()


# 牛顿法
def newton(X, Y):
    W = np.mat(np.linspace(0, 0, X.shape[1]))  # 1*(m+1)
    iter = 0
    while((iter < 100)):
        try:
            H_I = Hessian(X,W).I
        except:
            return W
        change = H_I.dot(gra(X,Y,W).T).T
        W = W + change
        if (change.dot(change.T) < stop_value):
            break
        iter += 1
    return W


# 有正则项的牛顿法
def newton_reg(X, Y):
    W = np.mat(np.linspace(0, 0, X.shape[1]))  # 1*(m+1)
    iter = 0
    while((iter < 100)):
        try:
            H_I = Hessian(X,W).I
        except:
            return W
        change = H_I.dot((gra(X,Y,W)-L*W).T).T
        W = W + change
        if (change.dot(change.T) < stop_value):
            break
        iter += 1
    return W


def Newton(X,Y,X0_Test,X1_Test,Y_Test):
    W = newton(X,Y)
    W_reg = newton(X,Y)
    plt.subplot(121)
    draw(X0_Test,X1_Test,Y_Test,W,"by Newton method without regular")
    plt.subplot(122)
    draw(X0_Test,X1_Test,Y_Test,W_reg,"by Neton method with regular")
    plt.show()


#求Hessian矩阵  X:n*(m+1)   W:1*(m+1)
def Hessian(X,W):
    temp = possibility(X,W) #1*n
    sum = np.multiply(temp,(1-temp))    #1*n
    A = np.multiply(sum,np.eye(X.shape[0]))
    return X.T.dot(A).dot(X)#(m+1)*(m+1)


# 梯度下降法
def gradient(X, Y):
    W = np.mat(np.linspace(0, 0, X.shape[1]))  # 1*(m+1)
    iter = 0
    while((iter < max_iter)):
        g = gra(X, Y, W)
        W = W + step*g
        if (g.dot(g.T) < stop_value):
            break
        iter += 1
    return W


# 带正则的梯度下降法
def gradient_reg(X, Y):
    W = np.mat(np.linspace(0, 0, X.shape[1]))  # 1*(m+1)
    iter = 0
    while((iter < max_iter)):
        g = gra(X, Y, W)
        W = W + step*g - L*W
        if (g.dot(g.T) < stop_value):
            break
        iter += 1
    return W


def Gradient(X,Y,X0_Test,X1_Test,Y_Test):
    W = gradient(X,Y)
    W_reg = gradient_reg(X,Y)
    plt.subplot(121)
    draw(X0_Test,X1_Test,Y_Test,W,"by gradient without regular")
    plt.subplot(122)
    draw(X0_Test,X1_Test,Y_Test,W_reg,"by gradient with regular")
    plt.show()


# 求梯度大小
def gra(X, Y, W):
    P = Y-possibility(X, W)  # 1*num
    update = X.T.dot(P.T)  # (m+1)*1
    return update.T


# X:num*(m+1)  W:1*(m+1) return:1*num
def possibility(X, W):
    Z = X.dot(W.T)  # num*1
    sum = Z.T
    temp = np.exp(sum)
    p = 1-1 / (1+temp)
    return p


# 计算准确度
def accuracy(X, Y, W):
    sum = 0
    Y = Y.tolist()[0]
    for i in range(X.shape[0]):
        t = W.dot(X[i].T)
        if ((t > 0) & (Y[i] == 1)) | ((t < 0) & (Y[i] == 0)):
            sum += 1
    return sum/len(Y)


# 计算precision
def precision(X, Y, W):
    all_one = 0
    TP = 0
    Y = Y.tolist()[0]
    for i in range(X.shape[0]):
        t = W.dot(X[i].T)
        if(t > 0):
            all_one += 1
            if (Y[i] == 1):
                TP += 1
    return TP/all_one


def recall(X, Y, W):
    real_one = 0
    TP = 0
    Y = Y.tolist()[0]
    for i in range(X.shape[0]):
        t = W.dot(X[i].T)
        if(Y[i] == 1):
            real_one += 1
            if t > 0:
                TP += 1
    return TP/real_one


# 生成X数据集
def birthX(lable, loc, scale, num):
    X = np.linspace(1, 1, num)
    for i in range(m):
        X = np.vstack((X, np.random.normal(loc, scale, num)))
    return X.transpose()


def logistic():
    X0 = np.mat(birthX(0, loc0, scale0, num0))
    X1 = np.mat(birthX(1, loc1, scale1, num1))
    X0_Test = np.mat(birthX(0, loc0, scale0, test_num0))
    X1_Test = np.mat(birthX(1, loc1, scale1, test_num1))
    Y_Test = np.mat(np.hstack((np.linspace(0, 0, test_num0), np.linspace(1, 1, test_num1))))
    X = np.mat(np.vstack((X0, X1)))
    Y = np.mat(np.hstack((np.linspace(0, 0, num0), np.linspace(1, 1, num1))))
    #W = gradient_reg(X, Y, m)
    Gradient(X,Y,X0_Test,X1_Test,Y_Test)
    Newton(X,Y,X0_Test,X1_Test,Y_Test)


def trainFromuci():
    (X,Y) = read_train()
    W = gradient(X, Y)
    W_reg = gradient_reg(X,Y)
    W_Newton = newton(X,Y)
    W_Newton_Reg = newton_reg(X,Y)
    (X_test,Y_test) = read_test(W)


def read(filename):
    try:
        f = open('train.txt')
        return f.readlines()
    finally:
        f.close()

#读取训练
def read_train():
    Y = []
    content = read('train.txt')
    N = len(content)
    M = len(content[0].split(','))-1
    X = np.linspace(0, 0, M+1)
    for line in content:
        value = line.rstrip('\n').split(',')
        Y.append(int(value[len(value)-1]))
        del value[M]
        x = [float(i) for i in value]
        x.insert(0, 1)
        X = np.vstack((X, x))
    X = np.delete(X, 0, axis=0)
    X = np.mat(X)
    Y = np.mat(Y)
    W = gradient(X, Y)
    W_reg = gradient_reg(X,Y)
    W_Newton = newton(X,Y)
    W_Newton_Reg = newton_reg(X,Y)
    return X,Y


#读取测试文件并进行通过学习的W进行计算并得到accuracy,precision,recall
def read_test(W):
    Y = []
    content = read('test.txt')
    M = len(content[0].split(','))-1
    X = np.linspace(0, 0, M+1)
    for line in content:
        value = line.rstrip('\n').split(',')
        Y.append(int(value[len(value)-1]))
        del value[M]
        x = [float(i) for i in value]
        x.insert(0, 1)
        X = np.vstack((X, x))
    X = np.delete(X, 0, axis=0)
    X = np.mat(X)
    Y = np.mat(Y)
    return X,Y
    print("accuracy:"+str("%.2f" % accuracy(X, Y, W)))
    print("precision:"+str("%.2f" % precision(X, Y, W)))
    print("recall:"+str("%.2f" % recall(X, Y, W)))


logistic()
trainFromuci()
