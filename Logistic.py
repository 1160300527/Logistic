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
test_num0 = 100  # 测试集中Y=0类大小
test_num1 = 100
step = 0.01  # 梯度下降步长
max_iter = 20000  # 梯度下降最大迭代次数
stop_value = 0.000001  # 梯度下降退出条件
L = -math.e**(-10)  # 正则项系数


# 当X维度为2时，用于画出分类图像
def draw(X0, X1, Y, W, choose):  # 画函数图像
    X = np.vstack((X0, X1))
    R = recall(X,Y,W)
    P = precision(X,Y,W)
    F1 = F1score(R,P)
    plt.title("Logistic function for test"+choose+"\nAccuracy="+str("%.2f" % accuracy(X, Y, W))+"    Precision=" +
              str("%.2f" % P)+"    Recall="+str("%.2f" % R)+"    F1 score="+str("%.2f"%F1))
    plt.scatter(X0.transpose().tolist()[
                1], X0.transpose().tolist()[2], label="Y = 0")
    plt.scatter(X1.transpose().tolist()[
                1], X1.transpose().tolist()[2], label="Y = 1")
    max = loc1
    min = loc0
    min_scale = scale0
    max_scale = scale1
    if loc0>loc1:
        max = loc0
        min = loc1
        max_scale = scale0
        min_scale = scale1
    X = np.linspace(min-3*min_scale, max+3*max_scale, 2)
    W = W.tolist()[0]
    plt.plot(X, -(W[0]+W[1]*X)/W[2])
    plt.legend()


# 牛顿法
def newton(X, Y):
    W = np.mat(np.linspace(0, 0, X.shape[1]))  # 1*(m+1)
    iter = 0
    while((iter < 100)):
        try:
            H_I = Hessian(X, W).I
        except:
            return W
        change = H_I.dot(gra(X, Y, W).T).T
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
            H_I = Hessian(X, W).I
        except:
            return W
        change = H_I.dot((gra(X, Y, W)-L*W).T).T
        W = W + change
        if (change.dot(change.T) < stop_value):
            break
        iter += 1
    return W


# 牛顿法计算W并画出图像
def Newton(X, Y, X0_Test, X1_Test, Y_Test):
    W = newton(X, Y)
    W_reg = newton(X, Y)
    plt.subplot(121)
    draw(X0_Test, X1_Test, Y_Test, W, "by Newton method without regular")
    plt.subplot(122)
    draw(X0_Test, X1_Test, Y_Test, W_reg, "by Neton method with regular")
    plt.show()


# 求Hessian矩阵  X:n*(m+1)   W:1*(m+1)
def Hessian(X, W):
    temp = possibility(X, W)  # 1*n
    sum = np.multiply(temp, (1-temp))  # 1*n
    A = np.multiply(sum, np.eye(X.shape[0]))
    return X.T.dot(A).dot(X)  # (m+1)*(m+1)


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


# 梯度上升方法计算W值并画出图像
def Gradient(X, Y, X0_Test, X1_Test, Y_Test):
    W = gradient(X, Y)
    W_reg = gradient_reg(X, Y)
    plt.subplot(121)
    draw(X0_Test, X1_Test, Y_Test, W, "by gradient without regular")
    plt.subplot(122)
    draw(X0_Test, X1_Test, Y_Test, W_reg, "by gradient with regular")
    plt.show()


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


# 计算recall值
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


#计算F1score
def F1score(R,P):
    return 2*P*R/(P+R)


# 生成X数据集
def birthX(lable, loc, scale, num):
    X = np.linspace(1, 1, num)
    for i in range(m):
        X = np.vstack((X, np.random.normal(loc, scale, num)))
    return X.transpose()


# 随机生成数据并进行测试
def logistic():
    X0 = np.mat(birthX(0, loc0, scale0, num0))
    X1 = np.mat(birthX(1, loc1, scale1, num1))
    X0_Test = np.mat(birthX(0, loc0, scale0, test_num0))
    X1_Test = np.mat(birthX(1, loc1, scale1, test_num1))
    Y_Test = np.mat(
        np.hstack((np.linspace(0, 0, test_num0), np.linspace(1, 1, test_num1))))
    X = np.mat(np.vstack((X0, X1)))
    Y = np.mat(np.hstack((np.linspace(0, 0, num0), np.linspace(1, 1, num1))))
    Gradient(X, Y, X0_Test, X1_Test, Y_Test)
    Newton(X, Y, X0_Test, X1_Test, Y_Test)


# 读取数据集文件并完成训练及测试过程
def trainFromuci():
    # 从训练集文件获取训练样本
    (X, Y) = read_train()
    # 计算通过不同优化方法得到的W
    W = gradient(X, Y)
    W_reg = gradient_reg(X, Y)
    W_Newton = newton(X, Y)
    W_Newton_Reg = newton_reg(X, Y)
    # 从测试集文件获取测试样本
    (X_test, Y_test) = read_test()
    print("The data from UCI:00267")
    print("By Gradient Ascent Without Regular: ")
    print_data(X_test, Y_test, W)
    print("By Gradient Ascent With Regular:")
    print_data(X_test, Y_test, W_reg)
    print("By Newton Wethod Without Regular:")
    print_data(X_test, Y_test, W_Newton)
    print("By Newton Method With Regular:")
    print_data(X_test, Y_test, W_Newton_Reg)


def read(filename):
    try:
        f = open('train.txt')
        return f.readlines()
    finally:
        f.close()

# 读取训练数据文件，得到训练数据集


def read_train():
    Y = []
    content = read('train.txt')  # 打开训练数据文件
    N = len(content)  # 数据集文件中每个样本属性值以\n分割
    M = len(content[0].split(','))-1  # 数据集文件以,分割不同数据
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
    return X, Y


# 读取测试数据文件，得到测试数据集
def read_test():
    Y = []
    content = read('test.txt')  # 打开测试数据文件
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
    return X, Y


# 打印输出Accuracy,precision,recall
def print_data(X, Y, W):
    R = recall(X,Y,W)
    P = recall(X,Y,W)
    F1 = F1score(R,P)
    print("accuracy:"+str("%.2f" % accuracy(X, Y, W)))  # 计算accuracy并输出
    print("precision:"+str("%.2f" % P))  # 计算precision并输出
    print("recall:"+str("%.2f" % R))  # 计算recall并输出
    print("F1 score:"+str("%2f"%F1))


# 随机生成数据并进行训练及测试
logistic()
# 通过从UCI获取的数据进行训练及测试
trainFromuci()
