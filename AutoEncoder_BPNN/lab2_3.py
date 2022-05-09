import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# 隐含层的层数为任意层，激活函数可选sigmoid、tanh
from sklearn.preprocessing import MinMaxScaler

# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# tanh激活函数
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

#激活函数 tanh导数
def d_tanh(x):
    return 1 + np.negative(np.square(tanh(x)))

# 参数初始化作用
def init_parameter(nodes_layer, layer_num):
    W = [0]
    b = [0]
    np.random.seed(5)
    for i in range(1,layer_num):
        wi = np.random.randn(nodes_layer[i],nodes_layer[i-1]) / np.sqrt(nodes_layer[i-1])
        bi = np.random.randn(nodes_layer[i], 1)
        W.append(np.array(wi))
        b.append(np.array(bi))
    parameters = {"W": W, "b": b}
    return parameters

# 前向传播,使用tanh作为激活函数
def forward_propagation(X, layer_num, parameters,method):
    Z = [0]
    A = [0]
    W = parameters['W']
    b = parameters['b']

    A[0] = X
    for i in range(1, layer_num):
        zi = np.dot(W[i], A[i-1]) + b[i]
        if method == 'sig':
            ai = sigmoid(zi)
        elif method == 'tan_h':
            ai = tanh(zi)
        Z.append(np.array(zi))
        A.append(np.array(ai))
    temp = {'Z': Z, 'A': A}
    return temp


def compute_cost(a2, Y):
    # 此处利用最小二乘法来计算cost
    cost = (1/2) * np.sum(np.power((a2-Y), 2))
    return cost


# 计算更新参数
# X，Y都是列向量
def backward_propagation(parameters, layer_num, temp, X, method):

    A = temp['A']
    Z = temp['Z']
    W = parameters['W']
    # 初始化存储矩阵
    error = [0]     # 每一层的error
    dW = [0]        # w的变化值
    db = [0]        # b的变化值
    for i in range(1,layer_num):
        error.append(0)
        dW.append(0)
        db.append(0)
    if method == 'sig':
        # 计算输出层的error和w,b的更新值
        error[layer_num-1] = (A[layer_num-1]-X) * A[layer_num-1]*(1-A[layer_num-1])
        dW[layer_num-1] = error[layer_num-1] * A[layer_num-2].T
        db[layer_num-1] = np.sum(error[layer_num-1], axis=1, keepdims=True)
        # 计算隐藏层的error和w,b的更新值
        for i in range(2,layer_num):
            layer = layer_num - i
            error[layer] = np.dot(W[layer+1].T, error[layer+1]) * A[layer] * (1-A[layer])
            dW[layer] = error[layer] * A[layer-1].T
            db[layer] = np.sum(error[layer], axis=1, keepdims=True)
    elif method == 'tan_h':
        # 计算输出层的error和w,b的更新值
        error[layer_num - 1] = (A[layer_num - 1] - X) * d_tanh(Z[layer_num - 1])
        dW[layer_num - 1] = error[layer_num - 1] * A[layer_num - 2].T
        db[layer_num - 1] = np.sum(error[layer_num - 1], axis=1, keepdims=True)
        # 计算隐藏层的error和w,b的更新值
        for i in range(2, layer_num):
            layer = layer_num - i
            error[layer] = np.dot(W[layer + 1].T, error[layer + 1]) * d_tanh(Z[layer])
            dW[layer] = error[layer] * A[layer - 1].T
            db[layer] = np.sum(error[layer], axis=1, keepdims=True)
    grads = {'dW': dW, 'db': db}
    return grads


# 更新参数
# learning_rate是学习率
def update_parameters(parameters, layer_num, grads, learning_rate):
    W = parameters['W']
    b = parameters['b']

    dW = grads['dW']
    db = grads['db']

    for i in range(1,layer_num):
        W[i] -= dW[i] * learning_rate
        b[i] -= db[i] * learning_rate

    parameters = {"W": W, "b": b}
    return parameters


def BPNN(X, nodes_layer, layer_num, max_iteration, learning_rate, scaler, method):
    # 初始化参数
    parameters = init_parameter(nodes_layer, layer_num)

    # costArray用于记录每次迭代的cost; zero_x表示cost第一次变为0时候的迭代次数; flag作辅助作用
    costArray = np.array([])
    zero_x = 0
    flag = 1

    # 根据最大迭代次数循环更新
    for i in range(0, max_iteration):
        # 向前传播
        temp = forward_propagation(X, layer_num, parameters, method)
        output = temp['A'][layer_num-1]
        # 反向传播
        grads = backward_propagation(parameters, layer_num, temp, X, method)
        # 更新参数
        parameters = update_parameters(parameters, layer_num, grads, learning_rate)
        # 计算并绘制cost
        X_init = scaler.inverse_transform(X)
        X_predict = scaler.inverse_transform(output)
        cost = compute_cost(X_predict, X_init)
        costArray = np.append(costArray, cost)
        if cost <= 0.1 and flag == 1:
            zero_x = i+1
            flag = 0

    # 绘制cost随迭代次数的变化情况图
    axis = 'cost<=0.1, '+'('+str(zero_x)+','+'0'+')'
    plt.plot(np.linspace(1, max_iteration, max_iteration), costArray, 'b-.', label='cost')
    if zero_x!=0:
        plt.plot(zero_x, 0, 'm^', label=axis)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("cost随迭代次数的变化情况")
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.grid()
    plt.legend()
    plt.show()

    return parameters, zero_x


def autoEncoder(X, nodes_layer, layer_num, learning_rate, max_iteration, method):
    # 将数据变成一列
    row = X.shape[0]
    col = X.shape[1]
    X_reshape = X.reshape(row * col, 1)

    # 数据归一化
    scaler = MinMaxScaler().fit(X_reshape)
    X_scaler = scaler.transform(X_reshape)

    # 放到BPNN网络里面进行训练
    parameters, zero_x = BPNN(X_scaler, nodes_layer, layer_num, max_iteration, learning_rate, scaler, method)

    # 以下为自编码过程
    W = parameters['W']
    b = parameters['b']

    print('W:\n',W)
    print('\n\nb:\n',b)

    # 通过前向传播计算Z和A
    Z = [0]
    A = [0]
    A[0] = X_scaler

    for i in range(1, layer_num):
        zi = np.dot(W[i], A[i-1]) + b[i]
        if method == 'sig':
            ai = sigmoid(zi)
        elif method == 'tan_h':
            ai = tanh(zi)
        Z.append(np.array(zi))
        A.append(np.array(ai))

    output = A[layer_num - 1]
    X_recode = scaler.inverse_transform(output)
    X_final = np.round(X_recode.reshape(row, col), decimals=1)
    print("\n自编码后的data：\n", X_final)
    # 计算精确度
    count = 0
    for i in range(0,row):
        for j in range(0,col):
            if X_final[i][j] == X[i][j]:
                count += 1
    precision = count/(row * col)
    return precision, zero_x


if __name__ == "__main__":
    # 获取数据
    iris = datasets.load_iris()
    data = iris["data"]
    X = data

    # 计算输入和输出数据长度，作为输入和输出层的节点个数
    size_io = X.shape[0] * X.shape[1]

    # 自定义神经网络的层数，每一层的节点个数用数组来存储
    nodes_layer = np.array([size_io, 10,3, 10, size_io])
    layer_num = len(nodes_layer)
    # 学习率
    learning_rate = 0.03
    # 最大迭代次数
    max_iteration = 10000
    # 激活函数
    method = 'sig'
    precision, zero_x = autoEncoder(X, nodes_layer, layer_num, learning_rate, max_iteration, 'tan_h')

    print("该模型的精确度为：", str(np.round(precision*100, 3))+'%')