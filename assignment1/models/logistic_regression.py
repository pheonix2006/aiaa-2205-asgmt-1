import numpy as np
import matplotlib.pyplot as plt
class LogisticRegression:
    def __init__(self, lr=0.1, epochs=2000):
        self.lr = lr
        self.epochs = epochs
        # You may choose to use more hyper-parameters

        self.loss_history = []
        self.W = None
        self.b = None

    def initialize_parameters(self, d):
        """
        TODO: 初始化 self.W 和 self.b
        参数:
        d -- 特征的数量
        """
        # 将权重 W 初始化为 (d, 1) 的零向量
        self.W = np.zeros((d, 1))
        # 将偏置 b 初始化为 0
        self.b = 0

    def sigmoid(self, z):
        """
        TODO: 实现 sigmoid 函数
        """
        # 计算 z 的 sigmoid 值
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        """
        TODO: 实现你的模型
        参数:
        X -- 输入数据，形状为 (n_samples, n_features)
        返回:
        y_hat -- 每个样本属于类别1的预测概率，形状为 (n_samples, 1)
        """
        # 计算线性组合 z = XW + b
        z = np.dot(X, self.W) + self.b
        
        # 将 z 通过 sigmoid 函数转换为概率
        y_hat = self.sigmoid(z)
        
        return y_hat

    def compute_loss(self, y_true, y_hat):
        """
        TODO: 实现损失函数
        参数:
        y_true -- 真实标签，形状 (n_samples, 1)
        y_hat  -- 预测概率，形状 (n_samples, 1)
        返回:
        loss -- 交叉熵损失值
        """
        m = y_true.shape[0]
        # 为了防止 log(0) 的情况，给 y_hat 加上一个极小值 epsilon
        epsilon = 1e-9
        loss = - (1/m) * np.sum(y_true * np.log(y_hat + epsilon) + (1 - y_true) * np.log(1 - y_hat + epsilon))
        
        # np.squeeze 会移除数组中维度为1的条目，将损失值从一个数组变成一个标量
        return np.squeeze(loss)

    def compute_gradients(self, X, y_true, y_hat):
            """
            TODO: 推导并实现梯度 dW, db
            """
            m = X.shape[0]
            
            # 计算预测值与真实值之间的误差
            error = y_hat - y_true
            
            # 计算梯度
            dW = (1/m) * np.dot(X.T, error)
            db = (1/m) * np.sum(error)
            
            return dW, db


    def fit(self, X, y):
        n, d = X.shape
        self.initialize_parameters(d)

        for epoch in range(self.epochs):
            y_hat = self.predict_proba(X)

            loss = self.compute_loss(y, y_hat)
            self.loss_history.append(loss)

            dW, db = self.compute_gradients(X, y, y_hat)

            # TODO: 梯度下降更新
            self.W = self.W - self.lr * dW
            self.b = self.b - self.lr * db


        # TODO: plot loss vs epoch
        plt.plot(range(self.epochs), self.loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.show()