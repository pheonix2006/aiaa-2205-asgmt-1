import numpy as np
import matplotlib.pyplot as plt
import random
class LogisticRegression:
    def __init__(self, lr=0.1, epochs=2000, batch_size=32,reg_lambda=0.0):
        self.lr = lr
        self.epochs = epochs
        # You may choose to use more hyper-parameters
        self.batch_size = batch_size
        self.reg_lambda = reg_lambda
        self.train_loss_history = []
        self.val_loss_history = []
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
            TODO: 实现 sigmoid 函数 (数值稳定版)
            """
            z = np.clip(z, -500, 500)
            
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
        cross_entropy_loss = - (1/m) * np.sum(y_true * np.log(y_hat + epsilon) + (1 - y_true) * np.log(1 - y_hat + epsilon))
        l2_regularization_cost = (self.reg_lambda / (2 * m)) * np.sum(np.square(self.W))
        total_loss = cross_entropy_loss + l2_regularization_cost
        # np.squeeze 会移除数组中维度为1的条目，将损失值从一个数组变成一个标量
        return np.squeeze(total_loss)

    def compute_gradients(self, X, y_true, y_hat):
            """
            TODO: 推导并实现梯度 dW, db
            """
            m = X.shape[0]
            
            # 计算预测值与真实值之间的误差
            error = y_hat - y_true
            
            # 计算梯度
            dW_original = (1/m) * np.dot(X.T, error)
            db = (1/m) * np.sum(error)

            dW = dW_original + (self.reg_lambda / m) * self.W
            
            return dW, db

    def fit(self, X_train, y_train, X_val, y_val, data_iter): # <--- 修改了输入参数
        # 确保 y 的形状正确
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)

        n, d = X_train.shape
        self.initialize_parameters(d)

        # --- 绘图设置 ---
        NUM_UPDATES = 50
        update_interval = self.epochs // NUM_UPDATES if self.epochs > NUM_UPDATES else 1
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 6))
        # -----------------

        for epoch in range(self.epochs):
            # --- Mini-Batch 核心：使用 data_iter 遍历小批量数据 ---
            for X_batch, y_batch in data_iter(self.batch_size, X_train, y_train):
                # 在当前批次上计算预测和梯度
                y_hat_batch = self.predict_proba(X_batch)
                dW, db = self.compute_gradients(X_batch, y_batch, y_hat_batch)
                
                # 更新权重
                self.W = self.W - self.lr * dW
                self.b = self.b - self.lr * db
            # ----------------------------------------------------

            # --- 在每个epoch结束后，计算整个数据集的损失用于绘图 ---
            y_hat_train = self.predict_proba(X_train)
            train_loss = self.compute_loss(y_train, y_hat_train)
            self.train_loss_history.append(train_loss)

            y_hat_val = self.predict_proba(X_val)
            val_loss = self.compute_loss(y_val, y_hat_val)
            self.val_loss_history.append(val_loss)
            # ----------------------------------------------------
            
            # --- 定期更新图像 (这部分代码保持不变) ---
            if (epoch + 1) % update_interval == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                ax.clear()
                ax.plot(self.train_loss_history, label='Train Loss (Mini-Batch)')
                ax.plot(self.val_loss_history, label='Validation Loss (Mini-Batch)')
                ax.legend()
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Training vs. Validation Loss")
                plt.pause(0.01)

        # ... (训练结束后的打印和最终绘图代码保持不变) ...
        print(f"Training finished. Final Train Loss: {self.train_loss_history[-1]:.4f}, Final Val Loss: {self.val_loss_history[-1]:.4f}")
        ax.clear()
        ax.plot(self.train_loss_history, label='Train Loss (Mini-Batch)')
        ax.plot(self.val_loss_history, label='Validation Loss (Mini-Batch)')
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Final Training vs. Validation Loss")
        plt.ioff()
        plt.show()