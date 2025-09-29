import numpy as np
import matplotlib.pyplot as plt
import random
class LogisticRegression:
    def __init__(self, lr=0.1, epochs=2000, batch_size=32, reg_lambda=0.0, optimizer='adam',
                 beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.epochs = epochs
        # You may choose to use more hyper-parameters
        self.batch_size = batch_size
        self.reg_lambda = reg_lambda
        self.optimizer = optimizer
        self.beta1 = beta1      # Adam的一阶矩估计的指数衰减率
        self.beta2 = beta2      # Adam的二阶矩估计的指数衰减率
        self.epsilon = epsilon  # 防止除零的小数值
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.W = None
        self.b = None
        # Adam优化器的变量
        self.m_W = None         # 一阶矩估计（动量）
        self.v_W = None         # 二阶矩估计（自适应学习率）
        self.m_b = None
        self.v_b = None
        self.t = 0             # 时间步

    def initialize_parameters(self, d):
        """
        TODO: 初始化 self.W 和 self.b
        参数:
        d -- 特征的数量
        """
        # 使用Xavier/Glorot初始化，比零初始化更好
        self.W = np.random.randn(d, 1) * np.sqrt(2.0 / d)
        self.b = 0

        # 初始化Adam优化器的变量
        if self.optimizer == 'adam':
            self.m_W = np.zeros_like(self.W)
            self.v_W = np.zeros_like(self.W)
            self.m_b = 0
            self.v_b = 0
            self.t = 0

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
    
    def predict(self, X, threshold=0.5):
        """
        根据给定的阈值，对输入数据进行二元分类预测。
        """
        # 调用 predict_proba 来获取概率
        probas = self.predict_proba(X)
        # 根据阈值将概率转换为 0 或 1
        return (probas >= threshold).astype(float)
    
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
        l1_regularization_cost = (self.reg_lambda / m) * np.sum(np.abs(self.W))
        total_loss = cross_entropy_loss + l1_regularization_cost
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

            # L1正则化的梯度（次梯度）
            l1_gradient = self.reg_lambda * np.sign(self.W)
            dW = dW_original + l1_gradient

            return dW, db

    def update_parameters(self, dW, db):
        """
        根据选择的优化器更新参数
        """
        if self.optimizer == 'adam':
            return self._adam_update(dW, db)
        else:
            # 默认使用SGD
            self.W = self.W - self.lr * dW
            self.b = self.b - self.lr * db

    def _adam_update(self, dW, db):
        """
        Adam优化器的参数更新
        """
        self.t += 1

        # 更新一阶矩估计（动量）
        self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * dW
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db

        # 更新二阶矩估计（自适应学习率）
        self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * (dW ** 2)
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)

        # 计算偏差修正后的一阶矩和二阶矩估计
        m_W_hat = self.m_W / (1 - self.beta1 ** self.t)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_W_hat = self.v_W / (1 - self.beta2 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

        # 更新参数
        self.W = self.W - self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
        self.b = self.b - self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def fit(self, X_train, y_train, X_val, y_val, data_iter_fn, accuracy_fn, threshold):
            # 确保 y 的形状正确
            y_train = y_train.reshape(-1, 1)
            y_val = y_val.reshape(-1, 1)

            d = X_train.shape[1]
            self.initialize_parameters(d)
            
            # --- [核心修改] 创建一个包含两个子图的窗口 ---
            # fig 是整个图窗, (ax1, ax2) 是两个子图的坐标轴
            plt.ion()
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
            fig.tight_layout(pad=3.0) # 增加子图间距
            # -------------------------------------------
            
            NUM_UPDATES = 50
            update_interval = self.epochs // NUM_UPDATES if self.epochs > NUM_UPDATES else 1

            for epoch in range(self.epochs):
                # Mini-Batch 训练
                for X_batch, y_batch in data_iter_fn(self.batch_size, X_train, y_train):
                    y_hat_batch = self.predict_proba(X_batch)
                    dW, db = self.compute_gradients(X_batch, y_batch, y_hat_batch)
                    # 使用选择的优化器更新参数
                    self.update_parameters(dW, db)

                # --- 在每个epoch结束后，计算 Loss 和 Accuracy ---
                # 计算 Loss
                y_hat_train_loss = self.predict_proba(X_train)
                train_loss = self.compute_loss(y_train, y_hat_train_loss)
                self.train_loss_history.append(train_loss)
                
                y_hat_val_loss = self.predict_proba(X_val)
                val_loss = self.compute_loss(y_val, y_hat_val_loss)
                self.val_loss_history.append(val_loss)

                # 计算 Accuracy
                train_pred = self.predict(X_train, threshold)
                train_acc = accuracy_fn(y_train, train_pred)
                self.train_acc_history.append(train_acc)

                val_pred = self.predict(X_val, threshold)
                val_acc = accuracy_fn(y_val, val_pred)
                self.val_acc_history.append(val_acc)
                # ----------------------------------------------------
                
                # --- 定期更新两个子图 ---
                if (epoch + 1) % update_interval == 0 or epoch == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
                    
                    # 更新第一个子图 (Loss)
                    ax1.clear()
                    ax1.plot(self.train_loss_history, label='Train Loss')
                    ax1.plot(self.val_loss_history, label='Validation Loss')
                    ax1.legend()
                    ax1.set_title("Loss Curve")

                    # 更新第二个子图 (Accuracy)
                    ax2.clear()
                    ax2.plot(self.train_acc_history, label='Train Accuracy')
                    ax2.plot(self.val_acc_history, label='Validation Accuracy')
                    ax2.legend()
                    ax2.set_title("Accuracy Curve")
                    ax2.set_ylim([0, 1.05])
                    
                    plt.pause(0.01)
                # -------------------------

            # ... (训练结束后的打印代码) ...
            print(f"Training finished. Final Val Acc: {self.val_acc_history[-1]:.4f}")

            # --- 显示最终的两个静态子图 ---
            ax1.clear()
            ax1.plot(self.train_loss_history, label='Train Loss')
            ax1.plot(self.val_loss_history, label='Validation Loss')
            ax1.legend()
            ax1.set_title("Final Loss Curve")

            ax2.clear()
            ax2.plot(self.train_acc_history, label='Train Accuracy')
            ax2.plot(self.val_acc_history, label='Validation Accuracy')
            ax2.legend()
            ax2.set_title("Final Accuracy Curve")
            ax2.set_ylim([0, 1.05])

            plt.ioff()
            plt.show()