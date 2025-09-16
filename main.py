import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.logistic_regression import LogisticRegression

# ======================= User Config =======================
VAL_SIZE = 0.15
LR = 0.0005
EPOCHS = 10000
THRESHOLD = 0.5       
SEED = 0
# ===========================================================

TRAIN_DATA_PATH = f"./data/classification_data_train.csv"  # Please put data csv in the `data` folder
TEST_DATA_PATH = f"./data/classification_data_test.csv"  # Please put data csv in the `data` folder


def rmse(y_true, y_pred):
    """
    TODO: Implement rmse function
    """
    return np.sqrt(np.mean((y_true - y_pred)**2))


def accuracy(y_true, y_pred):
    """
    TODO: 实现 accuracy 函数
    """
    # y_true == y_pred 会生成一个布尔数组 (True/False)
    # np.mean 会将 True 当作 1，False 当作 0 来计算平均值，这恰好就是准确率
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    """
    TODO: 实现 confusion_matrix 函数
    """
    y_true = y_true.flatten() # 确保是一维数组
    y_pred = y_pred.flatten() # 确保是一维数组
    
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TP = np.sum((y_true == 1) & (y_pred == 1))
    
    return np.array([[TN, FP], [FN, TP]])


def roc_curve(y_true, scores):
    """
    TODO: 实现 ROC 曲线函数
    """
    y_true = y_true.flatten()
    scores = scores.flatten()

    # 将阈值从高到低排序
    thresholds = np.unique(scores)[::-1]
    
    tpr_list = [0]
    fpr_list = [0]
    
    num_positives = np.sum(y_true == 1)
    num_negatives = np.sum(y_true == 0)

    for thresh in thresholds:
        y_pred = (scores >= thresh).astype(int)
        
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        
        tpr = TP / num_positives if num_positives > 0 else 0
        fpr = FP / num_negatives if num_negatives > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # 计算 AUC (Area Under Curve)
    auc = np.trapz(tpr_list, fpr_list)
    
    return np.array(fpr_list), np.array(tpr_list), auc


def plot_roc(y_true, scores, title="ROC Curve"):

    y_true = y_true.reshape(-1)
    scores = scores.reshape(-1)

    fpr, tpr, auc = roc_curve(y_true, scores)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

    return auc


def load_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    X = df.drop(columns="target").values
    y = df["target"].values
    print(f"data loaded from {csv_file_path}")
    return X, y


if __name__ == "__main__":
    X_train, y_train = load_data(TRAIN_DATA_PATH)
    X_test, y_test = load_data(TEST_DATA_PATH)

    # TODO: you might want to standardize features
    # 计算训练集的均值和标准差
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # 对训练集和测试集进行标准化
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # TODO: you may choose to split training set to form a validation set
    # 设置随机种子以保证每次划分结果都一样
    np.random.seed(SEED)
    
    # 生成一个随机的索引排列
    shuffled_indices = np.random.permutation(len(X_train))
    
    # 计算验证集的大小
    val_set_size = int(len(X_train) * VAL_SIZE)
    
    # 划分索引
    val_indices = shuffled_indices[:val_set_size]
    train_indices = shuffled_indices[val_set_size:]
    
    # 根据索引划分数据
    X_val, y_val = X_train[val_indices], y_train[val_indices]
    X_train, y_train = X_train[train_indices], y_train[train_indices]

    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    model = LogisticRegression(lr=LR, epochs=EPOCHS)  # You may choose to pass more hyper-parameters to your model
    model.fit(X_train, y_train)


    proba = model.predict_proba(X_test)

    y_pred = (proba >= THRESHOLD).astype(float)
    print(f">> Test Accuracy: {accuracy(y_test, y_pred):.4f}")

    print(">> Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, y_pred))

    auc = plot_roc(y_test, proba)
    print(f">> Test AUC: {auc:.4f}")
