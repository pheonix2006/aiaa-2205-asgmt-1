import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from models.logistic_regression import LogisticRegression

# ======================= User Config =======================
VAL_SIZE = 0.15
LR = 0.001          # 提高学习率，L1正则化需要更大的学习率
EPOCHS = 3000        # 减少训练轮数，防止过拟合
THRESHOLD = 0.5
SEED = 0
BATCH_SIZE = 32
REG_LAMBDA = 0.01    # 降低L1正则化系数，防止过度惩罚
# ===========================================================

TRAIN_DATA_PATH = f"./data/classification_data_train.csv"  # Please put data csv in the `data` folder
TEST_DATA_PATH = f"./data/classification_data_test.csv"  # Please put data csv in the `data` folder

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def rmse(y_true, y_pred):
    """
    TODO: Implement rmse function
    """
    return np.sqrt(np.mean((y_true - y_pred)**2))


def accuracy(y_true, y_pred):
    """
    TODO: 实现 accuracy 函数
    """
    # [修正] 使用 .flatten() 来确保两个数组都是一维的，避免广播错误
    return np.mean(y_true.flatten() == y_pred.flatten())


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
    
    model = LogisticRegression(lr=LR, epochs=EPOCHS,batch_size=BATCH_SIZE,reg_lambda=REG_LAMBDA)  # You may choose to pass more hyper-parameters to your model
    model.fit(X_train, y_train, X_val, y_val, data_iter,accuracy,THRESHOLD)


    proba = model.predict_proba(X_test)

    y_pred = (proba >= THRESHOLD).astype(float)
    test_accuracy = accuracy(y_test, y_pred)
    print(f">> Test Accuracy: {test_accuracy:.4f}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    print(">> Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(conf_matrix)

    auc = plot_roc(y_test, proba)
    print(f">> Test AUC: {auc:.4f}")

    # 将测试结果保存到txt文件
    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write("=== L1正则化逻辑回归测试结果 ===\n\n")
        f.write(f"数据集信息:\n")
        f.write(f"- 训练集大小: {len(X_train)}\n")
        f.write(f"- 验证集大小: {len(X_val)}\n")
        f.write(f"- 测试集大小: {len(X_test)}\n\n")
        f.write(f"超参数设置:\n")
        f.write(f"- 学习率(LR): {LR}\n")
        f.write(f"- 训练轮数(EPOCHS): {EPOCHS}\n")
        f.write(f"- 批量大小(BATCH_SIZE): {BATCH_SIZE}\n")
        f.write(f"- L1正则化系数(REG_LAMBDA): {REG_LAMBDA}\n")
        f.write(f"- 分类阈值(THRESHOLD): {THRESHOLD}\n\n")
        f.write(f"测试结果:\n")
        f.write(f"- 测试准确率: {test_accuracy:.4f}\n")
        f.write(f"- AUC值: {auc:.4f}\n")
        f.write(f"- 混淆矩阵:\n")
        f.write(f"  [[{conf_matrix[0,0]}, {conf_matrix[0,1]}],\n")
        f.write(f"   [{conf_matrix[1,0]}, {conf_matrix[1,1]}]]\n\n")

        # 计算额外指标
        TN, FP, FN, TP = conf_matrix[0,0], conf_matrix[0,1], conf_matrix[1,0], conf_matrix[1,1]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        f.write(f"- 精确率(Precision): {precision:.4f}\n")
        f.write(f"- 召回率(Recall): {recall:.4f}\n")
        f.write(f"- F1分数: {f1_score:.4f}\n")

    print("\n测试结果已保存到 test_results.txt 文件中")
