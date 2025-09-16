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
    raise NotImplementedError


def accuracy(y_true, y_pred):
    """
    TODO: Implement accuracy function
    """
    raise NotImplementedError


def confusion_matrix(y_true, y_pred):
    """
    TODO: Implement confusion_matrix function
    """
    raise NotImplementedError


def roc_curve(y_true, scores):
    """
    TODO: Implement ROCcurve function
    """
    raise NotImplementedError


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

    # TODO: you may choose to split training set to form a validation set

    model = LogisticRegression(lr=LR, epochs=EPOCHS)  # You may choose to pass more hyper-parameters to your model
    model.fit(X_train, y_train)


    proba = model.predict_proba(X_test)

    y_pred = (proba >= THRESHOLD).astype(float)
    print(f">> Test Accuracy: {accuracy(y_test, y_pred):.4f}")

    print(">> Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, y_pred))

    auc = plot_roc(y_test, proba)
    print(f">> Test AUC: {auc:.4f}")
