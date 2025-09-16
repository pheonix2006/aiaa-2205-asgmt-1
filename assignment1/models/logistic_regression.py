import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=2000):
        self.lr = lr
        self.epochs = epochs
        # You may choose to use more hyper-parameters

        self.loss_history = []
        self.W = None
        self.b = None

    def initialize_parameters(d):
        """
        TODO: initialize self.W and self.b
        """
        raise NotImplementedError

    def sigmoid(z):
        """
        TODO: Implement sigmoid function
        """
        raise NotImplementedError

    def predict_proba(self, X):
        """
        TODO: Implement your model
        """
        raise NotImplementedError

    def compute_loss(self, y_true, y_hat):
        """
        TODO: Implement loss
        """
        raise NotImplementedError

    def compute_gradients(self, X, y_true, y_hat):
        """
        TODO: Derive and implement gradients dW, db
        """
        raise NotImplementedError


    def fit(self, X, y):
        n, d = X.shape
        self.initialize_parameters(d)

        for epoch in range(self.epochs):
            y_hat = self.predict_proba(X)

            loss = self.compute_loss(y, y_hat)
            self.loss_history.append(loss)

            dW, db = self.compute_gradients(X, y, y_hat)

            # TODO: Gradient descent update
            # self.W = ?
            # self.b = ?


        # TODO: plot loss vs epoch
