import numpy as np
from matplotlib import pyplot as plt
from .metrics import mean_squared_error, r2_score

class SGD:
    def __init__(self, lr=0.001, epochs=1000, batch_size=1024, tol=1e-3, momentum=0.9, l2_lambda=0.01, decay_rate=0.95):
        self.initial_lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.tolerance = tol
        self.momentum = momentum
        self.l2_lambda = l2_lambda
        self.decay_rate = decay_rate  # Decay rate for learning rate
        self.weights = None
        self.bias = None
        self.velocity_weights = None
        self.velocity_bias = None
        self.learning_rate = lr

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def gradient(self, X_batch, y_batch):
        y_pred = self.predict(X_batch)
        error = y_pred - y_batch
        gradient_weights = (np.dot(X_batch.T, error) / X_batch.shape[0]) + (self.l2_lambda * self.weights)
        gradient_bias = np.mean(error)
        return gradient_weights, gradient_bias

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        self.velocity_weights = np.zeros(n_features)
        self.velocity_bias = 0
        self.losses = []
        self.learning_rates = []

        for epoch in range(self.epochs):
            # Update learning rate with decay
            self.learning_rate = self.initial_lr * (self.decay_rate ** epoch)

            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                gradient_weights, gradient_bias = self.gradient(X_batch, y_batch)
                self.velocity_weights = self.momentum * self.velocity_weights - self.learning_rate * gradient_weights
                self.velocity_bias = self.momentum * self.velocity_bias - self.learning_rate * gradient_bias
                self.weights += self.velocity_weights
                self.bias += self.velocity_bias

            y_pred = self.predict(X)
            loss = mean_squared_error(y, y_pred)
            self.losses.append(loss)
            self.learning_rates.append(self.learning_rate)

            if epoch % 100 == 0:
                r2 = r2_score(y, y_pred)
                print(f"Epoch {epoch}: Loss {loss}, R² {r2}")
            if np.linalg.norm(gradient_weights) < self.tolerance:
                print("Convergence reached.")
                break

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(range(len(self.losses)), self.losses, "b-", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Loss function over epochs")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(len(self.learning_rates)), self.learning_rates, "r-", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate over Epochs")
        plt.grid(True)

        plt.show()

        return self.weights, self.bias
