import numpy as np
from matplotlib import pyplot as plt

from .metrics import mean_squared_error, r2_score

class SGD:
    def __init__(self, lr=0.001, epochs=100, batch_size=512, tol=1e-3, momentum=0.9):
        self.learning_rate = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.tolerance = tol
        self.momentum = momentum
        self.weights = None
        self.bias = None
        self.velocity_weights = None
        self.velocity_bias = None

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def gradient(self, X_batch, y_batch):
        y_pred = self.predict(X_batch)
        error = y_pred - y_batch
        gradient_weights = np.dot(X_batch.T, error) / X_batch.shape[0]
        gradient_bias = np.mean(error)
        return gradient_weights, gradient_bias

    def fit(self, X, y):
        # Ensure that X and y are numpy arrays
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.01  # Small random weights
        self.bias = 0  # Initialize bias with 0

        # Initialize velocities
        self.velocity_weights = np.zeros(n_features)
        self.velocity_bias = 0

        # List to store loss for each epoch
        self.losses = []

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                gradient_weights, gradient_bias = self.gradient(X_batch, y_batch)

                # Update velocities
                self.velocity_weights = self.momentum * self.velocity_weights - self.learning_rate * gradient_weights
                self.velocity_bias = self.momentum * self.velocity_bias - self.learning_rate * gradient_bias

                # Update weights and bias
                self.weights += self.velocity_weights
                self.bias += self.velocity_bias

            # Calculate and store loss at the end of each epoch
            y_pred = self.predict(X)
            loss = mean_squared_error(y, y_pred)
            self.losses.append(loss)  # Store each epoch's loss

            if epoch % 10 == 0:
                r2 = r2_score(y, y_pred)
                print(f"Epoch {epoch}: Loss {loss}, R² {r2}")

            if np.linalg.norm(gradient_weights) < self.tolerance:
                print("Convergence reached.")
                break

        # Plot the loss over epochs
        plt.plot(range(len(self.losses)), self.losses, "b-", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Loss function over epochs")
        plt.grid(True)
        plt.show()

        return self.weights, self.bias

