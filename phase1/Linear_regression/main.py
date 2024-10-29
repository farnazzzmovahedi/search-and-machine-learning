from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the train and test datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Drop the 'id' column in both train and test sets
train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])

# Separate features (X) and label (y) in train and test datasets
X_train = train_df.drop(columns=['FloodProbability'])
y_train = train_df['FloodProbability']

X_test = test_df.drop(columns=['FloodProbability'])
y_test = test_df['FloodProbability']

# corr_matrix = X_train.corr()
# print(corr_matrix)
# plt.figure(figsize=(10, 8))  # Adjust the size as needed
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# plt.title("Feature Correlation Matrix")
# plt.show()


# # Check the missing values in both train and test sets
# train_df_missing_values = train_df.isnull().sum()
# test_df_missing_values = test_df.isnull().sum()
# print(train_df_missing_values)
# print(test_df_missing_values)



class SGD:
    def __init__(self, lr=0.001, epochs=100, batch_size=64, tol=1e-3, momentum=0.9):
        self.learning_rate = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.tolerance = tol
        self.momentum = momentum
        self.weights = None
        self.bias = None
        self.velocity_weights = None  # Velocity for weights
        self.velocity_bias = None  # Velocity for bias

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

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

        for epoch in range(self.epochs):
            # Shuffle the data using numpy
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

            if epoch % 10 == 0:
                y_pred = self.predict(X)
                loss = self.mean_squared_error(y, y_pred)
                r2 = self.r2_score(y, y_pred)
                print(f"Epoch {epoch}: Loss {loss}, R² {r2}")

            if np.linalg.norm(gradient_weights) < self.tolerance:
                print("Convergence reached.")
                break

        return self.weights, self.bias


# Assuming X_train and y_train are defined with appropriate data
# Normalize or standardize the features to avoid instability
X_train_np = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
y_train_np = y_train  # Assuming y_train does not need normalization here

# Train the model with momentum
model = SGD(lr=0.001, epochs=100, batch_size=64, tol=1e-3, momentum=0.9)
w, b = model.fit(X_train_np, y_train_np)

# Normalize test features and predict
X_test_np = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)
y_pred = model.predict(X_test_np)

# Evaluate on test data
test_loss = model.mean_squared_error(y_test, y_pred)
r2_test = model.r2_score(y_test, y_pred)
print(f"Test Loss: {test_loss}, R²: {r2_test}")
