import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def plot_dataset(X, y):
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X, y)
    plt.title("dataset")
    plt.xlabel("first feature")
    plt.ylabel("second feature")
    plt.show()

def train_gradient_descent(X, y, learning_rate=0.01, n_iters=100):
    """Gradient descent optimization"""
    n_samples, n_features = X.shape
    weights = np.zeros(shape=(n_features, 1))
    bias = 0
    costs = []
    for i in range(n_iters):
        y_predict = np.dot(X, weights) + bias

        cost = (1 / n_samples) * np.sum((y_predict - y) ** 2)
        costs.append(cost)

        if i % 100 == 0:
            print(f"Cost of iteration {i}: {cost}")

        dJ_dw = (2 / n_samples) * np.dot(X.T, (y_predict - y))
        dJ_db = (2 / n_samples) * np.sum((y_predict - y))

        weights = weights - learning_rate * dJ_dw
        bias = bias - learning_rate * dJ_db
    return weights, bias, costs

def predict(X, weights, bias):
    """Predict function to calculate results"""
    return np.dot(X, weights) + bias


np.random.seed(123)
X = 2 * np.random.rand(500, 1)
y = 5 + 3 * X + np.random.randn(500, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y)

w_trained, b_trained, costs = train_gradient_descent(X_train,
                                                     y_train,
                                                     learning_rate=0.005,
                                                     n_iters=600)

fig = plt.figure(figsize=(8, 6))
plt.plot(np.arange(600), costs)
plt.title("Development of cost during training")
plt.xlabel("Number of iterations")
plt.ylabel("Cost")

y_p_test = predict(X_test, w_trained, b_trained)
fig = plt.figure(figsize=(8, 6))
plt.title("Dataset in blue, predictions for test set in orange")
plt.scatter(X_train, y_train)
plt.scatter(X_test, y_p_test)
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()
