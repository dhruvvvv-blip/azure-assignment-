import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


# =========================
# Linear Regression
# =========================
class LinearRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            loss = np.mean((y - y_pred) ** 2)
            self.losses.append(loss)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# =========================
# Logistic Regression
# =========================
class LogisticRegressionGD:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            linear = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear)
        return np.array([1 if i > 0.5 else 0 for i in y_pred])


# ==================================================
# LINEAR REGRESSION EXPERIMENT
# ==================================================
print("\n" + "=" * 50)
print("LINEAR REGRESSION")
print("=" * 50)

noise = np.random.randint(10, 60)

X, y = make_regression(
    n_samples=200,
    n_features=1,
    noise=noise,
    random_state=None
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=None
)

lr = np.random.uniform(0.001, 0.05)
epochs = np.random.randint(500, 2000)

print(f"Noise Level : {noise}")
print(f"Learning Rate : {lr:.5f}")
print(f"Epochs : {epochs}")

model = LinearRegressionGD(
    lr=lr,
    epochs=epochs
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f"MSE : {mean_squared_error(y_test, predictions):.2f}")
print(f"R2 Score : {r2_score(y_test, predictions):.4f}")

# Sort for smooth regression line
sorted_idx = np.argsort(X_test[:, 0])

plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, label="Actual Data")
plt.plot(
    X_test[sorted_idx],
    predictions[sorted_idx],
    color="red",
    linewidth=2,
    label="Regression Line"
)
plt.title("Linear Regression")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(model.losses)
plt.title("Linear Regression Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()


# ==================================================
# LOGISTIC REGRESSION EXPERIMENT
# ==================================================
print("\n" + "=" * 50)
print("LOGISTIC REGRESSION")
print("=" * 50)

class_sep = np.random.uniform(0.5, 3.0)
flip_y = np.random.uniform(0.0, 0.1)

X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    class_sep=class_sep,
    flip_y=flip_y,
    random_state=None
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=None
)

lr = np.random.uniform(0.01, 0.5)
epochs = np.random.randint(500, 2000)

print(f"Class Separation : {class_sep:.2f}")
print(f"Label Noise : {flip_y:.3f}")
print(f"Learning Rate : {lr:.5f}")
print(f"Epochs : {epochs}")

clf = LogisticRegressionGD(
    lr=lr,
    epochs=epochs
)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=pred,
    cmap="coolwarm",
    edgecolors="black"
)

plt.title("Logistic Regression Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()