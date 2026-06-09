import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


# ==================================================
# LOAD DATASET
# ==================================================
iris = load_iris()
X = iris.data
y = iris.target

# Random train-test split every run
test_size = np.random.uniform(0.15, 0.35)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=None,
    shuffle=True
)

# ==================================================
# DECISION TREE
# ==================================================
print("\n" + "=" * 50)
print("DECISION TREE")
print("=" * 50)

max_depth = np.random.randint(2, 8)

dt = DecisionTreeClassifier(
    max_depth=max_depth,
    random_state=None
)

dt.fit(X_train, y_train)

pred = dt.predict(X_test)

print("Tree Depth:", max_depth)
print("Decision Tree Accuracy:",
      round(accuracy_score(y_test, pred), 4))

plt.figure(figsize=(12, 7))

plot_tree(
    dt,
    filled=True,
    feature_names=iris.feature_names,
    class_names=iris.target_names
)

plt.title(f"Decision Tree (Depth={max_depth})")
plt.show()


# ==================================================
# RANDOM FOREST
# ==================================================
print("\n" + "=" * 50)
print("RANDOM FOREST")
print("=" * 50)

n_estimators = np.random.randint(50, 201)
max_depth = np.random.randint(2, 10)

rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=None
)

rf.fit(X_train, y_train)

pred = rf.predict(X_test)

print("Trees:", n_estimators)
print("Max Depth:", max_depth)
print("Random Forest Accuracy:",
      round(accuracy_score(y_test, pred), 4))

plt.figure(figsize=(8, 5))

plt.bar(
    iris.feature_names,
    rf.feature_importances_,
    color="skyblue"
)

plt.title("Random Forest Feature Importance")
plt.ylabel("Importance")
plt.xticks(rotation=20)
plt.grid(axis="y")
plt.show()


# ==================================================
# SUPPORT VECTOR MACHINE
# ==================================================
print("\n" + "=" * 50)
print("SUPPORT VECTOR MACHINE")
print("=" * 50)

kernel = np.random.choice([
    "linear",
    "rbf",
    "poly"
])

C = np.random.uniform(0.1, 10)

svm = SVC(
    kernel=kernel,
    C=C
)

svm.fit(X_train, y_train)

pred = svm.predict(X_test)

print("Kernel:", kernel)
print("C:", round(C, 3))
print("SVM Accuracy:",
      round(accuracy_score(y_test, pred), 4))


# ==================================================
# PCA
# ==================================================
print("\n" + "=" * 50)
print("PCA")
print("=" * 50)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 5))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="viridis"
)

plt.title("PCA Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter)
plt.grid(True)
plt.show()


# ==================================================
# REINFORCEMENT LEARNING (Q TABLE)
# ==================================================
print("\n" + "=" * 50)
print("REINFORCEMENT LEARNING (Q TABLE)")
print("=" * 50)

grid_size = 5

q_table = np.zeros(
    (grid_size, grid_size, 4)
)

alpha = np.random.uniform(0.05, 0.3)
gamma = np.random.uniform(0.7, 0.99)
epsilon = np.random.uniform(0.05, 0.3)

episodes = np.random.randint(300, 1000)

goal = (4, 4)

for episode in range(episodes):

    state = [0, 0]

    while tuple(state) != goal:

        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(
                q_table[state[0], state[1]]
            )

        next_state = state.copy()

        if action == 0 and state[0] > 0:
            next_state[0] -= 1

        elif action == 1 and state[0] < 4:
            next_state[0] += 1

        elif action == 2 and state[1] > 0:
            next_state[1] -= 1

        elif action == 3 and state[1] < 4:
            next_state[1] += 1

        reward = (
            10
            if tuple(next_state) == goal
            else -1
        )

        old_q = q_table[
            state[0],
            state[1],
            action
        ]

        next_max = np.max(
            q_table[
                next_state[0],
                next_state[1]
            ]
        )

        q_table[
            state[0],
            state[1],
            action
        ] = old_q + alpha * (
            reward +
            gamma * next_max -
            old_q
        )

        state = next_state

print("Episodes:", episodes)
print("Alpha:", round(alpha, 3))
print("Gamma:", round(gamma, 3))
print("Epsilon:", round(epsilon, 3))

print("\nQ Values at Start State:")
print(q_table[0, 0])


# ==================================================
# SIMPLE LSTM CELL
# ==================================================
print("\n" + "=" * 50)
print("SIMPLE LSTM CELL")
print("=" * 50)

input_size = 3
hidden_size = 4

x = np.random.randn(input_size)

h_prev = np.random.randn(hidden_size)

c_prev = np.random.randn(hidden_size)

Wf = np.random.randn(
    hidden_size,
    input_size + hidden_size
)

combined = np.concatenate(
    [h_prev, x]
)

forget_gate = 1 / (
    1 + np.exp(
        -np.dot(Wf, combined)
    )
)

print("Input Vector:")
print(x)

print("\nForget Gate Output:")
print(forget_gate)


# ==================================================
# SIMPLE Q NETWORK
# ==================================================
print("\n" + "=" * 50)
print("SIMPLE Q NETWORK")
print("=" * 50)

input_nodes = 4
hidden_nodes = np.random.randint(8, 33)
output_nodes = 2

W1 = np.random.randn(
    input_nodes,
    hidden_nodes
)

W2 = np.random.randn(
    hidden_nodes,
    output_nodes
)

sample_state = np.random.randn(
    1,
    input_nodes
)

hidden = np.maximum(
    0,
    np.dot(sample_state, W1)
)

q_values = np.dot(
    hidden,
    W2
)

print("Hidden Nodes:", hidden_nodes)

print("\nSample State:")
print(sample_state)

print("\nQ Values:")
print(q_values)