import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


# ==================================================
# IMAGE PROCESSING - BLUR FILTER
# ==================================================
print("\n" + "=" * 50)
print("IMAGE PROCESSING")
print("=" * 50)

img_size = np.random.randint(48, 97)

image = np.random.rand(img_size, img_size)

kernel_size = np.random.choice([3, 5])

blur_kernel = np.ones(
    (kernel_size, kernel_size)
) / (kernel_size * kernel_size)

pad = kernel_size // 2

blurred = np.zeros_like(image)

for i in range(pad, image.shape[0] - pad):
    for j in range(pad, image.shape[1] - pad):

        region = image[
            i - pad:i + pad + 1,
            j - pad:j + pad + 1
        ]

        blurred[i, j] = np.sum(
            region * blur_kernel
        )

print("Image Size:", image.shape)
print("Kernel Size:", kernel_size)

plt.figure(figsize=(6, 6))
plt.imshow(blurred, cmap="gray")
plt.title(f"Blurred Image ({kernel_size}x{kernel_size})")
plt.axis("off")
plt.show()


# ==================================================
# COMPUTER VISION - EDGE DETECTION
# ==================================================
print("\n" + "=" * 50)
print("EDGE DETECTION")
print("=" * 50)

sobel_choice = np.random.choice(["x", "y"])

if sobel_choice == "x":
    sobel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
else:
    sobel = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

edges = np.zeros_like(image)

for i in range(1, image.shape[0] - 1):
    for j in range(1, image.shape[1] - 1):

        region = image[
            i - 1:i + 2,
            j - 1:j + 2
        ]

        edges[i, j] = np.sum(
            region * sobel
        )

print("Sobel Direction:", sobel_choice.upper())

plt.figure(figsize=(6, 6))
plt.imshow(edges, cmap="gray")
plt.title(f"Edge Detection ({sobel_choice.upper()})")
plt.axis("off")
plt.show()


# ==================================================
# CNN DEMO
# ==================================================
print("\n" + "=" * 50)
print("CNN DEMO")
print("=" * 50)

input_dim = np.random.randint(24, 41)

input_image = np.random.rand(
    input_dim,
    input_dim
)

filter_size = np.random.choice([3, 5])

filter_kernel = np.random.rand(
    filter_size,
    filter_size
)

output_dim = input_dim - filter_size + 1

feature_map = np.zeros(
    (output_dim, output_dim)
)

for i in range(output_dim):
    for j in range(output_dim):

        region = input_image[
            i:i + filter_size,
            j:j + filter_size
        ]

        feature_map[i, j] = np.sum(
            region * filter_kernel
        )

relu_output = np.maximum(
    0,
    feature_map
)

print("Input Shape:",
      input_image.shape)

print("Filter Shape:",
      filter_kernel.shape)

print("Feature Map Shape:",
      relu_output.shape)

plt.figure(figsize=(6, 6))
plt.imshow(relu_output, cmap="gray")
plt.title("CNN Feature Map")
plt.axis("off")
plt.show()


# ==================================================
# IMAGE CLASSIFICATION
# ==================================================
print("\n" + "=" * 50)
print("IMAGE CLASSIFICATION")
print("=" * 50)

samples = np.random.randint(250, 501)

features = np.random.choice([
    128,
    256,
    512
])

X = np.random.rand(
    samples,
    features
)

y = np.random.randint(
    0,
    2,
    samples
)

test_size = np.random.uniform(
    0.15,
    0.35
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size
)

n_trees = np.random.randint(
    50,
    201
)

clf = RandomForestClassifier(
    n_estimators=n_trees
)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

acc = accuracy_score(
    y_test,
    pred
)

print("Samples:", samples)
print("Features:", features)
print("Trees:", n_trees)

print("\nAccuracy:",
      round(acc, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    pred
))


# ==================================================
# SIMPLE RNN
# ==================================================
print("\n" + "=" * 50)
print("SIMPLE RNN")
print("=" * 50)

input_size = np.random.randint(
    3,
    6
)

hidden_size = np.random.randint(
    4,
    9
)

sequence_length = np.random.randint(
    5,
    11
)

Wxh = np.random.randn(
    hidden_size,
    input_size
)

Whh = np.random.randn(
    hidden_size,
    hidden_size
)

h = np.zeros(
    (hidden_size, 1)
)

sequence = [
    np.random.randn(
        input_size,
        1
    )
    for _ in range(sequence_length)
]

for x in sequence:

    h = np.tanh(
        np.dot(Wxh, x)
        +
        np.dot(Whh, h)
    )

print("Input Size:", input_size)
print("Hidden Size:", hidden_size)
print("Sequence Length:", sequence_length)

print("\nFinal Hidden State:")
print(h)


# ==================================================
# SUMMARY
# ==================================================
print("\n" + "=" * 50)
print("EXECUTION COMPLETE")
print("=" * 50)

print("Topics Covered:")
print("1. Image Processing")
print("2. Edge Detection")
print("3. CNN Feature Extraction")
print("4. Image Classification")
print("5. Recurrent Neural Network")