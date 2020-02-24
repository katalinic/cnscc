import matplotlib.pyplot as plt
import numpy as np
import pickle

Q = np.array([
    [0.15, 0.1],
    [0.1, 0.12]
])

# Q1.
eigs, eigvs = np.linalg.eigh(Q)
# print(eigvs)


with open('w7/c10p1.pickle', 'rb') as f:
    data = pickle.load(f)

X = data['c10p1']
X -= np.mean(X, axis=0)

# plt.scatter(X[:, 0], X[:, 1])
# plt.show()

eta = 1.0
alpha = 1.0
dt = 1e-2

C = X.T @ X / X.shape[0]
eigs, eigvs = np.linalg.eigh(C)
# print(eigs)
# print(eigvs)


def train(data, alpha, num_iters):
    w = np.random.normal(data.shape[1])
    w /= np.linalg.norm(w)

    for _ in range(num_iters):
        for x in data:
            v = np.sum(x * w)
            w += dt * eta * (v * x - alpha * v ** 2 * w)

    return w


# plt.scatter(X[:, 0], X[:, 1])
# for _ in range(100):
#     w = train(X, alpha, 100)
#     plt.plot([0, w[0]], [0, w[1]])
# plt.show()


# X_offset = X + np.array([3.0, -1.0])
# plt.scatter(X_offset[:, 0], X_offset[:, 1])
# for _ in range(100):
#     w = train(X_offset, 1.0, 100)
#     plt.plot([0, w[0]], [0, w[1]])
# plt.show()

# plt.scatter(X[:, 0], X[:, 1])
# for _ in range(10):
#     w = train(X, 0.0, 20)
#     plt.plot([0, w[0]], [0, w[1]])
# plt.show()
