import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X = iris["data"][:,(2,3)]
y = iris["target"]

X_with_bias = np.c_[np.ones([len(X),1]), X]

np.random.seed(2042)

test_ratio = 0.2
val_ratio = 0.2
total_size = len(X_with_bias)

rnd_indices = np.random.permutation(total_size)

test_size = int(total_size * test_ratio)
val_size = int(total_size * val_ratio)
train_size = total_size - test_size - val_size

x_train, y_train = X_with_bias[rnd_indices[:train_size]], y[rnd_indices[:train_size]]
x_val, y_val = X_with_bias[rnd_indices[train_size:-test_size]], y[rnd_indices[train_size:-test_size]]
x_test, y_test = X_with_bias[rnd_indices[-test_size:]], y[rnd_indices[-test_size:]]


def one_hot(y):
    n_classes = y.max() + 1
    y_one_hot = np.zeros((len(y), n_classes))
    y_one_hot[np.arange(len(y)), y] = 1
    return y_one_hot

def soft_max(logits):
    exp = np.exp(logits)
    exp_sum = np.sum(exp, axis=1, keepdims=True)
    return exp / exp_sum
    
y_train_one_hot = one_hot(y_train)
y_val_one_hot = one_hot(y_val)
y_test_one_hot = one_hot(y_test)

n_inputs = x_train.shape[1]
n_outputs = len(np.unique(y_train))

eta = 0.01              # learning rate
n_iterations = 5001     # number of iterations
m = x_train.shape[0]    # number of features

theta = np.random.randn(n_inputs, n_outputs)
print(theta)

