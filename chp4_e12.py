import numpy as np
from sklearn import datasets

np.random.seed(2042)

iris_dataset = datasets.load_iris()

X = iris_dataset["data"][:,(2,3)]
y = iris_dataset["target"]


X_with_bias = np.c_[np.ones((len(X), 1)), X]

test_ratio = 0.2
val_ratio = 0.2
total_size = len(X)

test_size = int(total_size * test_ratio)
val_size = int(total_size * val_ratio)
train_size = total_size - test_size - val_size

rnd_indices = np.random.permutation(total_size)

x_train, y_train = X_with_bias[rnd_indices[:train_size]], y[rnd_indices[:train_size]]
x_val, y_val = X_with_bias[rnd_indices[train_size:train_size + val_size]], y[rnd_indices[train_size:train_size + val_size]]
x_test, y_test = X_with_bias[rnd_indices[-test_size:]], y[rnd_indices[-test_size:]]

def one_hot_encoder(y):
    one_hot = np.zeros((len(y), y.max()+1))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

def soft_max(logits):
    exp = np.exp(logits)
    sum_exp = np.sum(exp, axis=1, keepdims=True)
    return exp / sum_exp

y_train_encode = one_hot_encoder(y_train)
y_val_encode = one_hot_encoder(y_val)
y_test_encode = one_hot_encoder(y_test)

n_features = x_train.shape[1]
n_outputs = y.max() + 1


n_iterations = 5001
m = len(x_train)
eta = 0.01
epsilon = 1e-7
alpha = 0.1


# gradient descent using softmax with l2 penalty
theta = np.random.rand(n_features, n_outputs)
for iteration in range(n_iterations):
    logits = np.dot(x_train, theta)
    y_proba = soft_max(logits)
    if iteration % 500 == 0:
        cross_entropy = -np.mean(np.sum(y_train_encode * np.log(y_proba + 1e-7), axis=1))
        l2_loss = 1/2 * np.sum(np.square(theta[1:]))
        loss = cross_entropy + alpha * l2_loss
        print(iteration, loss)
    error = y_proba - y_train_encode
    gradients = 1/m * x_train.T.dot(error) + np.r_[np.zeros([1, n_outputs]), alpha * theta[1:]]
    theta = theta - eta * gradients

# Checking the accuracy score of the validation set #                                     
logits = x_val.dot(theta)
y_proba = soft_max(logits)
y_predict = np.argmax(y_proba, axis=1)

accuracy_score = np.mean(y_predict == y_val)
print(accuracy_score)


