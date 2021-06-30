from NumericalDifferentiation import differentiate
import numpy as np

x_data = np.array([60, 73, 90, 55, 81, 70, 69, 95, 88, 83, 65])
y_data = [False, False, True, False, True,
          False, False, True, True, True, False]


def activate_sigmoid(z):
    return 1 / (1+np.exp(-z))


def hypothesis(x, w, b):
    g = np.dot(x, w) + b
    return activate_sigmoid(g)


def binaryCrossentropy(p, y):
    delta = 0.0000001
    return -np.sum(y*np.log(p+delta) + (1-y)*np.log(1-p+delta))


def calculate_cost(x, y, w, b):
    return binaryCrossentropy(hypothesis(x,w,b), y)


w = np.random.random((1, 1))
b = np.random.random(1)

x_data_normalized = (x_data - np.mean(x_data, axis=0)) / np.std(x_data, axis=0)
x_data_normalized = np.reshape(x_data_normalized, [len(x_data_normalized), 1])
y_data = np.array([(1. if element else 0.) for element in y_data]).reshape([len(y_data), 1])

for i in range(10001):
    if i % 100 == 0:
        print(f'ephoc : {i}, cost : {calculate_cost(x_data, y_data, w, b)}')
    w -= (0.01 * differentiate(lambda t: calculate_cost(x_data_normalized, y_data, t, b), w))
    b -= (0.01 * differentiate(lambda t: calculate_cost(x_data_normalized, y_data, w, t), b))

print("w : {}, b : {}".format(w, b))


def evaluate(X):
    H = hypothesis(X, w, b)
    correctCount = 0

    for x, y, h in zip(x_data, y_data, H):
        y_predic = h >= 0.5
        print(f"x : {x} , predict : {y_predic}")
        if y == y_predic:
            correctCount += 1

    print("accuracy : %f" % (correctCount/H.size))


evaluate(x_data_normalized)
