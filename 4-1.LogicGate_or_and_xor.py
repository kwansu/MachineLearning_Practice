import numpy as np
import numpy as np


def activate_sigmoid(z):
    return 1 / (1+np.exp(-z))


def hypothesis(x, w, b):
    g = np.dot(x, w) + b
    return activate_sigmoid(g)


def binaryCrossentropy(p, y):
    delta = 1e-7
    return -np.sum(y*np.log(p+delta) + (1-y)*np.log(1-p+delta))


def calculate_loss(x, y, w, b):
    return binaryCrossentropy(hypothesis(x, w, b), y)


def differentiate(f, x):
    gradient = np.zeros_like(x)
    x_iter = np.nditer(x, flags=['multi_index'])

    while not x_iter.finished:
        mi = x_iter.multi_index
        source = x[mi]
        dx = 1e-4 * source
        y = f(x)
        x[mi] = source + dx
        y_plus_dx = f(x)
        gradient[mi] = (y_plus_dx - y) / dx
        x[mi] = source
        x_iter.iternext()
    return gradient


def evaluate(X, Y, w, b):
    H = hypothesis(X, w, b)
    correct_count = 0

    for x, y, h in zip(X, Y, H):
        y_predic = h >= 0.5
        print(f"x : {x} , predict : {y_predic}")
        if y == y_predic:
            correct_count += 1

    print("accuracy : %f" % (correct_count/H.size))


def train(x, y, w, b, epoch, learning_rate = 0.01):
    print_count = epoch / 10
    for i in range(epoch):
        if i % print_count == 0:
            print(f'ephoc : {i}, loss : {calculate_loss(x, y, w, b)}')
        w -= (learning_rate * differentiate(lambda t: calculate_loss(x, y, t, b), w))
        b -= (learning_rate * differentiate(lambda t: calculate_loss(x, y, w, t), b))


def test(x, y, epoch = 10000, learning_rate = 0.01):
    w = np.random.random((2, 1))
    b = np.random.random(1)
    train(x, y, w, b, epoch, learning_rate)
    evaluate(x, y, w, b)


# or
x_data = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
y_data = np.array([0, 1, 1, 1]).reshape(4,1)
test(x_data, y_data)

# and
x_data = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
y_data = np.array([0, 0, 0, 1]).reshape(4,1)
test(x_data, y_data)

# xor
x_data = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
y_data = np.array([0, 1, 1, 0]).reshape(4,1)
test(x_data, y_data)