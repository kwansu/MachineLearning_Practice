from NumericalDifferentiation import*

# 90점 이상 A, 80이상 B, 70이상 C, 70 미만 D
x_data = np.array([60, 73, 90, 55, 81, 71, 69, 95, 88, 98, 65]).reshape(11, 1)
y_data = np.array([(0, 0, 0, 1), (0, 0, 1, 0), (1, 0, 0, 0), (0, 0, 0, 1), (0, 1, 0, 0), (0, 0, 1, 0),
                  (0, 0, 0, 1), (1, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1)], dtype=float)


def activateSoftmax(z):
    z = np.exp(z)
    return z / np.sum(z, axis=-1).reshape(11, 1)


def hypothesis(x, W, B):
    g = np.dot(x, W) + B
    return activateSoftmax(g)


def crossentropy(P):
    return -np.sum(y_data*np.log(P))


W = np.random.random((1, 4))
B = np.random.random(4)
learning_rate = 0.01
cost = lambda _x, _w, _b: crossentropy((hypothesis(_x, _w, _b)))
x_data_normalized = (x_data - np.mean(x_data)) / np.std(x_data)

for i in range(10001):
    if i % 1000 == 0:
        print('epoch %d, cost : %f' % (i, cost(x_data_normalized, W, B)))
    W -= (learning_rate * differentiate(lambda t: cost(x_data_normalized, t, B), W))
    B -= (learning_rate * differentiate(lambda t: cost(x_data_normalized, W, t), B))

print("W : {}, B : {}".format(W, B))


def predict(x):
    #Y = hypothesis(x, W, B)
    Y = np.dot(x, W) + B
    correctCount = 0
    onehot = np.zeros_like(Y[0])

    for i in range(len(Y)):
        onehot.fill(0.0)
        onehot[np.argmax(Y[i])] = 1.0
        print("x : {} , predict : {}".format(x[i], onehot))
        if np.array_equal(onehot, y_data[i]):
            correctCount += 1

    print("accuracy : %f" % (correctCount / len(y_data)))


predict(x_data_normalized)
