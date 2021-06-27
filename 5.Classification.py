from NumericalDifferentiation import*

x_data = np.array([60, 73, 90, 55, 81, 70, 69, 95, 88, 83, 65]).reshape(11, 1)
y_data = np.array([((0, 0, 0, 1), (0, 0, 1, 0), (1, 0, 0, 0), (0, 0, 0, 1), (0, 1, 0, 0),
                  (0, 0, 1, 0), (0, 0, 0, 1), (1, 0, 0, 0), (0, 1, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1))])


def softmax(z):
    z = np.exp(z)
    temp = np.sum(z,axis=-1).reshape(11, 1)
    z = z / temp
    return z


def hypothesis(x, W, B):
    temp = np.dot(x, W)
    g = temp + B
    return softmax(g)


def crossentropy(P):
    return -np.sum(y_data*np.log(P))


W = np.random.random((1, 4))
B = np.random.random(4)
def cost(_x, _w, _b): return crossentropy((hypothesis(_x, _w, _b)))


for i in range(1001):
    if i % 100 == 0:
        print('epoch %d, cost : %f' % (i, cost(x_data, W, B)))
    W -= (0.01 * numerical_derivative(lambda t: cost(x_data, t, B), W))
    B -= (0.01 * numerical_derivative(lambda t: cost(x_data, W, t), B))

print("W : {}, B : {}".format(W, B))


def predict(x):
    Y = hypothesis(x, W, B)
    # iterator = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    correctCount = 0
    onehot = np.zeros_like(Y[0])

    # while not iterator.finished:
    for i in range(len(Y)):
        # iterIndex = iterator.multi_index
        onehot.fill(0.0)
        onehot[np.argmax(Y[i])] = 1.0
        print("x : {} , predict : {}".format(x[i], onehot))
        if onehot == y_data[i]:
            correctCount += 1
        # iterator.iternext()

    print("accuracy : %f" % (correctCount / y_data.size))


predict(x_data)
