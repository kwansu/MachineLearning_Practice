from NumericalDifferentiation import*
import matplotlib.pyplot as plt

# 90점 이상 A, 80이상 B, 70이상 C, 70 미만 D
x_data = np.array([60, 73, 90, 55, 81, 71, 69, 95, 88, 98, 65]).reshape(11, 1)
y_data = np.array([(0, 0, 0, 1), (0, 0, 1, 0), (1, 0, 0, 0), (0, 0, 0, 1), (0, 1, 0, 0), (0, 0, 1, 0),
                  (0, 0, 0, 1), (1, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1)], dtype=float)


def activateNormalization(z):
    min = np.min(z)
    z = (z - min) / (np.max(z) - min)
    temp = z / np.sum(z, axis=-1).reshape(len(z), 1)
    if np.min(temp) < 0:
        a = 10
    return temp


def hypothesis(x, W, B):
    g = np.dot(x, W) + B
    return activateNormalization(g)


def crossentropy(P):
    return -np.sum(y_data*np.log(P))


W = np.array([(1.1, 1.25, 1.4, 1.6)])  # np.random.random((1, 4))
B = np.array([-1., 0.1, 1., 2.])  # np.random.random(4)
learning_rate = 0.05


def loss(_x, _w, _b): 
    return crossentropy((hypothesis(_x, _w, _b)))


minV = np.min(x_data)
disV = np.max(x_data) - np.min(x_data)

x_data_normalized = (x_data - np.min(x_data)) / \
    (np.max(x_data) - np.min(x_data))

for i in range(10001):
    if i % 1000 == 0:
        # if i % 5000 == 0:
        #     learning_rate /= 2
        print('epoch %d, loss : %f' % (i, loss(x_data_normalized, W, B)))

    # x_set = tuple(i * 0.01 for i in range(0, 100))
    # y_set = np.array([(1, 0, 0, 0) if x*disV+minV >= 90 else ((0, 1, 0, 0) if x*disV+minV >=
    #                  80 else((0, 0, 1, 0) if x*disV+minV >= 70 else (0, 0, 0, 1))) for x in x_set], dtype=float)
    # y_set = numerical_derivative(lambda t: loss(
    #     np.array(x_set).reshape(100, 1), t, B, y_set), W)

    w_set = tuple(i * 0.01 for i in range(0, 100))
    dw_set = []
    _w = np.array([(1.0,1.0,1.0,1.0)])
    for i in w_set:
        _w[0,-1] = i
        dw_set.append(differentiate(lambda t: loss(x_data, t, B), _w)[0,-1])
    
    plt.plot(w_set, tuple(dw_set))
    plt.show()

    W -= (learning_rate *
          differentiate(lambda t: loss(x_data_normalized, t, B), W))
    B -= (learning_rate *
          differentiate(lambda t: loss(x_data_normalized, W, t), B))

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
