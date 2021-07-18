from NumericalDifferentiation import*

# 90점 이상 A, 80이상 b, 70이상 C, 70 미만 D
x_data = np.array([60, 73, 90, 55, 81, 71, 69, 95, 88, 98, 65]).reshape(11, 1)
y_data = np.array([(0, 0, 0, 1), (0, 0, 1, 0), (1, 0, 0, 0), (0, 0, 0, 1), (0, 1, 0, 0), (0, 0, 1, 0),
                  (0, 0, 0, 1), (1, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1)], dtype=float)


def activateSoftmax(z):
    z = np.exp(z)
    return z / np.sum(z, axis=-1).reshape(11, 1)


def hypothesis(x, w, b):
    g = np.dot(x, w) + b
    return activateSoftmax(g)


def crossentropy(p, y):
    return -np.sum(y*np.log(p))


def calculate_loss(x, y, w, b):
    return crossentropy(hypothesis(x, w, b), y)


w = np.random.random((1, 4))
b = np.random.random(4)
learning_rate = 1.0
x_data_normalized = (x_data - np.mean(x_data)) / np.std(x_data)

for i in range(10001):
    if i % 1000 == 0:
        print(f'ephoc : {i}, loss : {calculate_loss(x_data_normalized, y_data, w, b)}')
    w -= learning_rate * differentiate(lambda t: calculate_loss(x_data_normalized, y_data, t, b), w)
    b -= learning_rate * differentiate(lambda t: calculate_loss(x_data_normalized, y_data, w, t), b)

print(f"w : {w}, b : {b}")


def evaluate(x):
    #h = hypothesis(x, w, b)
    H = np.dot(x, w) + b
    count_correct = 0
    onehot = np.zeros_like(y_data[0])

    for i, h in enumerate(H):
        onehot.fill(0.0)
        onehot[np.argmax(h)] = 1.0
        print(f"x : {x_data[i]} , predict : {onehot}")
        if np.array_equal(onehot, y_data[i]):
            count_correct += 1

    print(f"accuracy : {count_correct / len(y_data)}")


evaluate(x_data_normalized)
