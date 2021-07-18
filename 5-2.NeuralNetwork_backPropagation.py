import numpy as np


class Layer:
    def __init__(self, input_shape, output_shape):
        self.output_shape = output_shape
        self.w = np.random.random((input_shape, output_shape))
        self.b = np.random.random((output_shape))
        self.dh_dw = None
        self.ds_dh = None

    def reset(self):
        self.w = np.random.random(self.w.shape)
        self.b = np.random.random(self.b.shape)

    def progress(self, x_input):
        self.dh_dw = x_input.swapaxes(1,2)
        return self.activate_sigmoid(np.matmul(x_input, self.w) + self.b)

    def activate_sigmoid(self, z):
        s = 1 / (1+np.exp(-z))
        self.ds_dh = s*(1-s)
        return s

    def update_backPropagation(self, g, learning_rate):
        g = self.ds_dh * g
        return self.update_layer(g, learning_rate)

    def update_layer(self, gradient, learning_rate):
        temp2 = np.sum(gradient, axis=(0,1))
        self.b -= learning_rate * temp2
        temp = np.matmul(self.dh_dw, gradient)
        temp = np.sum(temp, axis=0)
        gradient = np.matmul(gradient, self.w.T)
        self.w -= learning_rate * temp
        return gradient


class Model:
    def __init__(self):
        self.layers = []

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def add_layer(self, output_shape, input_shape=None):
        if input_shape is None:
            input_shape = self.layers[-1].output_shape
        self.layers.append(Layer(input_shape, output_shape))

    def predict(self, x):
        for layer in self.layers:
            x = layer.progress(x)
        return x

    def calc_binary_cross_entorpy(self, p, y):
        delta = 0.0000001
        return -np.sum(y*np.log(p+delta) + (1-y)*np.log(1-p+delta))

    def update_layers(self, x, y, learning_rate, ephoc = None):
        s = self.predict(x)
        if ephoc is not None:
            print(f'ephoc : {ephoc}, loss : {self.calc_binary_cross_entorpy(s, y)}')

        gradient = s-y
        gradient = self.layers[-1].update_layer(gradient, learning_rate)
        other_layers = self.layers[:-1]
        for layer in other_layers[::-1]:
            gradient = layer.update_backPropagation(gradient, learning_rate)

    def fit(self, x, y, epochs, learning_rate=0.01, print_count=None):
        if print_count is None:
            print_count = epochs / 10
        for i in range(epochs):
            if i%print_count == 0:
                self.update_layers(x, y, learning_rate,i)
            else:
                self.update_layers(x, y, learning_rate)

    def evaluate(self, x, y):
        p = self.predict(x)
        correct_count = 0

        for xi, yi, pi in zip(x, y, p):
            y_predic = pi >= 0.5
            print(f"x : {xi} , predict : {y_predic}")
            if yi == y_predic:
                correct_count += 1

        print(f"accuracy : {correct_count/p.size}")


model = Model()
model.add_layer(input_shape=2, output_shape=3)
model.add_layer(1)


def test(x, y):
    model.reset()
    model.fit(x, y, learning_rate=0.1, epochs=10000)
    model.evaluate(x, y)


# or
x_data = np.array([(0, 0), (0, 1), (1, 0), (1, 1)]).reshape(4, 1, 2)
y_data = np.array([0, 1, 1, 1]).reshape(4, 1, 1)
test(x_data, y_data)

# and
x_data = np.array([(0, 0), (0, 1), (1, 0), (1, 1)]).reshape(4, 1, 2)
y_data = np.array([0, 0, 0, 1]).reshape(4, 1, 1)
test(x_data, y_data)

# xor
x_data = np.array([(0, 0), (0, 1), (1, 0), (1, 1)]).reshape(4, 1, 2)
y_data = np.array([0, 1, 1, 0]).reshape(4, 1, 1)
test(x_data, y_data)
