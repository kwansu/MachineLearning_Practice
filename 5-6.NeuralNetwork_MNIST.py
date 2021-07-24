import numpy as np
import pandas as pd


class ActivationFunc:
    def forward(self, h_input):
        self.da_dh = None

    def backward(self, gradient):
        return self.da_dh * gradient


class ActivationSigmoid(ActivationFunc):
    def forward(self, h_input):
        a = 1 / (1+np.exp(-h_input))
        self.da_dh = a
        return a


class ActivationReLU(ActivationFunc):
    def forward(self, h_input):
        self.da_dh = np.where(h_input > 0., 1., 0.)
        return self.da_dh * h_input


class ActivationSoftmax(ActivationFunc):
    def forward(self, h_input):
        h_input = np.exp(h_input)
        return h_input / np.expand_dims(np.sum(h_input, axis=-1), axis=-1)



class OptimizerSGD:
    def __call__(self, w, gradient, learning_rate):
        w -= learning_rate * np.mean(gradient, axis=0)


class Layer:
    def __init__(self, input_shape, output_shape, activation: ActivationFunc):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation

    def reset(self, is_end_layer=False):
        if is_end_layer:
            self.w = np.random.random((self.input_shape+1, self.output_shape))
        else:
            self.w = np.random.random((self.input_shape+1, self.output_shape+1))
            self.w[:, self.output_shape] = 0.0
            self.w[self.input_shape, self.output_shape] = 1.0

    def forward(self, x_input):
        self.dh_dw = x_input.swapaxes(-2, -1)
        h = np.matmul(x_input, self.w)
        return self.activation.forward(h)

    def backword(self, gradient, learning_rate):
        gradient = self.activation.backward(gradient)
        self.backword_update(gradient, learning_rate)

    def backword_update(self, gradient, learning_rate):
        #self.optimizer(self.w, self.dh_dw * gradient, learning_rate)
        t = self.dh_dw * gradient
        temp2 = np.mean(t, axis=0)
        gradient = np.matmul(gradient, self.w.T)
        gradient[:,:,-1] = 0.0
        self.w -= learning_rate * temp2
        return gradient
        


class Model:
    def __init__(self) -> None:
        self.layers = []

    def add_layer(self, output_shape, input_shape=None, *, activation_func='ReLU'):
        if activation_func == 'ReLU':
            activation = ActivationReLU()
        elif activation_func == 'Sigmoid':
            activation = ActivationSigmoid()
        elif activation_func == 'Softmax':
            activation = ActivationSoftmax()

        if input_shape is None:
            input_shape = self.layers[-1].output_shape

        self.layers.append(Layer(input_shape, output_shape, activation))

    def compile(self):
        for layer in self.layers[:-1]:
            layer.reset()
        self.layers[-1].reset(True)

    def predict(self, x):
        x = np.insert(x, x.shape[-1], 1., axis=-1)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def calc_cross_entropy(self, p, y):
        delta = 0.0000001
        return -np.sum(y*np.log(p+delta))

    def update_layers(self, x, y, learning_rate, ephoc=None):
        c = self.predict(x)
        if ephoc is not None:
            temp = self.calc_cross_entropy(c, y) / len(y)
            self.loss += temp
        gradient = c-y # dL/dh
        gradient = self.layers[-1].backword_update(gradient, learning_rate)
        other_layers = self.layers[:-1]
        for layer in other_layers[::-1]:
            gradient = layer.backword_update(gradient, learning_rate)

    def fit(self, x, y, epochs, learning_rate=0.01, batch_size = None, print_count=None):
        if print_count is None:
            print_count = epochs / 10
        if batch_size is None:
            batch_size = len(y)
        bins, other = divmod(len(y), batch_size)

        for i in range(epochs):
            s = 0
            self.loss = 0.0
            i = i if i % print_count == 0 else None
            for b in range(bins):
                e = s+batch_size
                self.update_layers(x[s:e], y[s:e], learning_rate, i)
                s = e
            if other != 0:
                self.update_layers(x[s:], y[s:], learning_rate, i)
            if i%print_count == 0:
                print(f'ephoc : {i}, loss : {self.loss}')

    def evaluate(self, x, y):
        p = self.predict(x)
        correct_count = 0

        for yi, pi in zip(y, p):
            if yi[0, np.argmax(pi)] == 1.0:
                correct_count += 1

        print(f"accuracy : {correct_count/len(y)}")


data = pd.read_csv("data/mnist_train.csv")
x_train = data.drop(['label'], axis=1).values / 255.0
x_train = x_train.reshape([x_train.shape[0], 1, x_train.shape[-1]])
y_data = data['label']
y_data = pd.get_dummies(y_data)
y_train = y_data.values
y_train = y_train.reshape([y_train.shape[0], 1, y_train.shape[-1]])

test_count = int(len(y_train)/10)
x_test = x_train[-test_count:]
x_train = x_train[:-test_count]
y_test = y_train[-test_count:]
y_train = y_train[:-test_count]

model = Model()
model.add_layer(input_shape=784, output_shape=512)
model.add_layer(output_shape=512)
model.add_layer(output_shape=256)
model.add_layer(output_shape=128)
model.add_layer(output_shape=10, activation_func='Softmax')
model.compile()
model.fit(x_train, y_train, epochs=20, batch_size=128, learning_rate=0.001)


# data = pd.read_csv("data/mnist_test.csv")
# x_test = data.drop(['label'], axis=1).values / 255.0
# x_test = x_test.reshape([x_test.shape[0], 1, x_test.shape[-1]])
# y_data = data['label']
# y_data = pd.get_dummies(y_data)
# y_test = y_data.values
# y_test = y_test.reshape([y_test.shape[0], 1, y_test.shape[-1]])

model.evaluate(x_test, y_test)