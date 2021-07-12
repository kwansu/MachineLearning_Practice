import numpy as np
import pandas as pd


class Layer:
    mr = 0.9
    vr = 0.999
    epsilon = 1e-8
    
    def __init__(self, input_shape, output_shape, *, activation='ReLU', is_output=False, optimizer=None):
        self.output_shape = output_shape
        self.w = np.random.random((input_shape, output_shape))
        self.b = np.random.random((output_shape))
        self.dh_dw = None
        self.do_dh = None
        self.v = 0.0
        self.m = 0.0
        if activation == 'ReLU':
            self.activate = self.activate_relu
        elif activation == 'Sigmoid':
            self.activate = self.activate_sigmoid
        elif activation == 'softmax':
            self.activate = self.activate_softmax

        if optimizer is None:
            self.update_variables = self.optimize_gradient_descent
        elif optimizer == 'momentum':
            self.update_variables = self.optimize_momentum
        elif optimizer == 'Adam':
            self.update_variables = self.optimize_Adam


    def reset(self):
        self.w = np.random.random(self.w.shape) - 0.5
        self.b = np.random.random(self.b.shape) - 0.5

    def progress(self, x_input):
        self.dh_dw = x_input.swapaxes(1, 2)
        return self.activate(np.matmul(x_input, self.w) + self.b)

    def activate_sigmoid(self, z):
        s = 1 / (1+np.exp(-z))
        self.do_dh = s*(1-s)
        return s

    def activate_relu(self, z):
        self.do_dh = np.where(z > 0., 1., 0.)
        return np.where(z > 0., z, 0.)

    def update_backPropagation(self, g, learning_rate):
        # g = self.backPropagate_activation(g)
        g = self.do_dh * g
        return self.update_layer(g, learning_rate)

    def update_layer(self, gradient, learning_rate):
        temp2 = np.sum(gradient, axis=(0, 1))
        self.b -= learning_rate * temp2
        temp = np.matmul(self.dh_dw, gradient)
        temp = np.sum(temp, axis=0)
        gradient = np.matmul(gradient, self.w.T)
        self.w -= learning_rate * temp
        return gradient

    def optimize_Adam(self, gradient, learning_rate):
        assert(self.b == 0)
        self.m = Layer.mr*self.v + (1-Layer.mr)*gradient
        m_hat = self.m / (1-Layer.mr**2)
        self.v = Layer.vr*self.v + (1-Layer.vr)*gradient**2
        v_hat = self.v / (1-Layer.vr**2)
        self.w -= learning_rate * m_hat / np.sqrt(v_hat+Layer.epsilon)

    def __iter__(self):
        yield self.w
        yield self.b


class Model:
    def __init__(self):
        self.layers = []
        self.loss = 0.0
        self.stop_count = 0

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def add_layer(self, output_shape, input_shape=None, *, activation='ReLU'):
        if input_shape is None:
            input_shape = self.layers[-1].output_shape
        self.layers.append(Layer(input_shape, output_shape, activation=activation))

    def predict(self, x):
        for layer in self.layers:
            x = layer.progress(x)
        return x

    def calc_binary_cross_entorpy(self, p, y):
        delta = 0.0000001
        return -np.sum(y*np.log(p+delta) + (1-y)*np.log(1-p+delta))

    def update_layers(self, x, y, learning_rate, ephoc=None):
        c = self.predict(x)
        if ephoc is not None:
            temp = self.calc_cross_entropy(c, y)
            print(f'ephoc : {ephoc}, loss : {temp}')
        gradient = c-y
        gradient = self.layers[-1].calc_backpropagation_and_update(gradient, learning_rate)
        other_layers = self.layers[:-1]
        for layer in other_layers[::-1]:
            gradient = layer.update_backpropagation(gradient, learning_rate)

    def fit(self, x, y, epochs, learning_rate=0.01, batch_size = None, print_count=None):
        if print_count is None:
            print_count = epochs / 10
        if batch_size is None:
            batch_size = len(y)
        bins, other = divmod(len(y), batch_size)

        for i in range(epochs):
            s = 0
            for b in range(bins):
                e = s+batch_size
                if i % print_count == 0 and b == 0:
                    self.update_layers(x[s:e], y[s:e], learning_rate, i)
                else:
                    self.update_layers(x[s:e], y[s:e], learning_rate)
                s = e
            if other != 0:
                self.update_layers(x[s:], y[s:], learning_rate)

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
model.add_layer(1, activation='Sigmoid')


def test(x, y):
    model.reset()
    model.fit(x, y, learning_rate=0.01, epochs=10000)
    model.evaluate(x, y)


data = pd.read_csv("data/mnist_train.csv")
x_train = data.drop(['label'], axis=1).values / 255.0
x_train = x_train.reshape([x_train.shape[0], 1, x_train.shape[-1]])
y_data = data['label']
y_data = pd.get_dummies(y_data)
y_train = y_data.values
y_train = y_train.reshape([y_train.shape[0], 1, y_train.shape[-1]])

model = Model()
model.add_layer(input_count=784, output_count=196, optimizer='Adam')
model.add_layer(output_count=196, optimizer='Adam')
model.add_layer(output_count=49, optimizer='Adam')
model.add_layer(output_count=10, is_output = True, activation='softmax', optimizer='Adam')
model.fit(x_train, y_train, epochs=10, batch_size=128, learning_rate=0.001)

model.evaluate(x_train, y_train)
