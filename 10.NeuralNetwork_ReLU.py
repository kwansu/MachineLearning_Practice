import numpy as np
import matplotlib.pyplot as plt

class Layer:
    def __init__(self, input_shape, output_shape, *, activation='ReLU'):
        self.output_shape = output_shape
        self.w = np.random.random((input_shape, output_shape))
        self.b = np.random.random((output_shape))
        self.dh_dw = None
        self.do_dh = None
        if activation == 'ReLU':
            # self.backPropagate_activation = lambda g : np.where(g > 0., g, 0.)
            self.activate = self.activate_relu
        elif activation == 'Sigmoid':
            # self.backPropagate_activation = lambda g : self.do_dh * g
            self.activate = self.activate_sigmoid
            
    def reset(self):
        self.w = np.random.random(self.w.shape) - 0.5
        self.b = np.random.random(self.b.shape) - 0.5

    def progress(self, x_input):
        self.dh_dw = x_input.swapaxes(1,2)
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
        temp2 = np.sum(gradient, axis=(0,1))
        self.b -= learning_rate * temp2
        temp = np.matmul(self.dh_dw, gradient)
        temp = np.sum(temp, axis=0)
        gradient = np.matmul(gradient, self.w.T)
        self.w -= learning_rate * temp
        return gradient

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

    def show_gradient_graph(self, x, search_rate, range_count = 200):
        for layer in self.layers:
            for narray in layer:
                iter = np.nditer(narray, flags=['multi_index'], op_flags=['readwrite'])
                while not iter.finished:
                    i = iter.multi_index
                    x_arr = []
                    y_arr = []
                    source = narray[i]
                    plt.axvline(x=source, color='r')
                    stride = search_rate*source
                    narray[i] -= range_count * stride / 2
                    for n in range(200):
                        x_arr.append(narray[i])
                        y_arr.append(np.sum(self.predict(x)))
                        narray[i] += stride
                    narray[i] = source
                    plt.plot(x_arr, y_arr)
                    plt.show()
                    iter.iternext()
                

    def update_layers(self, x, y, learning_rate, ephoc = None):
        c = self.predict(x)
        if ephoc is not None:
            temp = self.calc_binary_cross_entorpy(c, y)
            print(f'ephoc : {ephoc}, loss : {temp}')
            # if self.loss - temp < 0.000001:
            #     if self.stop_count > 2:
            #         self.show_gradient_graph(x, 0.1)
            #         self.stop_count = 0
            #     self.stop_count += 1
            # else:
            #     self.stop_count = 0
        gradient = c-y
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
model.add_layer(1,activation='Sigmoid')


def test(x, y):
    model.reset()
    model.fit(x, y, learning_rate=0.1, epochs=10000)
    model.evaluate(x, y)


# # or
# x_data = np.array([(0, 0), (0, 1), (1, 0), (1, 1)]).reshape(4, 1, 2)
# y_data = np.array([0, 1, 1, 1]).reshape(4, 1, 1)
# test(x_data, y_data)

# # and
# x_data = np.array([(0, 0), (0, 1), (1, 0), (1, 1)]).reshape(4, 1, 2)
# y_data = np.array([0, 0, 0, 1]).reshape(4, 1, 1)
# test(x_data, y_data)

# xor
x_data = np.array([(0, 0), (0, 1), (1, 0), (1, 1)]).reshape(4, 1, 2)
y_data = np.array([0, 1, 1, 0]).reshape(4, 1, 1)
test(x_data, y_data)
