from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

class Layer:
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    def __init__(self, input_count, output_count, *, activation='ReLU', optimizer=None):
        self.output_count = output_count
        self.input_count = input_count
        self.reset()
        self.dh_dw = None
        self.do_dh = None
        self.velocity_rate = 0.8
        self.v = 0.0
        self.m = 0.0
        self.adam_count = 1
        if activation == 'ReLU':
            self.activate = self.activate_relu
        elif activation == 'Sigmoid':
            self.activate = self.activate_sigmoid

        if optimizer is None:
            self.update_variables = self.optimize_gradient_descent
        elif optimizer == 'momentum':
            self.update_variables = self.optimize_momentum
        elif optimizer == 'Adam':
            self.update_variables = self.optimize_Adam


    def print_weight(self):
        return self.w

    def reset(self):
        if self.output_count == 1:
            self.w = np.random.random((self.input_count+1, self.output_count))
        else:
            self.w = np.random.random((self.input_count+1, self.output_count+1))
            for i in range(self.input_count):
                self.w[i, self.output_count] = 0.0
            self.w[self.input_count, self.output_count] = 1.0


    def progress(self, x_input):
        self.dh_dw = x_input.swapaxes(1, 2)
        return self.activate(np.matmul(x_input, self.w))


    def activate_sigmoid(self, z):
        s = 1 / (1+np.exp(-z))
        self.do_dh = s*(1-s)
        return s


    def activate_relu(self, z):
        self.do_dh = np.where(z > 0., 1., 0.)
        return np.where(z > 0., z, 0.)


    def update_backpropagation(self, g, learning_rate):
        g = self.do_dh * g
        return self.calc_backpropagation_and_update(g, learning_rate)


    def calc_backpropagation_and_update(self, gradient, learning_rate):
        temp = np.matmul(self.dh_dw, gradient)
        gradient = np.matmul(gradient, self.w.T)
        gradient[:,:,-1] = 0.0
        self.update_variables(np.sum(temp, axis=0), learning_rate)
        return gradient


    def optimize_momentum(self, gradient, learning_rate):
        self.v = self.velocity_rate * self.v + learning_rate*gradient
        self.w -= self.v


    def optimize_gradient_descent(self, gradient, learning_rate):
        self.w -= learning_rate * gradient


    def optimize_Adam(self, gradient, learning_rate):
        self.m = Layer.beta1*self.m + (1-Layer.beta1)*gradient
        self.v = Layer.beta2*self.v + (1-Layer.beta2)*np.power(gradient,2)
        m_hat = self.m / (1 - np.power(Layer.beta1, self.adam_count))
        v_hat = self.v / (1 - np.power(Layer.beta2, self.adam_count))
        self.w -= learning_rate * m_hat / (np.sqrt(v_hat) + Layer.epsilon)
        self.adam_count += 1


    def get_variable_iter(self, multi_index, step_size, range):
        source = self.w[multi_index]
        for var in range:
            var = self.w[multi_index] = var*step_size
            yield var


class Model:
    def __init__(self):
        self.layers = []
        self.loss = 0.0
        self.stop_count = 0


    def reset(self):
        for layer in self.layers:
            layer.reset()


    def add_layer(self, output_count, input_count=None, *, activation='ReLU', optimizer=None):
        if input_count is None:
            input_count = self.layers[-1].output_count
        self.layers.append(Layer(input_count, output_count, activation=activation, optimizer=optimizer))


    def predict(self, x):
        x = np.insert(x, x.shape[-1], 1., axis=-1)
        for layer in self.layers:
            x = layer.progress(x)
        return x


    def calc_binary_cross_entorpy(self, p, y):
        delta = 0.0000001
        return -np.sum(y*np.log(p+delta) + (1-y)*np.log(1-p+delta))


    def update_layers(self, x, y, learning_rate, ephoc=None):
        c = self.predict(x)
        if ephoc is not None:
            temp = self.calc_binary_cross_entorpy(c, y)
            print(f'ephoc : {ephoc}, loss : {temp}')
            if abs(self.loss - temp) < 0.005:
                if self.stop_count > 2:
                    #self.show_gradient_graph(x, 0.05)
                    self.stop_count = 0
                self.stop_count += 1
            else:
                self.stop_count = 0
            self.loss = temp
        gradient = c-y
        gradient = self.layers[-1].calc_backpropagation_and_update(gradient, learning_rate)
        other_layers = self.layers[:-1]
        for layer in other_layers[::-1]:
            gradient = layer.update_backpropagation(gradient, learning_rate)


    def fit(self, x, y, epochs, learning_rate=0.01, print_count=None):
        if print_count is None:
            print_count = epochs / 10
        for i in range(epochs):
            if i % print_count == 0:
                self.update_layers(x, y, learning_rate, i)
            else:
                self.update_layers(x, y, learning_rate)

        for layer in self.layers:
            layer.print_weight()


    def evaluate(self, x, y):
        p = self.predict(x)
        correct_count = 0

        for xi, yi, pi in zip(x, y, p):
            y_predic = pi >= 0.5
            print(f"x : {xi} , predict : {y_predic}")
            if yi == y_predic:
                correct_count += 1
        print(f"accuracy : {correct_count/p.size}")


    def show_gradient_graph(self, x, y):
        w1_iter = self.layers[0].get_variable_iter((0,1),0.1,range(-100, 100))
        w2_iter = self.layers[0].get_variable_iter((1,0),0.1,range(-100, 100))
        loss_surface = np.zeros((200, 200))
        w1_tile = np.linspace(-1.0, 1.0, 200)
        w1_tile = np.tile(w1_tile, (200, 1))
        w2_tile = np.transpose(w1_tile)
        for i, w1 in enumerate(w1_iter):
            for j, w2 in enumerate(w2_iter):
                loss_surface[i,j] = self.calc_binary_cross_entorpy(self.predict(x),y)

        ax.plot_surface(w1_tile, w2_tile, loss_surface)
        ax.set_zlim(-2, 2)
        
        plt.tight_layout()
        plt.show()


model = Model()
model.add_layer(input_count=2, output_count=2, optimizer='Adam')
model.add_layer(1, activation='Sigmoid', optimizer='Adam')


def test(x, y):
    model.reset()
    #model.show_gradient_graph(x,y)
    model.fit(x, y, learning_rate=0.001, epochs=10000)
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
