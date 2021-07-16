from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

# x_data = np.array([1, 2, 3, 4])
# y_data = np.array([-1, 5, 15, 29])
x_data = np.linspace(0., 4, 100)
y_data = np.array([2*x**2 - 3 for x in x_data])

def hypothesis(x, W, b):
    return W[0] * x**2 + W[1] * x + b


def calc_loss(x, y, W, b):  # mean square error
    return np.sum((hypothesis(x, W, b) - y)**2)/len(y)


def differentiate(f, x):
    gradient = np.zeros_like(x)
    x_iter = np.nditer(x, flags=['multi_index'])

    while not x_iter.finished:
        mi = x_iter.multi_index
        source = x[mi]
        dx = 1e-4 * source
        x[mi] += dx
        f_plus_dx = f(x)
        x[mi] = source
        gradient[mi] = (f_plus_dx - f(x)) / dx
        x_iter.iternext()
    return gradient


def train(x, y, W, b, *, learning_rate = 0.01, epochs = 1000, bins = 10):
    interval = int(epochs / bins)
    for i in range(epochs):
        if i % interval == 0:
            print(f'ephoc : {i}, loss : {calc_loss(x, y, W, b)}')
        W -= learning_rate * differentiate(lambda v: calc_loss(x, y, v, b), W)
        b -= learning_rate * differentiate(lambda v: calc_loss(x, y, W, v), b)
    print(f'ephoc : {i}, loss : {calc_loss(x, y, W, b)}')


def show_gradient_graph(x, y, *, bins = 100):
    _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    w_tile = np.linspace(0.0, 4.0, bins)
    b_tile = np.linspace(-6.0, 4.0, bins)
    loss_surface = np.zeros((bins, bins))
    for row, w in enumerate(w_tile):
        for col, b in enumerate(b_tile):
            loss_surface[row,col] = calc_loss(x, y, w, b) #+ 0.3*calc_loss(x,y,w,-2-b)
    w_tile = np.tile(w_tile, (bins, 1))
    b_tile = np.tile(b_tile, (bins, 1))
    b_tile = np.transpose(b_tile)
    ax.plot_surface(w_tile, b_tile, loss_surface, cmap='viridis')
    plt.tight_layout()
    plt.show()  

    # a_tile = np.linspace(-12., 15., bins)
    # b = 1.0
    # error_tile = np.fromiter((calc_loss(x,y,a,b) for a in a_tile),float)
    # plt.plot(a_tile, error_tile)

    # a = 10.0
    # e = calc_loss(x,y,a,b)
    # plt.plot(a, e,'ro')

    # for i in range(5):
    #     a_before = a
    #     e_before = e
    #     temp = differentiate(lambda v: calc_loss(x,y,v,b), a)
    #     a = a - 0.015 * temp
    #     e = calc_loss(x,y,a,b)
    #     plt.plot(a, e,'ro')
    #     plt.plot((a_before, a_before-2.), (e_before, e_before-2.*temp), 'g')
    plt.tight_layout()
    plt.show()  


W = np.random.random(2)
b = np.random.random(1)

#show_gradient_graph(x_data, y_data)
train(x_data, y_data, W, b, learning_rate=0.001, epochs=50000)
print(f"w : {W}, b : {b}")
