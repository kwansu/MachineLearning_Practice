from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import random

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# x_data = np.array((1, 2, 3, 4, 5))
# y_data = np.array((4, 6, 8, 10, 12))
x_data = np.array((1, 4, 9, 12, 13))
y_data = np.array((1, -8, -25, -31, -36))


def hypothesis(x, w, b):  # 실제 y값을 예측할 때 쓴다.
    return x*w + b


def calc_loss(x, y, w, b):
    return sum((y - (x*w + b))**2)


def differentiate(f, x):
    dx = 0.0001 * x                  # x에 0.0001을 곱해서 0에 가까운 값을 나타낸다.
    y_plus_dx = f(x+dx)              # f(x+dx)
    y = f(x)                         # f(x)
    gradient = (y_plus_dx - y) / dx  # f(x+dx) - f(x) / dx
    return gradient


def show_gradient_graph(x, y, *, num_coordinates = 100):
    w_tile = np.linspace(10.0, -10.0, num_coordinates)
    loss_surface = np.zeros((num_coordinates, num_coordinates))
    for row, w in enumerate(w_tile):
        for col, b in enumerate(w_tile):
            loss_surface[row,col] = calc_loss(x, y, w, b)
    w_tile = np.tile(w_tile, (num_coordinates, 1))
    b_tile = np.transpose(w_tile)
    ax.plot_surface(w_tile, b_tile, loss_surface)
    #ax.set_zlim(0, 1)
    
    plt.tight_layout()
    plt.show()


w = random.random()
b = random.random()
learning_rate = 0.001

show_gradient_graph(x_data, y_data)

for i in range(10001):
    if i % 1000 == 0:
        print(f'ephoc : {i}, loss : {calc_loss(x_data, y_data, w, b)}')
    w -= learning_rate * differentiate(lambda v: calc_loss(x_data, y_data, v, b), w)
    b -= learning_rate * differentiate(lambda v: calc_loss(x_data, y_data, w, v), b)

print(f"w : {w}, b : {b}")
print(f"x : 4, predict : {hypothesis(4, w, b)}")
