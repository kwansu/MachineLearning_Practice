import numpy as np

x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([1, 3, 5, 7, 9])
# x_data = np.array([1,  4,   9,  12, 13])    # 계산을 쉽게하기 위해 numpy를 사용
# y_data = np.array([1, -8, -25, -30, -36])
 

def hypothesis(x, w, b):    # 실제 y값을 예측할 때 쓴다.
    return x*w + b


def calc_loss(x, y, w, b):  # x,y,a,b에 따른 최소제곱법 오차
    return np.sum(np.abs(hypothesis(x, w, b) - y))


def differentiate(f, x):
    dx = 0.0001 * x                  # x에 0.0001을 곱해서 0에 가까운 값을 나타낸다.
    y_plus_dx = f(x+dx)              # f(x+dx)
    y = f(x)                         # f(x)
    gradient = (y_plus_dx - y) / dx  # f(x+dx) - f(x) / dx
    return gradient


def train(x, y, w, b, *, learning_rate = 0.01, epochs = 1000, bins = 10):
    interval = int(epochs / bins)
    for i in range(epochs):
        if i % interval == 0:
            print(f'ephoc : {i}, loss : {calc_loss(x_data, y_data, w, b)}')
        w -= learning_rate * differentiate(lambda v: calc_loss(x, y, v, b), w)
        b -= learning_rate * differentiate(lambda v: calc_loss(x, y, w, v), b)
    print(f'ephoc : {i}, loss : {calc_loss(x_data, y_data, w, b)}')


w = np.random.random(1) # 0~1사이의 값 1개를 만든다.
b = np.random.random(1)

train(x_data, y_data, w, b, learning_rate=0.01, epochs=1000)
print(f"w : {w}, b : {b}")
print(f"x : 10, predict : {hypothesis(10, w, b)}")
