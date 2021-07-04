import numpy as np

x_data = np.array((1, 2, 3, 4, 5))
y_data = np.array((4, 6, 8, 10, 12))

# x_data = np.array((1, 4, 9, 12,13))
# y_data = np.array((-15, -20, -24, -28, -30))

def hypothesis(x, w, b): # 실제 y값을 예측할 때 쓴다.
    return x*w + b

def calculate_loss(x, y, w, b):
    return sum((y - (x*w + b))**2)


def differentiate_numerical(expression, x):
    gradient = np.zeros_like(x)
    x_iter = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    for var in x_iter:
        limited_distance = 1e-4 * var
        y = expression(x)
        var += limited_distance
        y_plus_dx = expression(x)
        var -= limited_distance
        gradient[x_iter.multi_index] += (y_plus_dx - y) / (limited_distance)
        
    return gradient


w = np.random.random(1)
b = np.random.random(1)

for i in range(10001):
    if i % 100 == 0:
        print(f'ephoc : {i}, loss : {calculate_loss(x_data, y_data, w, b)}')
    w -= 0.001 * differentiate_numerical(lambda t: calculate_loss(x_data, y_data, t, b), w)
    b -= 0.001 * differentiate_numerical(lambda t: calculate_loss(x_data, y_data, w, t), b)

print(f"w : {w}, b : {b}")
print(f"x : 4, predict : {hypothesis(4, w, b)}")
