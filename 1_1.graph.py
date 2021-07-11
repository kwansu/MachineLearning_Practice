import numpy as np
import matplotlib.pyplot as plt
import random

x_data = np.array([ 1, 2,  3, 4])
y_data = np.array([-1, 5, 15, 29])
x = np.linspace(0, 5, 100)

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.set_xlim(0, 5)
ax1.set_ylim(-5, 35)
ax1.plot(x_data, y_data, 'ro')

ax2.set_xlim(0, 5)
ax2.set_ylim(-5, 35)
ax2.plot(x_data, y_data, 'ro')
ax2.plot(x, 2*x**2-3)
plt.tight_layout()
plt.show()


x = [1, 5, 10]
y = [0, 5, 11]

plt.plot(x, y, 'ro')
plt.plot(x, x)
plt.tight_layout()
plt.show()

x.append(7)
y.append(11)

plt.plot(x, y, 'ro')
plt.plot(x, x)
plt.tight_layout()
plt.show()


x = [11, 15, 21, 23, 25, 26, 28, 30, 34]
y = [10, 20, 30, 40, 50]
z = [10000 * (i - 2 + 6*random.random()) for i in x]

plt.ylim(0, 400000)
plt.xlim(9.5, 38.)
plt.tight_layout()
plt.plot(x, y, 'ro')
plt.show()

plt.ylim(0, 400000)
plt.xlim(9.5, 38.)
plt.tight_layout()
plt.plot(x, y, 'ro')
plt.plot(x, [10000*(1+i) for i in x])
plt.show()

plt.ylim(0, 400000)
plt.xlim(9.5, 38.)
plt.tight_layout()
plt.plot(x, y, 'ro')
plt.plot(x, [10000*(1+i) for i in x])
plt.plot((18, 18), (0, 190000), 'g--')
plt.plot((9, 18), (190000, 190000), 'y--')
plt.plot(18, 190000, 'go')
plt.show()


x_data = np.array((1,  4,   9,  12, 13))
y_date = np.array((1, -8, -25, -30, -36))

x = [0, 15]
y = [-4, -34]
_y = -2*x_data-4

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax2.vlines(1, _y[0], y_date[0], color='r')
ax2.vlines(4, _y[1], y_date[1], color='r')
ax2.vlines(9, _y[2], y_date[2], color='r')
ax2.vlines(12, _y[3], y_date[3], color='r')
ax2.vlines(13, _y[4], y_date[4], color='r')

ax2.text(1 +1, y_date[0], f'{y_date[0] - _y[0]}', horizontalalignment='center')
ax2.text(4 +1, y_date[1], f'{y_date[1] - _y[1]}', horizontalalignment='center')
ax2.text(9 -1, y_date[2], f'{y_date[2] - _y[2]}', horizontalalignment='center')
ax2.text(12-1 ,y_date[3], f'{y_date[3] - _y[3]}', horizontalalignment='center')
ax2.text(13-1 ,y_date[4], f'{y_date[4] - _y[4]}', horizontalalignment='center')

ax1.plot(x,y,color='g')
ax2.plot(x,y,color='g')

ax1.scatter(x_data, y_date)
ax2.scatter(x_data, y_date)
plt.show()
