import numpy as np
import matplotlib.pyplot as plt

x_data = np.array((1,  4,   9,  12, 13))
y_date = np.array((1, -8, -25, -30, -36))

x = [0, 15]
y = [-4, -34]
_y = -2*x_data-4

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax2.vlines(1, _y[0], y_date[0], color='r')
ax2.vlines(4, _y[1], y_date[1], color='r')
ax2.vlines(9, _y[2], y_date[2], color='r')
ax2.vlines(12,_y[3], y_date[3], color='r')
ax2.vlines(13,_y[4], y_date[4], color='r')

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