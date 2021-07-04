import numpy as np
import matplotlib.pyplot as plt

x_data = (  1,   4,   9,  12, 13)
y_date = (-15, -20, -24, -28, -30)

x = [0, 15]
y = [0, -45]

plt.vlines(1, -3, -15, color='r')
plt.vlines(4, -12, -20, color='r')
plt.vlines(9, -27, -24, color='r')
plt.vlines(12, -36, -28, color='r')
plt.vlines(13, -39, -30, color='r')

plt.text(1, -18, '12',horizontalalignment='center')
plt.text(4, -23, '8',horizontalalignment='center')
plt.text(9, -21, '-3',horizontalalignment='center')
plt.text(12, -25, '-8',horizontalalignment='center')
plt.text(13, -27, '-9',horizontalalignment='center')

plt.plot(x,y,color='g')

plt.scatter(x_data, y_date)
plt.show()