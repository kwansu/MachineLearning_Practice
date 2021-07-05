import numpy as np
import matplotlib.pyplot as plt

x_data = (1,  4,   9,  12, 13)
y_date = (1, -8, -25, -31, -36)

x = [0, 15]
y = [-5, -35]

plt.vlines(1, -2*x_data[0]-5,  y_date[0], color='r')
plt.vlines(4, -2*x_data[1]-5, y_date[1], color='r')
plt.vlines(9, -2*x_data[2]-5, y_date[2], color='r')
plt.vlines(12,-2*x_data[3]-5, y_date[3], color='r')
plt.vlines(13,-2*x_data[4]-5, y_date[4], color='r')

# plt.text(1, -x_data[0]+1,  f'{y_date[0] +2*x_data[0] +5}', horizontalalignment='center')
# plt.text(4, -x_data[1]+1,  f'{y_date[1] +2*x_data[1] +5}', horizontalalignment='center')
# plt.text(9, -x_data[2]+1,  f'{y_date[2] +2*x_data[2] +5}', horizontalalignment='center')
# plt.text(12,-x_data[3]+1, f'{y_date[3]  +2*x_data[3] +5}', horizontalalignment='center')
# plt.text(13,-x_data[4]+1, f'{y_date[4]  +2*x_data[4] +5}', horizontalalignment='center')

plt.plot(x,y,color='g')

plt.scatter(x_data, y_date)
plt.show()