from math import cos, sin
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from pylab import rcParams
import csv
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Input


def weierstrass(x, n, step=0.0001):
    a = 0.5
    b = 12.6
    intervalBegin = 0
    intervalEnd = 1
    data = []
    for k in range(int((1 / step) * abs(intervalEnd - intervalBegin))):
        output = 0
        for i in range(n):
            output += pow(a, i) * sin(pow(b, i) * i * k * step)
        data.append(output)

    return np.array(data)


plot_num = 10000
x = np.linspace(0, 1, num=plot_num)
#colorlist = ['r', 'b', 'g']

#for n, color in enumerate(colorlist):
y0 = weierstrass(x=x, n=3, step=1/plot_num)
rcParams['figure.figsize'] = 16, 9
plt.gca().xaxis.grid(True)
plt.gca().yaxis.grid(True)
plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(x * 1/plot_num, ',')))
plt.plot(y0, color='r')

plt.show()