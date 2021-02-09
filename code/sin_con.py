import numpy as np
import matplotlib.pyplot as plt


x0 = np.linspace(0, 1, num=10000)

y0 = np.sin(4*np.pi*x0) + np.sin(8*np.pi*x0)


plt.plot(x0, y0, color='red', label='sin(4*pi*x) + sin(8*pi*x)')
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.savefig("sin_cos.png")
plt.clf()