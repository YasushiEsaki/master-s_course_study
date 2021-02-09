import numpy as np
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt

start = time.time()



x_data =  np.random.rand(100)
y_data = 0.1 * x_data + 0.3

W = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))



