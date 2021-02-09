from math import cos
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import csv
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Input
from pylab import rcParams

def model2_weights_initializer():
    weights_initializer2 = [[[]]]

    for _ in range(4):
        weights_initializer2[0][0].append(np.random.randn())
    


    weights_initializer2.append([])

    for j in range(4):
        weights_initializer2[1].append(np.random.randn())


    weights_initializer2.append([[]])


    for j in range(4):
        for _ in range(4):
            weights_initializer2[2][j].append(np.random.randn())
    
        if j < 3:
            weights_initializer2[2].append([])


    weights_initializer2.append([])

    for j in range(4):
        weights_initializer2[3].append(np.random.randn())


    weights_initializer2.append([[]])

    for j in range(4):
        for _ in range(4):
            weights_initializer2[4][j].append(np.random.randn())
    
        if j < 3:
            weights_initializer2[4].append([])


    weights_initializer2.append([])

    for j in range(4):
        weights_initializer2[5].append(np.random.randn())


    weights_initializer2.append([[]])

    for j in range(4):
        for _ in range(4):
            weights_initializer2[6][j].append(np.random.randn())
    
        if j < 3:
            weights_initializer2[6].append([])


    weights_initializer2.append([])

    for j in range(4):
        weights_initializer2[7].append(np.random.randn())


    weights_initializer2.append([[]])

    for j in range(4):
        for _ in range(4):
            weights_initializer2[8][j].append(np.random.randn())
    
        if j < 3:
            weights_initializer2[8].append([])


    weights_initializer2.append([])

    for j in range(4):
        weights_initializer2[9].append(np.random.randn())


    weights_initializer2.append([[]])

    for j in range(4):
        weights_initializer2[10][j].append(np.random.randn())
        if j < 3:
            weights_initializer2[10].append([])


    weights_initializer2.append([])

    weights_initializer2[11].append(np.random.randn())

    
    return weights_initializer2


def model3_weights_initializer():
    weights_initializer3 = [[[]]]

    for _ in range(20):
        weights_initializer3[0][0].append(np.random.randn())


    weights_initializer3.append([])

    for j in range(20):
        weights_initializer3[1].append(np.random.randn())


    weights_initializer3.append([[]])


    for j in range(20):
        weights_initializer3[2][j].append(np.random.randn())
        if j < 19:
            weights_initializer3[2].append([])


    weights_initializer3.append([])

    weights_initializer3[3].append(np.random.randn())


    return weights_initializer3


def weierstrass(x, step=0.0001):
    a = 0.5
    b = 15
    n = 3
    intervalBegin = 0
    intervalEnd = 1
    data = []
    for k in range(int((1 / step) * abs(intervalEnd - intervalBegin))):
        output = 0
        for i in range(n):
            output += pow(a, i) * cos(pow(b, i) * i * k * step)
        data.append(output)

    return np.array(data)





plot_num = 10000
x0 = np.linspace(0, 1, num=plot_num)

y0 = weierstrass(x=x0, step=1/plot_num)


x1 = np.reshape(x0, newshape=(plot_num, 1))
y1 = np.reshape(y0, newshape=(plot_num, 1))



inputs = Input(shape=(1,))
x = Dense(units=4, activation='relu')(inputs)
x = Dense(units=4, activation='relu')(x)
x = Dense(units=4, activation='relu')(x)
x = Dense(units=4, activation='relu')(x)
x = Dense(units=4, activation='relu')(x)
predictions = Dense(units=1)(x)

model2 = Model(inputs=inputs, outputs=predictions)



inputs = Input(shape=(1,))
x = Dense(units=20, activation='relu')(inputs)
predictions = Dense(units=1)(x)

model3 = Model(inputs=inputs, outputs=predictions)


count_model2 = [0]*10000
count_model3 = [0]*10000
epsilon = np.linspace(0, 5, 10000)
for i in range(10000):
    model2.set_weights(model2_weights_initializer())
    model2.save_weights(filepath='weights_initializer2_%03.f'%(i+1)+'.hdf5')

    if i == 0:
        y2 = model2.predict_on_batch(x1)
        y2 = (y2-np.mean(y2, axis=0))/np.std(y2, axis=0)
        err2 = np.mean(np.square(y1-y2), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err2 <= k:
                count_model2[j] += 1
        with open('err2.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([y2, err2])
    else:
        y = model2.predict_on_batch(x1)
        y = (y-np.mean(y, axis=0))/np.std(y, axis=0)
        err = np.mean(np.square(y1-y), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err <= k:
                count_model2[j] += 1
        with open('err2.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([y, err])


        if err < err2:
            err2 = err
            y2 = y
    

    model3.set_weights(model3_weights_initializer())
    model3.save_weights(filepath='weights_initializer3_%03.f'%(i+1)+'.hdf5')
    
    if i == 0:
        y3 = model3.predict_on_batch(x1)
        y3 = (y3-np.mean(y3, axis=0))/np.std(y3, axis=0)
        err3 = np.mean(np.square(y1-y3), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err3 <= k:
                count_model3[j] += 1
        with open('err3.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([y3, err3])
    else:
        y = model3.predict_on_batch(x1)
        y = (y-np.mean(y, axis=0))/np.std(y, axis=0)
        err = np.mean(np.square(y1-y), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err <= k:
                count_model3[j] += 1
        with open('err3.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([y, err])


        if err < err3:
            err3 = err
            y3 = y
    
    

    print(i+1)

y2 = np.reshape(y2, newshape=plot_num)
y3 = np.reshape(y3, newshape=plot_num)

rcParams['figure.figsize'] = 16, 9
plt.gca().xaxis.grid(True)
plt.gca().yaxis.grid(True)
plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x0, p: format(x0 * 1/plot_num, ',')))
plt.plot(y0, color='red', label='weierstrass')
plt.plot(y2, color='blue', label='network 1')
plt.plot(y3, color='green', label='network 2')
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.savefig("weierstrass_uniform_func.png")
plt.clf()

count_model2 = np.array(count_model2)
count_model3 = np.array(count_model3)

rate2 = count_model2/10000
rate3 = count_model3/10000

with open('rate.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(rate2)
    writer.writerow(rate3)


plt.plot(epsilon, rate2, color='blue', label='network 1')
plt.plot(epsilon, rate3, color='green', label='network 2')
plt.legend()
plt.xlabel("epsilon")
plt.ylabel("R(epsilon)")
plt.savefig("weierstrass_uniform_rate.png")
plt.clf()