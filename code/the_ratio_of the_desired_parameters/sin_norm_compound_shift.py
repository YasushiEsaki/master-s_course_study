from math import cos, sin
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





epsilon = np.linspace(0, 1, 10000)
plot_num = 10000
x0 = np.linspace(0, 1, num=plot_num)
x1 = np.reshape(x0, newshape=(plot_num, 1))

y0 = np.sin(2*np.pi*x0)
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
for i in range(20000):
    model2.set_weights(model2_weights_initializer())
    model2.save_weights(filepath='weights_initializer2_%06.f'%(i+1)+'.hdf5')
    
    if i == 0:
        y2 = model2.predict_on_batch(x1)
        y2 = (y2-np.mean(y2, axis=0))/np.std(y2, axis=0)
        err2 = np.mean(np.square(y1-y2), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err2 <= k:
                count_model2[j] += 1
        with open('err2_1.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([y2, err2])
    else:
        y = model2.predict_on_batch(x1)
        y = (y-np.mean(y, axis=0))/np.std(y, axis=0)
        err = np.mean(np.square(y1-y), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err <= k:
                count_model2[j] += 1
        with open('err2_1.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([y, err])


        if err < err2:
            err2 = err
            y2 = y
    
    


    model3.set_weights(model3_weights_initializer())
    model3.save_weights(filepath='weights_initializer3_%06.f'%(i+1)+'.hdf5')
    
    if i == 0:
        y3 = model3.predict_on_batch(x1)
        y3 = (y3-np.mean(y3, axis=0))/np.std(y3, axis=0)
        err3 = np.mean(np.square(y1-y3), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err3 <= k:
                count_model3[j] += 1
        with open('err3_1.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([y3, err3])
    else:
        y = model3.predict_on_batch(x1)
        y = (y-np.mean(y, axis=0))/np.std(y, axis=0)
        err = np.mean(np.square(y1-y), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err <= k:
                count_model3[j] += 1
        with open('err3_1.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([y, err])


        if err < err3:
            err3 = err
            y3 = y
    
    

    #print(i+1)

print(1, 'err2 =', err2)
print(1, 'err3 =', err3)




count_model2 = np.array(count_model2)
count_model3 = np.array(count_model3)

rate2 = count_model2/20000
rate3 = count_model3/20000

with open('rate2_1.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(rate2)

with open('rate3_1.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(rate2)


plt.plot(epsilon, rate2, color='red', linestyle='solid', label='target function 1, Network 1')
plt.plot(epsilon, rate3, color='red', linestyle='dashed', label='target function 1, Network 2')



y0 = np.sin(2*np.pi*(x0-0.5))
y1 = np.reshape(y0, newshape=(plot_num, 1))

count_model2 = [0]*10000
count_model3 = [0]*10000
for i in range(20000):
    model2.load_weights('weights_initializer2_%06.f'%(i+1)+'.hdf5')
       
    if i == 0:
        y2 = model2.predict_on_batch(x1)
        y2 = (y2-np.mean(y2, axis=0))/np.std(y2, axis=0)
        err2 = np.mean(np.square(y1-y2), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err2 <= k:
                count_model2[j] += 1
        with open('err2_2.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([y2, err2])
    else:
        y = model2.predict_on_batch(x1)
        y = (y-np.mean(y, axis=0))/np.std(y, axis=0)
        err = np.mean(np.square(y1-y), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err <= k:
                count_model2[j] += 1
        with open('err2_2.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([y, err])


        if err < err2:
            err2 = err
            y2 = y
    
    
    model3.load_weights('weights_initializer3_%06.f'%(i+1)+'.hdf5')
    
    if i == 0:
        y3 = model3.predict_on_batch(x1)
        y3 = (y3-np.mean(y3, axis=0))/np.std(y3, axis=0)
        err3 = np.mean(np.square(y1-y3), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err3 <= k:
                count_model3[j] += 1
        with open('err3_2.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([y3, err3])
    else:
        y = model3.predict_on_batch(x1)
        y = (y-np.mean(y, axis=0))/np.std(y, axis=0)
        err = np.mean(np.square(y1-y), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err <= k:
                count_model3[j] += 1
        with open('err3_2.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([y, err])


        if err < err3:
            err3 = err
            y3 = y
    
    

    #print(i+1)

print(2, 'err2 =', err2)
print(2, 'err3 =', err3)


count_model2 = np.array(count_model2)
count_model3 = np.array(count_model3)

rate2 = count_model2/20000
rate3 = count_model3/20000

with open('rate2_2.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(rate2)


with open('rate3_2.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(rate3)

plt.plot(epsilon, rate2, color='blue', linestyle='solid', label='target function 2, Network 1')
plt.plot(epsilon, rate3, color='blue', linestyle='dashed', label='target function 2, Network 2')



y0 = np.sin(2*np.pi*(x0-0.25))
y1 = np.reshape(y0, newshape=(plot_num, 1))

count_model2 = [0]*10000
count_model3 = [0]*10000
for i in range(20000):
    model2.load_weights('weights_initializer2_%06.f'%(i+1)+'.hdf5')
       
    if i == 0:
        y2 = model2.predict_on_batch(x1)
        y2 = (y2-np.mean(y2, axis=0))/np.std(y2, axis=0)
        err2 = np.mean(np.square(y1-y2), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err2 <= k:
                count_model2[j] += 1
        with open('err2_3.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([y2, err2])
    else:
        y = model2.predict_on_batch(x1)
        y = (y-np.mean(y, axis=0))/np.std(y, axis=0)
        err = np.mean(np.square(y1-y), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err <= k:
                count_model2[j] += 1
        with open('err2_3.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([y, err])


        if err < err2:
            err2 = err
            y2 = y
    
    
    model3.load_weights('weights_initializer3_%06.f'%(i+1)+'.hdf5')
    
    if i == 0:
        y3 = model3.predict_on_batch(x1)
        y3 = (y3-np.mean(y3, axis=0))/np.std(y3, axis=0)
        err3 = np.mean(np.square(y1-y3), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err3 <= k:
                count_model3[j] += 1
        with open('err3_3.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([y3, err3])
    else:
        y = model3.predict_on_batch(x1)
        y = (y-np.mean(y, axis=0))/np.std(y, axis=0)
        err = np.mean(np.square(y1-y), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err <= k:
                count_model3[j] += 1
        with open('err3_3.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([y, err])


        if err < err3:
            err3 = err
            y3 = y
    
    

    #print(i+1)


print(3, 'err2 =', err2)
print(3, 'err3 =', err3)

count_model2 = np.array(count_model2)
count_model3 = np.array(count_model3)

rate2 = count_model2/20000
rate3 = count_model3/20000

with open('rate2_3.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(rate2)


with open('rate3_3.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(rate3)

plt.plot(epsilon, rate2, color='green', linestyle='solid', label='target function 3, Network 1')
plt.plot(epsilon, rate3, color='green', linestyle='dashed', label='target function 3, Network 2')



y0 = np.sin(2*np.pi*(x0-0.75))
y1 = np.reshape(y0, newshape=(plot_num, 1))

count_model2 = [0]*10000
count_model3 = [0]*10000
for i in range(20000):
    model2.load_weights('weights_initializer2_%06.f'%(i+1)+'.hdf5')
       
    if i == 0:
        y2 = model2.predict_on_batch(x1)
        y2 = (y2-np.mean(y2, axis=0))/np.std(y2, axis=0)
        err2 = np.mean(np.square(y1-y2), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err2 <= k:
                count_model2[j] += 1
        with open('err2_4.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([y2, err2])
    else:
        y = model2.predict_on_batch(x1)
        y = (y-np.mean(y, axis=0))/np.std(y, axis=0)
        err = np.mean(np.square(y1-y), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err <= k:
                count_model2[j] += 1
        with open('err2_4.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([y, err])


        if err < err2:
            err2 = err
            y2 = y
    
    
    model3.load_weights('weights_initializer3_%06.f'%(i+1)+'.hdf5')
    
    if i == 0:
        y3 = model3.predict_on_batch(x1)
        y3 = (y3-np.mean(y3, axis=0))/np.std(y3, axis=0)
        err3 = np.mean(np.square(y1-y3), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err3 <= k:
                count_model3[j] += 1
        with open('err3_4.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([y3, err3])
    else:
        y = model3.predict_on_batch(x1)
        y = (y-np.mean(y, axis=0))/np.std(y, axis=0)
        err = np.mean(np.square(y1-y), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err <= k:
                count_model3[j] += 1
        with open('err3_4.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([y, err])


        if err < err3:
            err3 = err
            y3 = y
    
    

    #print(i+1)


print(4, 'err2 =', err2)
print(4, 'err3 =', err3)

count_model2 = np.array(count_model2)
count_model3 = np.array(count_model3)

rate2 = count_model2/20000
rate3 = count_model3/20000

with open('rate2_4.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(rate2)


with open('rate3_4.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(rate3)

plt.plot(epsilon, rate2, color='yellow', linestyle='solid', label='target function 4, Network 1')
plt.plot(epsilon, rate3, color='yellow', linestyle='dashed', label='target function 4, Network 2')




plt.legend()
plt.xlabel("epsilon")
plt.ylabel("R(epsilon)")
plt.savefig("sin_norm_compound_shift.png")
plt.clf()