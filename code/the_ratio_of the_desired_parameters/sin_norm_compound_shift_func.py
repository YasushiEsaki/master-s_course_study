from math import cos, sin
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import csv
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Input
from pylab import rcParams




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
epsilon = np.linspace(0, 1, 10000)
for i in range(20000):
    model2.load_weights('weights_initializer2_%06.f'%(i+1)+'.hdf5')
    
    if i == 0:
        y2 = model2.predict_on_batch(x1)
        y2 = (y2-np.mean(y2, axis=0))/np.std(y2, axis=0)
        err2 = np.mean(np.square(y1-y2), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err2 <= k:
                count_model2[j] += 1
    else:
        y = model2.predict_on_batch(x1)
        y = (y-np.mean(y, axis=0))/np.std(y, axis=0)
        err = np.mean(np.square(y1-y), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err <= k:
                count_model2[j] += 1
        

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
    else:
        y = model3.predict_on_batch(x1)
        y = (y-np.mean(y, axis=0))/np.std(y, axis=0)
        err = np.mean(np.square(y1-y), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err <= k:
                count_model3[j] += 1

        if err < err3:
            err3 = err
            y3 = y
    
    

    #print(i+1)

print(1, 'err2 =', err2)
print(1, 'err3 =', err3)

y2 = np.reshape(y2, newshape=plot_num)
y3 = np.reshape(y3, newshape=plot_num)


plt.plot(y0, color='red', label='sin')
plt.plot(y2, color='blue', label='Network 1')
plt.plot(y3, color='green', label='Network 2')
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.savefig("sin1_norm_func.png")
plt.clf()


count_model2 = np.array(count_model2)
count_model3 = np.array(count_model3)

rate2 = count_model2/20000
rate3 = count_model3/20000



y0 = np.sin(2*np.pi*(x0-0.5))
y1 = np.reshape(y0, newshape=(plot_num, 1))

count_model2 = [0]*10000
count_model3 = [0]*10000
epsilon = np.linspace(0, 1, 10000)
for i in range(20000):
    model2.load_weights('weights_initializer2_%06.f'%(i+1)+'.hdf5')
       
    if i == 0:
        y2 = model2.predict_on_batch(x1)
        y2 = (y2-np.mean(y2, axis=0))/np.std(y2, axis=0)
        err2 = np.mean(np.square(y1-y2), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err2 <= k:
                count_model2[j] += 1
        
    else:
        y = model2.predict_on_batch(x1)
        y = (y-np.mean(y, axis=0))/np.std(y, axis=0)
        err = np.mean(np.square(y1-y), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err <= k:
                count_model2[j] += 1
        

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
        
    else:
        y = model3.predict_on_batch(x1)
        y = (y-np.mean(y, axis=0))/np.std(y, axis=0)
        err = np.mean(np.square(y1-y), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err <= k:
                count_model3[j] += 1
        

        if err < err3:
            err3 = err
            y3 = y
    
    

    #print(i+1)

print(2, 'err2 =', err2)
print(2, 'err3 =', err3)

y2 = np.reshape(y2, newshape=plot_num)
y3 = np.reshape(y3, newshape=plot_num)

plt.plot(y0, color='red', label='sin')
plt.plot(y2, color='blue', label='Network 1')
plt.plot(y3, color='green', label='Network 2')
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.savefig("sin2_norm_func.png")
plt.clf()


count_model2 = np.array(count_model2)
count_model3 = np.array(count_model3)

rate2 = count_model2/20000
rate3 = count_model3/20000


y0 = np.sin(2*np.pi*(x0-0.25))
y1 = np.reshape(y0, newshape=(plot_num, 1))

count_model2 = [0]*10000
count_model3 = [0]*10000
epsilon = np.linspace(0, 1, 10000)
for i in range(20000):
    model2.load_weights('weights_initializer2_%06.f'%(i+1)+'.hdf5')
       
    if i == 0:
        y2 = model2.predict_on_batch(x1)
        y2 = (y2-np.mean(y2, axis=0))/np.std(y2, axis=0)
        err2 = np.mean(np.square(y1-y2), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err2 <= k:
                count_model2[j] += 1
    else:
        y = model2.predict_on_batch(x1)
        y = (y-np.mean(y, axis=0))/np.std(y, axis=0)
        err = np.mean(np.square(y1-y), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err <= k:
                count_model2[j] += 1
        
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
    else:
        y = model3.predict_on_batch(x1)
        y = (y-np.mean(y, axis=0))/np.std(y, axis=0)
        err = np.mean(np.square(y1-y), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err <= k:
                count_model3[j] += 1
        
        if err < err3:
            err3 = err
            y3 = y
    
    

    #print(i+1)


print(3, 'err2 =', err2)
print(3, 'err3 =', err3)

y2 = np.reshape(y2, newshape=plot_num)
y3 = np.reshape(y3, newshape=plot_num)

plt.plot(y0, color='red', label='sin')
plt.plot(y2, color='blue', label='Network 1')
plt.plot(y3, color='green', label='Network 2')
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.savefig("sin3_norm_func.png")
plt.clf()

count_model2 = np.array(count_model2)
count_model3 = np.array(count_model3)

rate2 = count_model2/20000
rate3 = count_model3/20000


y0 = np.sin(2*np.pi*(x0-0.75))
y1 = np.reshape(y0, newshape=(plot_num, 1))

count_model2 = [0]*10000
count_model3 = [0]*10000
epsilon = np.linspace(0, 1, 10000)
for i in range(20000):
    model2.load_weights('weights_initializer2_%06.f'%(i+1)+'.hdf5')
       
    if i == 0:
        y2 = model2.predict_on_batch(x1)
        y2 = (y2-np.mean(y2, axis=0))/np.std(y2, axis=0)
        err2 = np.mean(np.square(y1-y2), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err2 <= k:
                count_model2[j] += 1
        
    else:
        y = model2.predict_on_batch(x1)
        y = (y-np.mean(y, axis=0))/np.std(y, axis=0)
        err = np.mean(np.square(y1-y), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err <= k:
                count_model2[j] += 1
        

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
        
    else:
        y = model3.predict_on_batch(x1)
        y = (y-np.mean(y, axis=0))/np.std(y, axis=0)
        err = np.mean(np.square(y1-y), axis=0)[0]
        for j, k in enumerate(epsilon):
            if err <= k:
                count_model3[j] += 1
        

        if err < err3:
            err3 = err
            y3 = y
    
    

    #print(i+1)

print(2, 'err2 =', err2)
print(2, 'err3 =', err3)

y2 = np.reshape(y2, newshape=plot_num)
y3 = np.reshape(y3, newshape=plot_num)

plt.plot(y0, color='red', label='sin')
plt.plot(y2, color='blue', label='Network 1')
plt.plot(y3, color='green', label='Network 2')
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.savefig("sin4_norm_func.png")
plt.clf()


count_model2 = np.array(count_model2)
count_model3 = np.array(count_model3)

rate2 = count_model2/20000
rate3 = count_model3/20000