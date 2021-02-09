import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Input
import csv

def linear_regions(x, plot_num, model, model_number):
    count = 0
    count_list = []
    for j in range(plot_num-2):
        y = []
        for k in range(3):
            x_reshape = np.reshape(x[j+k], newshape=(1, 1))
            output = model.predict_on_batch(x_reshape)
            y.append(np.reshape(output, newshape=(1)))

        gradient1 = (y[1]-y[0])/(x[j+1]-x[j])
        gradient2 = (y[2]-y[1])/(x[j+2]-x[j+1])
        d = abs(gradient1 - gradient2)[0]

        if d < 0.2:
            count += 1
        else:
            count_list.append(count)
            count = 0

    

    count_list.append(count)
    fineness = max(count_list)/plot_num
    
    with open('err_%03.f'%(model_number)+'.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(count_list)
        writer.writerow([fineness])

    return fineness

plot_num = 100000
x0 = np.linspace(0, 1, num=plot_num)

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


model2_fineness_list = []
model3_fineness_list = []
for i in range(1000):
    model2.load_weights('weights_initializer2_%03.f'%(i+1)+'.hdf5')

    model2_fineness = linear_regions(x0, plot_num, model2, 2)
    print('fineness of model2 is', model2_fineness)

    model2_fineness_list.append(model2_fineness)

    model3.load_weights('weights_initializer3_%03.f'%(i+1)+'.hdf5')

    model3_fineness = linear_regions(x0, plot_num, model3, 3)
    print('fineness of model3 is', model3_fineness)

    model3_fineness_list.append(model3_fineness)

    print(i+1)
    

print(min(model2_fineness_list))
print(min(model3_fineness_list))