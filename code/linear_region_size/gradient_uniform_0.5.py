import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Input
import csv



# genarete the velues of the parameters in Network 1
def model1_weights_initializer():
    weights_initializer1 = [[[]]]

    for _ in range(4):
        weights_initializer1[0][0].append(np.random.rand())
    


    weights_initializer1.append([])

    for j in range(4):
        weights_initializer1[1].append(np.random.rand())


    weights_initializer1.append([[]])


    for j in range(4):
        for _ in range(4):
            weights_initializer1[2][j].append(np.random.rand())
    
        if j < 3:
            weights_initializer1[2].append([])


    weights_initializer1.append([])

    for j in range(4):
        weights_initializer1[3].append(np.random.rand())


    weights_initializer1.append([[]])

    for j in range(4):
        for _ in range(4):
            weights_initializer1[4][j].append(np.random.rand())
    
        if j < 3:
            weights_initializer1[4].append([])


    weights_initializer1.append([])

    for j in range(4):
        weights_initializer1[5].append(np.random.rand())


    weights_initializer1.append([[]])

    for j in range(4):
        for _ in range(4):
            weights_initializer1[6][j].append(np.random.rand())
    
        if j < 3:
            weights_initializer1[6].append([])


    weights_initializer1.append([])

    for j in range(4):
        weights_initializer1[7].append(np.random.rand())


    weights_initializer1.append([[]])

    for j in range(4):
        for _ in range(4):
            weights_initializer1[8][j].append(np.random.rand())
    
        if j < 3:
            weights_initializer1[8].append([])


    weights_initializer1.append([])

    for j in range(4):
        weights_initializer1[9].append(np.random.rand())


    weights_initializer1.append([[]])

    for j in range(4):
        weights_initializer1[10][j].append(np.random.rand())
        if j < 3:
            weights_initializer1[10].append([])


    weights_initializer1.append([])

    weights_initializer1[11].append(np.random.rand())

    
    return weights_initializer1

# genarete the values of the parameters in Network 2
def model2_weights_initializer():
    weights_initializer2= [[[]]]

    for _ in range(20):
        weights_initializer2[0][0].append(np.random.rand())


    weights_initializer2.append([])

    for j in range(20):
        weights_initializer2[1].append(np.random.rand())


    weights_initializer2.append([[]])


    for j in range(20):
        weights_initializer2[2][j].append(np.random.rand())
        if j < 19:
            weights_initializer2[2].append([])


    weights_initializer2.append([])

    weights_initializer2[3].append(np.random.rand())


    return weights_initializer2

# compute the fineness of linear regions
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

        if d < 0.5:
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

# set the input values
plot_num = 100000
x0 = np.linspace(0, 1, num=plot_num)

# define Network 1
inputs = Input(shape=(1,))
x = Dense(units=4, activation='relu')(inputs)
x = Dense(units=4, activation='relu')(x)
x = Dense(units=4, activation='relu')(x)
x = Dense(units=4, activation='relu')(x)
x = Dense(units=4, activation='relu')(x)
predictions = Dense(units=1)(x)

model1 = Model(inputs=inputs, outputs=predictions)

# define Network 2
inputs = Input(shape=(1,))
x = Dense(units=20, activation='relu')(inputs)
predictions = Dense(units=1)(x)

model2 = Model(inputs=inputs, outputs=predictions)


model1_fineness_list = []
model2_fineness_list = []
for i in range(1000):

    # set the values of the parameters in Network 1 
    model1.set_weights(model1_weights_initializer())
    model1.save_weights(filepath='weights_initializer1_%03.f'%(i+1)+'.hdf5')

    # compute the fineness of linear regions of Network 1
    model1_fineness = linear_regions(x0, plot_num, model1, 1)
    print('fineness of model1 =', model1_fineness)

    model1_fineness_list.append(model1_fineness)

    # set the values of the parameters in Network 2
    model2.set_weights(model2_weights_initializer())
    model2.save_weights(filepath='weights_initializer2_%03.f'%(i+1)+'.hdf5')

    # compute the fineness of linear regions of Network 2
    model2_fineness = linear_regions(x0, plot_num, model2, 2)
    print('fineness of model2 =', model2_fineness)

    model2_fineness_list.append(model2_fineness)

    print(i+1)
    
# show the minimum fineness of linear regions
print(min(model1_fineness_list))
print(min(model2_fineness_list))





    

    


        
        





