import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Input
import csv




def model2_weights_initializer():
    weights_initializer2 = [[[]]]

    for _ in range(4):
        weights_initializer2[0][0].append(np.random.rand())
    


    weights_initializer2.append([])

    for j in range(4):
        weights_initializer2[1].append(np.random.rand())


    weights_initializer2.append([[]])


    for j in range(4):
        for _ in range(4):
            weights_initializer2[2][j].append(np.random.rand())
    
        if j < 3:
            weights_initializer2[2].append([])


    weights_initializer2.append([])

    for j in range(4):
        weights_initializer2[3].append(np.random.rand())


    weights_initializer2.append([[]])

    for j in range(4):
        for _ in range(4):
            weights_initializer2[4][j].append(np.random.rand())
    
        if j < 3:
            weights_initializer2[4].append([])


    weights_initializer2.append([])

    for j in range(4):
        weights_initializer2[5].append(np.random.rand())


    weights_initializer2.append([[]])

    for j in range(4):
        for _ in range(4):
            weights_initializer2[6][j].append(np.random.rand())
    
        if j < 3:
            weights_initializer2[6].append([])


    weights_initializer2.append([])

    for j in range(4):
        weights_initializer2[7].append(np.random.rand())


    weights_initializer2.append([[]])

    for j in range(4):
        for _ in range(4):
            weights_initializer2[8][j].append(np.random.rand())
    
        if j < 3:
            weights_initializer2[8].append([])


    weights_initializer2.append([])

    for j in range(4):
        weights_initializer2[9].append(np.random.rand())


    weights_initializer2.append([[]])

    for j in range(4):
        weights_initializer2[10][j].append(np.random.rand())
        if j < 3:
            weights_initializer2[10].append([])


    weights_initializer2.append([])

    weights_initializer2[11].append(np.random.rand())

    
    return weights_initializer2


def model3_weights_initializer():
    weights_initializer3 = [[[]]]

    for _ in range(20):
        weights_initializer3[0][0].append(np.random.rand())


    weights_initializer3.append([])

    for j in range(20):
        weights_initializer3[1].append(np.random.rand())


    weights_initializer3.append([[]])


    for j in range(20):
        weights_initializer3[2][j].append(np.random.rand())
        if j < 19:
            weights_initializer3[2].append([])


    weights_initializer3.append([])

    weights_initializer3[3].append(np.random.rand())


    return weights_initializer3


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


def identification(x, model):
    id_list = []
    for xi in x:
        xi = np.reshape(xi, newshape=(1, 1))
        output = model.predict_on_batch(xi)[0][0]
        if abs(output - 0.1) < 0.01:
            id_list.append(xi[0][0])

    return id_list



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
    model2.set_weights(model2_weights_initializer())
    model2.save_weights(filepath='weights_initializer2_%05.f'%(i+1)+'.hdf5')

    model2_fineness = linear_regions(x0, plot_num, model2, 2)
    print('fineness of model2 is', model2_fineness)

    model2_fineness_list.append(model2_fineness)

    model3.set_weights(model3_weights_initializer())
    model3.save_weights(filepath='weights_initializer3_%05.f'%(i+1)+'.hdf5')

    model3_fineness = linear_regions(x0, plot_num, model3, 3)
    print('fineness of model3 is', model3_fineness)

    model3_fineness_list.append(model3_fineness)

    print(i+1)
    

print(min(model2_fineness_list))
print(min(model3_fineness_list))

model2_fineness_list = np.array(model2_fineness_list)
model3_fineness_list = np.array(model3_fineness_list)
model2_fineness_argmin = np.argmin(model2_fineness_list)
model3_fineness_argmin = np.argmin(model3_fineness_list)

print('model2_fineness_argmin =', model2_fineness_argmin)
print('model3_fineness_argmin =', model3_fineness_argmin)

model2.load_weights('weights_initializer2_%05.f'%(model2_fineness_argmin+1)+'.hdf5')
model3.load_weights('weights_initializer3_%05.f'%(model3_fineness_argmin+1)+'.hdf5')


layer1_model2 = Model(inputs=model2.input, outputs=model2.get_layer(index=1).output)
layer2_model2 = Model(inputs=model2.input, outputs=model2.get_layer(index=2).output)
layer3_model2 = Model(inputs=model2.input, outputs=model2.get_layer(index=3).output)
layer4_model2 = Model(inputs=model2.input, outputs=model2.get_layer(index=4).output)
layer5_model2 = Model(inputs=model2.input, outputs=model2.get_layer(index=5).output)

hidden_model3 = Model(inputs=model3.input, outputs=model3.get_layer(index=1).output)


print('layer1_model2 =', identification(x0, layer1_model2))
print('layer2_model2 =', identification(x0, layer2_model2))
print('layer3_model2 =', identification(x0, layer3_model2))
print('layer4_model2 =', identification(x0, layer4_model2))
print('layer5_model2 =', identification(x0, layer5_model2))
print('hidden_model3 =', identification(x0, hidden_model3))



