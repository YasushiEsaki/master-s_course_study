import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Input

def identification(x, model):
    id_list = []
    for xi in x:
        xi = np.reshape(xi, newshape=(1, 1))
        output = model.predict_on_batch(xi)[0][1]
        if abs(output - 1) < 0.00001:
            id_list.append(xi[0][0])

    return id_list


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

model2.load_weights('weights_initializer2_%05.f'%(171)+'.hdf5')
model3.load_weights('weights_initializer3_%05.f'%(168)+'.hdf5')


layer1_model2 = Model(inputs=model2.input, outputs=model2.get_layer(index=1).output)
layer2_model2 = Model(inputs=model2.input, outputs=model2.get_layer(index=2).output)
layer3_model2 = Model(inputs=model2.input, outputs=model2.get_layer(index=3).output)
layer4_model2 = Model(inputs=model2.input, outputs=model2.get_layer(index=4).output)
layer5_model2 = Model(inputs=model2.input, outputs=model2.get_layer(index=5).output)

hidden_model3 = Model(inputs=model3.input, outputs=model3.get_layer(index=1).output)

plot_num = 100000
x0 = np.linspace(0, 1, num=plot_num)

print('layer1_model2 =', identification(x0, layer1_model2))
print('layer2_model2 =', identification(x0, layer2_model2))
print('layer3_model2 =', identification(x0, layer3_model2))
print('layer4_model2 =', identification(x0, layer4_model2))
print('layer5_model2 =', identification(x0, layer5_model2))
print('hidden_model3 =', identification(x0, hidden_model3))