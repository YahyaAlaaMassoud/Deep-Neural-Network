from keras.datasets import mnist
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'deep_neural_network_class')))
from deep_neural_net_class import DNN

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28)
x_train = x_train.T
x_train = x_train / 255.

x_test = x_test.reshape(10000, 28*28)
x_test = x_test.T
x_test = x_test / 255.

y_train = y_train.reshape(60000, 1)
y_train = y_train.T

y_test = y_test.reshape(10000, 1)
y_test = y_test.T

def one_hot_encode(arr, classes):
    one_hot_encoded = []
    for item in arr:
        temp = np.zeros((classes))
        temp[item] = 1
        one_hot_encoded.append(temp)
    one_hot_encoded = np.asarray(one_hot_encoded)
    return one_hot_encoded

y_train = one_hot_encode(y_train.T, 10).T
y_test = one_hot_encode(y_test.T, 10).T

num_iterations = 1000
learning_rate = 0.3
layers_dims = [784, 200, 10]

model = DNN(x_train, y_train, layers_dims, learning_rate, num_iterations, plot = False, print_every = 20,
            initialization = 'he', lambd = 0, dropout = 1)
params = model.optimize()

p, acc = model.predict(x_test, y_test, params)


#import matplotlib.pyplot as plt
#plt.imshow(x_test[:,9980].reshape(28,28))













