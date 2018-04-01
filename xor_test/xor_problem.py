import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'deep_neural_network_class')))
from deep_neural_net_class import DNN

X_train = np.array([[0,0],[1,1],
                    [0,1],[1,0]])
X_train = X_train.T

Y_train = np.array([0,0,1,1])
Y_train = Y_train.reshape(1, 4)

num_iterations = 20000
learning_rate = 0.8
layers_dims = [2, 3, 1]

model = DNN(X_train, Y_train, layers_dims, learning_rate, num_iterations, plot = True, print_every = 1000,
            initialization = 'he', activations = ["sigmoid", "sigmoid"])

params = model.optimize()

p0,acc = model.forward_prop(np.array([[0],
                                      [0]]), params)
p1,acc = model.forward_prop(np.array([[0],
                                      [1]]), params)
p2,acc = model.forward_prop(np.array([[1],
                                      [0]]), params)
p3,acc = model.forward_prop(np.array([[1],
                                      [1]]), params)
