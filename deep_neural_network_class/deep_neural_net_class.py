import numpy as np
import matplotlib.pyplot as plt
from helper_functions import sigmoid, relu, sigmoid_derivative, relu_derivative

class DNN():
    def __init__(self, X, Y, layers_dims, learning_rate, epochs, print_cost = True, print_every = 100, plot = True, initialization = 'he', lambd = 0.1, dropout = 0.):
        self.__X = X
        self.__Y = Y
        self.__layers_dims = layers_dims
        self.__learning_rate = learning_rate
        self.__epochs = epochs
        self.__print_every = print_every
        self.__plot = plot
        self.__print_cost = print_cost
        self.__parameters = self.init_params_he(layers_dims)
        self.__initialization = initialization
        self.__lambd = lambd
        self.__dropout = dropout
        self.__D = {}
        
    def get_params(self):
        return self.__parameters
        
    def optimize(self):
        costs = []
        if self.__initialization == 'random':
            parameters = self.init_params_random(self.__layers_dims)
        elif self.__initialization == 'zero':
            parameters = self.init_params_zeros(self.__layers_dims)
        else:
            parameters = self.init_params_he(self.__layers_dims)
        for i in range(0, self.__epochs + 1):
            AL, caches = self.forward_prop(self.__X, parameters)
            cost = self.compute_cost(AL, self.__Y)
            grads = self.backward_prop(AL, self.__Y, caches)
            parameters = self.update_parameters(parameters, grads, self.__learning_rate)
            if i % self.__print_every == 0 and self.__print_cost == True:
                _, acc = self.predict(self.__X, self.__Y, parameters) 
                print ("Cost after iteration " + str(i) + ": " + str(np.round(np.sum(cost),4)) + " Accuracy: " + str(acc))
                costs.append(cost)
        self.__parameters = parameters
        if self.__plot == True:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(self.__learning_rate))
            plt.show()
        return parameters
        
    def init_params_he(self, layers_dims):
        np.random.seed(1)
        parameters = {}
        L = len(layers_dims)
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2. / layers_dims[l - 1]) #/ np.sqrt(layers_dims[l - 1])#* np.sqrt(2. / layers_dims[l - 1])
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        return parameters
    
    def init_params_zeros(self, layers_dims):
        parameters = {}
        L = len(layers_dims)
        for l in range(1, L):
            parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        return parameters
    
    def init_params_random(self, layers_dims):
        parameters = {}
        L = len(layers_dims)
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.rand(layers_dims[l], layers_dims[l - 1]) * 0.1
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        return parameters
    
    def linear_forward(self, A, W, b):
        Z = W.dot(A) + b
        cache = (A, W, b)
        return Z, cache
    
    def activate_forward(self, A_prev, W, b, activation):
        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)
        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)
        cache = (linear_cache, activation_cache)
        return A, cache
    
    def forward_prop(self, X, parameters):
        caches = []
        A = X
        L = len(parameters) // 2    
        for l in range(1, L):
            A_prev = A 
            A, cache = self.activate_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
            caches.append(cache)
        AL, cache = self.activate_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
        caches.append(cache)
        return AL, caches
    
    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
        cost = np.squeeze(cost)
        return cost
    
    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = 1./m * np.dot(dZ,A_prev.T)
        db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T,dZ)
        return dA_prev, dW, db
    
    def activate_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        if activation == "relu":
            dZ = relu_derivative(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        elif activation == "sigmoid":
            dZ = sigmoid_derivative(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db
    
    def backward_prop(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = caches[L-1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.activate_backward(dAL, current_cache, activation = "sigmoid")
        
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.activate_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
    
        return grads
    
    def update_parameters(self, parameters, grads, learning_rate):
        L = len(parameters) // 2 
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        return parameters
    
    def predict(self, X, y, parameters):
        m = X.shape[1]
        p = np.zeros((y.shape[0],m))
        probas,_ = self.forward_prop(X, parameters)
        maxs = np.amax(probas, axis = 0)
        for i in range(0, probas.shape[1]):
            for j in range(0, probas.shape[0]):
                if probas[j,i] == maxs[i]:
                    p[j,i] = 1
                else:
                    p[j,i] = 0
        if y != 'empty':
            acc = 0.
            if p.shape[0] == 1:
                acc = np.round(np.sum((p == y)/m)*100,4)
#                print("Accuracy: "  + str(np.round(np.sum((p == y)/m)*100,4)))
            else:
                x = np.argmax(p, axis = 0)
                y = np.argmax(y, axis = 0)
                acc = np.round(np.sum(x==y)/m*100,4)
#                print("Accuracy: " + str(np.round(np.sum(x==y)/m*100,4)))
        return p, acc