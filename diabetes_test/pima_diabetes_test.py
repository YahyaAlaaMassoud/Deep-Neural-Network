import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'deep_neural_network_class')))
from deep_neural_net_class import DNN
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
#--------------------------Data Preprocessing--------------------------#
dataset = pd.read_csv('diabetes.csv')
# Choosing columns
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 8].values
# Splitting data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
# Scaling features
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Getting the proper shapes
Y_train = Y_train.reshape(576, 1)
Y_test = Y_test.reshape(192, 1)
X_train = X_train.T
X_test = X_test.T
Y_train = Y_train.T
Y_test = Y_test.T

num_iterations = 2500
learning_rate = 0.8
layers_dims = [8, 5, 1]

model = DNN(X_train, Y_train, layers_dims, learning_rate, num_iterations, plot = False, print_every = 20,
            initialization = 'he', lambd = 0, dropout = 1)
params = model.optimize()

