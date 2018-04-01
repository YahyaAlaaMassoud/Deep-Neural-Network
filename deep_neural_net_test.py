import numpy as np
import matplotlib.pyplot as plt
from helper_functions import sigmoid, relu, sigmoid_derivative, relu_derivative
from deep_neural_net_class import DNN
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score


dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 8].values
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
Y_train = Y_train.reshape(576, 1)
Y_test = Y_test.reshape(192, 1)
X_train = X_train.T
X_test = X_test.T
Y_train = Y_train.T
Y_test = Y_test.T 
layers_dims = [8, 50, 120, 70, 50, 20, 5, 1]
learning_rate = 0.2
num_iterations = 1500
lr = DNN(X_train, Y_train, layers_dims, learning_rate, num_iterations, plot = True, initialization = 'he', lambd = 0, dropout = 1)
parameters = lr.optimize()
Y_prediction_train = lr.predict(X_train, Y_train, parameters)
#print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
Y_prediction_test = lr.predict(X_test, Y_test, parameters)
#print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    

def testSocialNetworkAds():
    dataset = pd.read_csv('Datasets/Social Network Ads Dataset/Social_Network_Ads.csv')
    X = dataset.iloc[:, [2, 3]].values
    Y = dataset.iloc[:, 4].values
    from sklearn.cross_validation import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

    sc_X = MinMaxScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    Y_train = Y_train.reshape(300, 1)
    Y_test = Y_test.reshape(100, 1)
    X_train = X_train.T
    X_test = X_test.T
    Y_train = Y_train.T
    Y_test = Y_test.T   
    
#    plt.scatter(X_train[0, :], X_train[1, :], c=Y_train, s=40, cmap=plt.cm.Spectral)
        
    layers_dims = [2, 50, 120, 70, 50, 20, 5, 1]
    learning_rate = 0.08
    num_iterations = 2800
    np.random.seed(1)
    
    lr = DNN(X_train, Y_train, layers_dims, learning_rate, num_iterations, plot = False, initialization = 'he', lambd = 0, dropout = 1) # lambd = 0.08
    params = lr.optimize()
    
    lr.predict(X_train, Y_train, params)
    lr.predict(X_test, Y_test, params)

    return 1

p = testSocialNetworkAds()

def testTitanic():
    train_set = pd.read_csv('Datasets/Titanic Survivals Dataset/titanic_train.csv')

    X_train = train_set.iloc[:, 2:]
    X_train = X_train.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], 1)
    
    X_train = X_train.values
    
    imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
    imputer = imputer.fit(X_train[:, 2:3])
    X_train[:, 2:3] = imputer.transform(X_train[:, 2:3])
    
    labelencoder_X = LabelEncoder()
    X_train[:, 1] = labelencoder_X.fit_transform(X_train[:, 1])
    
    onehotencoder = OneHotEncoder(categorical_features = [0])
    X_train = onehotencoder.fit_transform(X_train).toarray()
    X_train = X_train[:, 1:]
    
    scmm = MinMaxScaler()
    X_train = scmm.fit_transform(X_train)
    
    X_train = X_train.T
    
    
    X_test = pd.read_csv('Datasets/Titanic Survivals Dataset/titanic_test.csv')
    passengerIds = X_test.iloc[:, 0:1]
    X_test = X_test.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], 1)
    X_test.fillna(X_test.mean(), inplace=True)
    X_test = X_test.values
    
    labelencoder_X = LabelEncoder()
    X_test[:, 1] = labelencoder_X.fit_transform(X_test[:, 1])
    
    X_test = X_test.astype(np.float)
    
    X_test = onehotencoder.fit_transform(X_test).toarray()
    X_test = X_test[:, 1:]
    
    X_test = scmm.fit_transform(X_test)
    
    X_test = X_test.T
    
    Y_train = train_set.iloc[:, 1]
    Y_train = np.asarray(Y_train).reshape(891,1)
    Y_train = Y_train.T
    

#    layers_dims = [7, 50, 120, 70, 50, 20, 5, 1]
#    learning_rate = 0.3
    layers_dims = [7, 50, 70, 80, 5, 1]
    learning_rate = 0.8
    num_iterations = 1000
    
    lr = DNN(X_train, Y_train, layers_dims, learning_rate, num_iterations, plot = False, initialization = 'he', lambd = 0., dropout = 1.) # lambd = 0.08
    params = lr.optimize()
    
    lr.predict(X_train, Y_train, params)
    p = lr.predict(X_test, 'empty', params)

    return p, passengerIds

p, ids = testTitanic()
    
ids = ids.astype(np.int32)
ids = ids.reshape(418, 1)
p = p.T
p = p.astype(np.int64)
a = np.concatenate((ids, p), axis = 1)
df = pd.DataFrame(a,columns = ["PassengerId", "Survived"])
df.to_csv("84.84 accuracy.csv", index=False)


