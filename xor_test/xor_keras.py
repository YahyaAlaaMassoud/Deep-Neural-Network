from keras.models import Sequential
from keras.layers import Dense
import numpy as np

X_train = np.array([[0,0],[1,1],
                    [0,1],[1,0]])

Y_train = np.array([0,0,1,1])
Y_train = Y_train.reshape(1, 4).T

classifier = Sequential()
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = 2))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, Y_train, batch_size = 4, epochs = 2000)
