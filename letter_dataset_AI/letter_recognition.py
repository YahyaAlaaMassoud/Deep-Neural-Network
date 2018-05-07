import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'deep_neural_network_class')))
from deep_neural_net_class import DNN
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
#--------------------------Data Preprocessing--------------------------#
dataset = pd.read_csv('letter-recognition.data', header = None)
# Choosing columns
X = dataset.iloc[:, 1:].values
Y = dataset.iloc[:, 0].values
Y = np.array([ord(letter) - ord('A') for letter in list(Y)]).reshape(20000, 1)
# one hot encoding for Y
onehotencoder = OneHotEncoder(categorical_features = [0])
Y = onehotencoder.fit_transform(Y).toarray()
# Splitting data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
# Scaling features
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#--------------------------Training--------------------------#
# Init model
from keras.models import Sequential
from keras.layers import Dense, Dropout
classifier = Sequential()
classifier.add(Dense(units = 40, kernel_initializer = 'uniform', activation = 'relu', input_dim = 16))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 26, kernel_initializer = 'uniform', activation = 'softmax'))
# Compile model
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Train model
history = classifier.fit(X_train, Y_train, batch_size = 64, epochs = 50)
# Test model
y_pred = classifier.predict(X_test)
y_pred_values = []
y_test_values = []
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
for row in y_pred:
    y_pred_values.append(np.argmax(row))
for row in Y_test:
    y_test_values.append(np.argmax(row))
cm = confusion_matrix(y_test_values, y_pred_values)
accuracy = accuracy_score(y_test_values, y_pred_values)
print(accuracy)
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Save model JSON
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")
# Load model func
def load_model():
    from keras.models import model_from_json
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    return loaded_model
load_model()