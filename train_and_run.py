import sys
if len(sys.argv)>1:
    print('Argument 1 is '+ sys.argv [1])

import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
np.random.seed(888)
from keras.utils.np_utils import to_categorical
from keras.layers.core import Dense, Dropout, Activation, ActivityRegularization,Flatten
#from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential, model_from_json
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

X_input_training_data= np.load('X_input_training_data.npy')
y_output_training_data=np.load('y_output_training_data.npy')
X_test= np.load('X_input_testing_data.npy')
y_output_testing_data=np.load('y_output_testing_data.npy')

X_train, X_validate_train, y_train, y_validate_train = train_test_split(X_input_training_data, y_output_training_data, test_size=0.3, random_state=8)

# one hot encode the ground truths
Y_train= to_categorical(y_train, num_classes=500)
Y_validate_train= to_categorical(y_validate_train, num_classes=500)
Y_test = to_categorical(y_output_testing_data,  num_classes=500)

X_train = np.array(X_train) # array with 0 to 255 values
X_validate_train = np.array(X_validate_train) # array with 0 to 255 values
X_test = np.array(X_test) # array with 0 to 255 values


X_train = X_train.reshape(X_train.shape[0], -1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], -1).astype('float32')
X_validate_train = X_validate_train.reshape(X_validate_train.shape[0], -1).astype('float32')

X_train = (X_train - np.mean(X_train))/np.std(X_train)
X_test = (X_test - np.mean(X_test))/np.std(X_test) 
X_validate_train = (X_validate_train - np.mean(X_validate_train))/np.std(X_validate_train)
'''
model = Sequential([
    Dense(128, input_dim=32*32),
    Activation('relu'),
    Dropout(0.4),
    Dense(500),
    Activation('softmax'),
])'''
model = Sequential([Dense(256, input_dim=32*32),
                    Activation('relu'),
                    Dropout(0.6),
                    Dense(512),
                    Activation('relu'),
                    Dropout(0.6),
                    Dense(500),
                    Activation('softmax'),
                    ])
    

''' 
model = Sequential([
        Dense(32, input_dim=32*32),
        Activation('relu'),
        Dropout(0.5),
        Dense(64),
        Activation('relu'),
        Dense(500),
        Activation('softmax'),
        ])'''
'''
model = Sequential([
Dense(32, input_dim=32*32),
Activation('relu'),
# 500 classes
Dense(500),
Activation('softmax'),
])

'''

def baseline_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(32,32,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.002), metrics=['accuracy'])
    return model

rmsprop=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.01,momentum=0.1, decay=0.01, nesterov=False)
adam = Adam(lr=0.001, beta_1=0.9,beta_2=0.999,epsilon=1e-08, decay=0.0)
adagrad = Adagrad(lr=1.0, epsilon=1e-8, decay=0.0)
adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-8, decay=0.0)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])


callback = [EarlyStopping(monitor='val_loss',patience=10,verbose=0)]
history=model.fit(X_train, Y_train, batch_size=32, epochs=200,
       validation_data=(X_validate_train, Y_validate_train),callbacks=callback)
score = model.evaluate(X_test, Y_test)


print('Test score:', score[0])
print('Test accuracy:', score[1])


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

print ("Finished")
