import numpy as np
import time, warnings
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner
from tqdm import tqdm
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.trial import Trial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras_tuner as kt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Dropout,
    Flatten,
    Conv2D,
    MaxPool2D,
    Activation,
)

classes = ['air_conditioning', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
sns.countplot(metadata, y="class")
plt.show()


(X_train, y_train), (X_test, y_test) = #load urbansounds data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = normalize(X_train)
X_test = normalize(X_test)
#X_train = X_train.reshape(-1,32, 32, 3)  # reshaping for convnet

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#Building a hypermodel:
# function to build a hypermodel
# takes an argument from which to sample hyperparameters
def build_model(hp):
    model = Sequential()

    model.add(Conv2D(hp.Int('input_units', min_value=32, max_value=256, step=32), (3, 3), input_shape=X_train.shape[1:]))
    model.add(Activation('tanh'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    for i in range(hp.Int('n_layers', 1, 4)):  # adding variation of layers, this parameter will have a convnet with 2–5 convolutions
        model.add(Conv2D(hp.Int(f'conv_{i}_units', min_value=32, max_value=256, step=32), (3, 3)))
        model.add(Activation('tanh'))
        # adding dropout
        model.add(tf.keras.layers.Dropout(rate=hp.Float('rate', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())



    for i in range(hp.Int('n_connections', 1, 4)):
        model.add(Dense(hp.Choice(f'n_nodes',
                                  values=[128, 256, 512, 1024])))
        model.add(Activation('tanh'))

    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(optimizer="adam", #optimization algorithm used is Adam

                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model

tuner = RandomSearch(
    build_model,
    objective='val_acc',
    max_trials=5,  # how many model variations to test?
    executions_per_trial=3,  # how many trials per variation? (same model could perform differently)
)

NUM_EPOCH = 25
tuner.search(x=X_train, y=y_train, epochs=NUM_EPOCH, batch_size=32, validation_data=(X_test, y_test))

print(tuner.get_best_models()[0].summary())
print(tuner.get_best_hyperparameters()[0].values)
model = tuner.get_best_models(num_models=1)[0]
print (model.summary())
# Evaluate the best model.
loss, accuracy = model.evaluate(X_test, y_test)
print('loss:', loss)
print('accuracy:', accuracy)
