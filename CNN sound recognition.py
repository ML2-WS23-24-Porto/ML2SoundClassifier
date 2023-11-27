import numpy as np
import os
import pandas as pd
import time, warnings
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import KFold
import keras_tuner
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

def conv_array(root_folder):
    image_data = []
    all_labels = []
    for class_label in range(1, 11):
        class_folder_path = os.path.join(root_folder, f"fold{class_label}")
        if not os.path.exists(class_folder_path):
            continue  # Skip if the folder doesn't exist
        for filename in os.listdir(class_folder_path):
            if filename.endswith(".jpeg"):
                image_path = os.path.join(class_folder_path, filename)
                image = Image.open(image_path)
                image_array = np.array(image)
                all_labels.append(class_label)
                image_data.append(image_array)

    image_data = np.array(image_data)
    all_labels = np.array(all_labels)

    # 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(image_data):
        X_train, X_test = image_data[train_index], image_data[test_index]
        y_train, y_test = all_labels[train_index], all_labels[test_index]
        print("Train images shape:", X_train.shape)
        print("Test images shape:", X_test.shape)
        print("Train labels shape:", y_train.shape)
        print("Test labels shape:", y_test.shape)

    return X_train, y_train, X_test, y_test

root_folder = 'sound_datasets/urbansound8k/melspec'
(X_train, y_train), (X_test, y_test) = conv_array(root_folder)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#X_train = X_train.reshape(-1,32, 32, 3)  # reshaping for convnet


classes = ['air_conditioning', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
metadata = pd.read_csv('sound_datasets/urbansound8k/metadata/UrbanSound8K.csv')
metadata.head(10)
sns.countplot(metadata, y="class")
plt.show()


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


# #training parameters
# num_epoch = 25
# batch_size =32
# max_trials = 8 # how many model variations to test?
# max_trial_retrys = 3 # how many trials per variation? (same model could perform differently)
# early_stop = 3 # early stoppping after 3 epochs with no improvement of test data
#
# #Ealry stopping
# EarlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop)
#
# # tuner = RandomSearch(build_model, objective='val_acc', max_trials=max_trials, executions_per_trial=3)
# # tuner.search(x=X_train, y=y_train, epochs=num_epoch, batch_size=batch_size, validation_data=(X_test, y_test))
# #
# # print(tuner.get_best_models()[0].summary())
# # print(tuner.get_best_hyperparameters()[0].values)
# # model = tuner.get_best_models(num_models=1)[0]
# # print (model.summary())
# # # Evaluate the best model.
# # loss, accuracy = model.evaluate(X_test, y_test)
# # print('loss:', loss)
# # print('accuracy:', accuracy)
