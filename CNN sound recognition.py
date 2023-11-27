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
from tensorflow.keras.callbacks import EarlyStopping
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


def normalize(clip):
    normalized_clip = (clip - np.min(clip)) / (np.max(clip) - np.min(clip))
    return normalized_clip

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
                image_array = normalize(image_array)
                all_labels.append(class_label)
                image_data.append(image_array)

    image_data = np.array(image_data)
    all_labels = np.array(all_labels)
    all_labels = to_categorical(all_labels - 1, num_classes=10)

    return image_data, all_labels

root_folder = 'sound_datasets/urbansound8k/melspec'
X, y = conv_array(root_folder)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#X_test = X_test.astype('float32')
#X_train = X_train.reshape(-1,32, 32, 3)  # reshaping for convnet


classes = ['air_conditioning', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
metadata = pd.read_csv('sound_datasets/urbansound8k/metadata/UrbanSound8K.csv')
metadata.head(10)
sns.countplot(metadata, y="class")
#plt.show()


#Building a hypermodel:
# function to build a hypermodel
# takes an argument from which to sample hyperparameters
def build_model(hp):
    model = Sequential()

    model.add(Conv2D(hp.Int('input_units', min_value=32, max_value=256, step=32), (3, 3), input_shape=X.shape[1:]))
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
                  metrics=["acc"])

    return model

#training parameters
num_epoch = 1
batch_size =32
max_trials = 8 # how many model variations to test?
max_trial_retrys = 3 # how many trials per variation? (same model could perform differently)
early_stop = 3 # early stoppping after 3 epochs with no improvement of test data

#10 Fold cross validation to obtain best hyperparameters
def k_fold_cross_validation(X, y, num_epoch, batch_size):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    best_hyperparameters_per_fold = []
    total_hyperparameters = {
        'input_units': 0,
        'n_layers': 0,
        'conv_0_units': 0,
        'rate': 0,
        'n_connections': 0,
        'n_nodes': 0
    }

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"Training on fold {fold + 1}")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        EarlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop)
        tuner = RandomSearch(build_model, objective='val_acc', max_trials=max_trials, executions_per_trial=max_trial_retrys,
                             directory=f'tuner_dir_fold_{fold}', project_name=f'project_fold_{fold}', metrics=["acc"])
        tuner.search(x=X_train, y=y_train, epochs=num_epoch, batch_size=batch_size,
                     validation_data=(X_val, y_val), callbacks=[EarlyStoppingCallback])

        best_hyperparameters = tuner.oracle.get_best_trials(1)[0].hyperparameters.values
        best_hyperparameters_per_fold.append(best_hyperparameters)

    for fold_hyperparameters in best_hyperparameters_per_fold:
        for key, value in fold_hyperparameters.items():
            total_hyperparameters[key] += value

    # Calculate the average hyperparameter values
    num_folds = len(best_hyperparameters_per_fold)
    average_hyperparameters = {key: value / num_folds for key, value in total_hyperparameters.items()}

    return best_hyperparameters_per_fold, average_hyperparameters


best_hyperparameters_per_fold, best_hyperparameters_overall = k_fold_cross_validation(X, y, num_epoch, batch_size)
print(best_hyperparameters_per_fold)
print(best_hyperparameters_overall)

final_model = build_model(
    input_units=best_hyperparameters_overall['input_units'],
    n_layers=best_hyperparameters_overall['n_layers'],
    conv_units=best_hyperparameters_overall['conv_0_units'],  # Adjust as needed based on your model architecture
    rate=best_hyperparameters_overall['rate'],
    n_connections=best_hyperparameters_overall['n_connections'],
    n_nodes=best_hyperparameters_overall['n_nodes']
)

# Train the final model on the entire dataset
EarlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop)
final_model.fit(X, y, epochs=num_epoch, batch_size=batch_size, callbacks=[EarlyStoppingCallback], validation_split=0.1,  metrics=["acc"])
final_model.summary()

#evaluate best model
validation_data = X_test, y_test
history = final_model.evaluate(X, y)
scores = final_model.evaluate(validation_data)
print("Test accuracy:",scores[1])


# plot training history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Test accuracy:",scores[1])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss','test_loss'], loc='upper left')
plt.show()

