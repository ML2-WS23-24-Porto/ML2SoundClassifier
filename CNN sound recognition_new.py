import numpy as np
import os
import pandas as pd
import time, warnings
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow
from PIL import Image
from sklearn.model_selection import KFold
import keras_tuner
from tensorflow import keras
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


#class for plotting
class PlotLearning(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # Plotting
        metrics = [x for x in logs if 'val' not in x]

        f, axs = plt.subplots(1, len(metrics), figsize=(15, 5))

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2),
                        self.metrics[metric],
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2),
                            self.metrics['val_' + metric],
                            label='val_' + metric)

            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()

def normalize(clip):
    normalized_clip = (clip - np.min(clip)) / (np.max(clip) - np.min(clip))
    return normalized_clip

def conv_array(root_folder):
    metadata = pd.read_csv('sound_datasets/urbansound8k/metadata/UrbanSound8K.csv')
    folds = {}
    for class_label in range(1, 11):
        class_folder_path = os.path.join(root_folder, f"fold{class_label}")
        image_data = []
        all_labels = []
        if not os.path.exists(class_folder_path):
            continue  # Skip if the folder doesn't exist
        for filename in os.listdir(class_folder_path):
            if filename.endswith(".jpeg"):
                image_path = os.path.join(class_folder_path, filename)
                image = Image.open(image_path)
                image_array = np.array(image)
                image_array = normalize(image_array)
                new_filename = filename.replace('.jpeg', '.wav')
                print(filename, new_filename)
                
                classID = metadata['classID'][new_filename]
                all_labels.append(classID)
                image_data.append(image_array)
        image_data = np.array(image_data)
        all_labels = np.array(all_labels)
        all_labels = to_categorical(all_labels - 1, num_classes=10)
        folds[f"fold{class_label}"] = [image_data, all_labels ]
    return folds


metadata = pd.read_csv('sound_datasets/urbansound8k/metadata/UrbanSound8K.csv')
root_folder = r"sound_datasets/urbansound8k/melspec"
folds = conv_array(root_folder)
print(folds)
print(folds.shape)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
metric = 'accuracy' #evaluation metric
#metric = tensorflow.keras.metrics.MeanAveragePrecisionMetric(topn=2)
loss= 'categorical_crossentropy' #loss function

#training parameters
num_epoch = 20
batch_size =128
early_stop = 3 # early stoppping after 3 epochs with no improvement of test data

#objective to specify the objective to select the best models, and we use max_trials to specify the number of different models to try.
objective='val_loss'
max_trials = 8 # how many model variations to test?
max_trial_retrys = 3 # how many trials per variation? (same model could perform differently)

metadata = pd.read_csv('sound_datasets/urbansound8k/metadata/UrbanSound8K.csv')
# metadata.head(10)
# sns.countplot(metadata, y="class")
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
        model.add(tensorflow.keras.layers.Dropout(rate=hp.Float('rate', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())

    for i in range(hp.Int('n_connections', 1, 4)):
        model.add(Dense(hp.Choice(f'n_nodes',
                                  values=[128, 256, 512, 1024])))
        model.add(Activation('tanh'))

    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(optimizer=Adam(learning_rate=1e-3), #optimization algorithm used is Adam
                  loss=loss,
                  metrics=[metric])

    return model

#get optimal hyperparameters using
def tuner(X, y, num_epoch, batch_size):
    EarlyStoppingCallback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop)
    tuner = RandomSearch(build_model, objective=objective, max_trials=max_trials, executions_per_trial=max_trial_retrys, metrics=[metric])
    tuner.search(x=X, y=y, epochs=num_epoch, batch_size=batch_size, validation_split=0.1, callbacks=[EarlyStoppingCallback]) #10% is validation data
    best_hyperparameters = tuner.oracle.get_best_trials(1)[0].hyperparameters.values
    return best_hyperparameters

#hyperparameters2 = tuner(X, y, num_epoch, batch_size)
#print(hyperparameters2)

def model_k_cross(hyperparameters, X, y):
    hp = kt.HyperParameters()
    for key, value in hyperparameters.items():
        hp.Fixed(key, value)

    # Merge inputs and targets
    inputs = np.concatenate((X_train, X_test), axis=0)
    targets = np.concatenate((y_train, y_test), axis=0)

    fold_no =1
    for train, test in kf.split(X, y):
        cmodel = build_model(hp)
        EarlyStoppingCallback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop)

        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        cmodel.fit(inputs[train], targets[train], epochs=num_epoch, batch_size=batch_size,
                   callbacks=[EarlyStoppingCallback, PlotLearning()], validation_split=0.1)
        cmodel.summary()

        # evaluation
        history = cmodel.evaluate(X_test, y_test)  # Use X_test and y_test for evaluation
        scores = cmodel.evaluate(X_test, y_test)
        print("Test accuracy:", scores[1])

        # Plot training history
        print(history.keys())
        plt.plot(history['val_loss'])  # Add validation loss if available
        plt.title("Training Loss")
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')
        plt.show()
        fold_no += 1


#creating custom hyperparameters to inspect model performance,inspired by the network we found on kaggle
custom_hyperparameters = {
        'input_units': 224,
        'n_layers': 2,
        'conv_0_units': 64,
        'rate': 0.2,
        'n_connections': 1,
        'n_nodes': 1012,
        'conv_1_units': 128,
    }

#model_k_cross(custom_hyperparameters, X, y)
#model(best_hyperparameters_overall)


