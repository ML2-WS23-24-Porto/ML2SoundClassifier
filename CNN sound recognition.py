import os
import json

import IPython.display as ipd
import librosa
import resampy
import librosa.display
import pandas as pd
import numpy as np
import time, warnings
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
import keras_tuner

from tqdm import tqdm
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.trial import Trial

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Dropout,
    Flatten,
    Conv2D,
    MaxPool2D,
)
(train_images, train_labels), (test_images, test_labels) = #load dataset

classes = ['air_conditioning', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

sns.countplot(metadata, y="class")
plt.show()

X=np.array(feature_df['feature'].tolist())
y=np.array(feature_df['class'].tolist())

num_classes = len(set(y))

labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=101)
X_val,X_test,y_val,y_test = train_test_split(X_test,y_test,test_size=0.50,random_state=101)

print(num_classes)