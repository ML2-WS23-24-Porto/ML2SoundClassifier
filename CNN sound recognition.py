
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


def model1(): #CNN
    model1 = Sequential()
    model1.add(Conv2D(64, (3, 3), padding="same", activation="tanh", input_shape=input_dim))
    model1.add(MaxPool2D(pool_size=(2, 2)))
    model1.add(Conv2D(128, (3, 3), padding="same", activation="tanh"))
    model1.add(MaxPool2D(pool_size=(2, 2)))
    model1.add(Dropout(0.1))
    model1.add(Flatten())
    model1.add(Dense(1024, activation="tanh"))
    model1.add(Dense(10, activation="softmax"))
    return model1

def model2(): #MLP
    model2 = Sequential()
    model2.add(Flatten(input_shape=(28,28)))
    model2.add(Dense(4, activation='relu'))
    model2.add(Dense(4, activation='relu'))
    model2.add(Dense(4, activation='relu'))
    model2.add(Dense(10, activation='softmax'))
    return model2


tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    jit_compile=True,
    name="Adam",
    **kwargs
)


model1 = model1()
model2 = model2()

#optimization algorithm used is Adam
model1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model1.fit(X_train, Y_train, epochs = 90, batch_size = 50, validation_data = (X_test, Y_test))
model1.summary()