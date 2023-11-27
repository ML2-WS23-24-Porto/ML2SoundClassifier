import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

#clear previous sessions
tf.keras.backend.clear_session()

#download and prepare the dataset
(ds_train, ds_test), ds_info = tfds.load('mnist', split=['train', 'test'], shuffle_files=False, as_supervised= True, with_info=True,)
print(ds_info)

#normalize the data
def normalize_img(image, label):
    """Normalizes images: `uint8` ->
    `float32`."""
    return tf.cast(image, tf.float32) / 255., label

#put the training and test data in batches of 128 labeled images and Scales the image pixes within the range [0,1]
ds_train = ds_train.map(normalize_img) #The map function is used to apply a given function (normalize_img in this case) to each element of the dataset.
ds_train = ds_train.batch(128)
ds_test = ds_test.map(normalize_img)
ds_test = ds_test.batch(128)



def model2(): #MLP
    model2 = Sequential()
    model2.add(Flatten(input_shape=(28,28)))
    model2.add(Dense(4, activation='relu'))
    model2.add(Dense(4, activation='relu'))
    model2.add(Dense(4, activation='relu'))
    model2.add(Dense(10, activation='softmax'))
    return model2

#define the neural network structure
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)), tf.keras.layers.Dense(4, activation='relu'), tf.keras.layers.Dense(4, activation='relu'), tf.keras.layers.Dense(4, activation='relu'), tf.keras.layers.Dense(10, activation='softmax')])
#display neural network structure
model.summary()

#compile model and define training
#learning rate 0.01, Stochastic gradient desent, cross entropy loss function
model.compile(optimizer=tf.keras.optimizers.SGD(0.01), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],)

#train the model
epochs = 20
batch_size = 266
validation_data = ds_test
history = model.fit(ds_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, )
scores = model.evaluate(ds_test)
print("Test accuracy:",scores[1])

# plot training history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Test accuracy:",scores[1])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss','test_loss'], loc='upper left')
plt.show()

#question one: doubling the amount of epochs aprrox.doubles the running time but increases the accruacy
#
#
#
#
