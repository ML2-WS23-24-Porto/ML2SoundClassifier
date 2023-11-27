# ML2SoundClassifier
Classify urban sounds


ideas for metrics:
- Classi¯cation Mean Average Precision (cmAP)




Hiatt22 compiled the dataset used in this work and proposed an approach. This
dataset contains a subset of records, labeled by species from California and Nevada.
A pre-processing methodology was used where a data generator combines Mel
Spectrograms with MFCC into a single two-dimensional array. The deep learning
model implemented is a CNN with three stacks of 2D Convolutions using ReLU
activation and MaxPooling layers followed by dropout layers with a rate of 0.2. The
model architecture is topped by a global average pooling layer, which is followed by a
fully connected dense layer using the softmax activation. The accuracy metric is used
to evaluate the performance of the models on a 33% holdout methodology. The
documented results correspond to the model from the epoch with the minimum
validation loss value. It achieved a validation accuracy of 19.27% and a training
accuracy of 20.44%.22

Possibility of using transfer learning??