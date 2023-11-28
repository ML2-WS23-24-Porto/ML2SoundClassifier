# ML2SoundClassifier
Classify urban sounds


choices for hypertuning (Keras):
Bayesian optimization, Random search and Hyperband

ideas for metrics:
- Classification Mean Average Precision (cmAP)

Available metrics
Accuracy metrics
Accuracy class
BinaryAccuracy class
CategoricalAccuracy class
SparseCategoricalAccuracy class
TopKCategoricalAccuracy class
SparseTopKCategoricalAccuracy class
Probabilistic metrics
BinaryCrossentropy class
CategoricalCrossentropy class
SparseCategoricalCrossentropy class
KLDivergence class
Poisson class
Regression metrics
MeanSquaredError class
RootMeanSquaredError class
MeanAbsoluteError class
MeanAbsolutePercentageError class
MeanSquaredLogarithmicError class
CosineSimilarity class
LogCoshError class
Classification metrics based on True/False positives & negatives
AUC class
Precision class
Recall class
TruePositives class
TrueNegatives class
FalsePositives class
FalseNegatives class
PrecisionAtRecall class
SensitivityAtSpecificity class
SpecificityAtSensitivity class




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

The one on kaggle uses:
CNN 2D with 64 units and tanh activation.
MaxPool2D with 2*2 window.
CNN 2D with 128 units and tanh activation.
MaxPool2D with 2*2 window.
Dropout Layer with 0.2 drop probability.
DL with 1024 units and tanh activation.
DL 10 units with softmax activation.
Adam optimizer with categorical_crossentropy loss function.



Possibility of using transfer learning??