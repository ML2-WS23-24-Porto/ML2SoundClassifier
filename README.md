In this report, we build an urban sound classifier by implementing deep learning
techniques. The Dataset used is the Urbansound8k [7] which classifies 8732 audio
clips of a maximum length of 4 s into 10 classes. These classes include the most
common sounds found in urban environments like car noises and sirens.
To create features for the deep learning algorithms, we combine mel spectrograms
and MFCCs (Mel-frequency cepstral coefficients) to create 2D image data to feed
into the neural networks.
The first model is a simple multi-layer perceptron (MLP), designed to recognize
patterns in the image data. The second model is a step up in complexity, it employs
a convolutional Neural Network (CNN), leveraging its ability to recognize spatial
hierarchies in feature maps. Additionally, we explore the possibilities of transfer
learning in a CNN with weights from ImageNet for improved feature extraction.
