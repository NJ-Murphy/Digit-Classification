# Digit-Classification

In this project, we develop a classification scheme that decides whether a handwritten digit, in the
form of an image, is even or odd. The idea is to use data sets consisiting of pixel values (1 for black,
0 for white) for the images whereby the first column of the data set is the true value (0 to 9) of the
digit shown in the image, and columns 2 to 785 are the pixel values of a given image which can be
reconstructed into a 28 by 28 grid to reveal the image of the handwritten digit.

The classification schemes will be trained on a training set consisting of 2500 images which include
labels of the images and tested on a set consisting of 2500 images which do not contain labels for the
images. We will implement the following classification schemes:
1. Pocket Learning Algorithm
2. Logistic Regression
3. Support Vector Machine
4. Neural Network
5. Random Forests
6. Boosting
7. Regression Trees

We will compare various results of the schemes, the most important being the cross validation error.
We will also compare the accuracy of the schemes for various sizes of the training and validation sets
as well as compare the accuracies both with and without Principal Component Analysis applied to
the training set. We will then apply the trained models to the test data set to produce predictions.
We also provide various plots of the resluts to get more insight into the performance of the schemes.
