# Multi-class Image Classification on Intel Image Classification Dataset

This is a project that implements multi-class image classification on the [Intel Image Classification Dataset](https://www.kaggle.com/puneet6060/intel-image-classification) from Kaggle, which has 6 balanced classes. The code was written in Jupyter Notebook using Python and it explores different machine learning and deep learning algorithms and techniques to achieve accurate classification.

## Dataset

The dataset contains images of natural scenes. This Data contains around 17k images of size 150x150 distributed under 6 categories.

- **buildings** -> 0
- **forest** -> 1
- **glacier** -> 2
- **mountain** -> 3
- **sea** -> 4
- **street** -> 5

The Train and Test data is separated in each zip files. There are around 14k images in Train, 3k in Test.

## Baseline Model

The baseline model for this project uses Support Vector Machine (SVM) for classification and Histogram of Oriented Gradients (HOG) for feature extraction. The accuracy of this model was 77%.

## Convolutional Neural Network (CNN)

The project also includes a simple CNN implemented in TensorFlow with only 2 convolutional layers. This model achieved an accuracy of 81%. ImageGenerator from Keras was used for preprocessing images for deep nets.

## Visualization and Dimensionality Reduction

Principal Component Analysis (PCA) from sklearn was used for visualization (with only 2 principal components) and for dimensionality reduction (with principal components which have >85% explainability).

## ResNet50

This project includes an implementation of ResNet50 from scratch, including functions for the identity block and the convolutional block, as well as a function that assembles the ResNet.

## Transfer Learning

In addition, transfer learning was employed by using three pretrained models (VGG-16, InceptionV3, and ResNet) that were trained on the Imagenet dataset. As expected, the best results were obtained by the transfer learning models, and the best accuracy was achieved by InceptionV3, with 88%.

## Conclusion

This project demonstrates the effectiveness of using both traditional machine learning techniques and modern deep learning techniques for multi-class image classification tasks. It also shows the power of transfer learning by utilizing pretrained models to achieve high accuracy on this challenging dataset.

