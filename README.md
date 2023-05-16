# Neural network model: Recognition of handwritten characters - numbers

`Autor: Stanislav Zelinskyi`

## Description

This neural network model is trained to recognize handwritten digits. The model is based on a convolutional neural network (CNN) and trained on the MNIST dataset.

## Data used

MNIST digits classification dataset, which consists of handwritten digits from 0 to 9, was used to train the model. This is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images. More info can be found at the [MNIST homepage](http://yann.lecun.com/exdb/mnist/).

## Methods and ideas

- The input images were pre-processed by resizing them to 28x28 pixels and normalizing the pixel values between 0 and 1.
- A convolutional neural network (CNN) with multiple convolution, pulling and fully connected layers was used.
- Categorical cross-entropy was used as loss function and Adam was used as optimizer.
- The accuracy metric was used to estimate the accuracy of the model.

## Model accuracy

Based on the results of model training on the MNIST test dataset, an accuracy of 33.73% was achieved. This means that the model correctly classifies 33.73% of the images of handwritten numbers and letters of the alphabet.

## Instructions for use

1. Set the dependencies specified in the `requirements.txt` file using the command<br>
    ` pip install -r requirements.txt `
2. Run the model training script `train.py` to train the model on the MNIST dataset:<br>
    ` python train.py `
3. After model training is complete, the file `mnist_model.h5` will be created, containing the trained model.
4. To use the model on new handwritten character images, place them in a separate folder (e.g. `test_data`).
5. Run the `inference.py` script with the path to the folder containing the test images:<br>
    ` python inference.py --input /path/to/test_data `

The prediction results will be output in CSV format.

## Project structure
#### 1.  model.py: The model created by the create_model function is a Convolutional Neural Network (CNN) for image classification.

Model architecture:

   - Input layer: Conv2D with 32 filters of size (3, 3) and ReLU activation function. Accepts input images of a given input_shape.
   - Pulling layer: MaxPooling2D with window size (2, 2), performs pulling operation on output of previous layer.
   - Convolution layer: Conv2D with 64 filter size (3, 3) and ReLU activation function.
   - Pooling layer: MaxPooling2D with window size (2, 2).
   - Convolution layer: Conv2D with 64 size filters (3, 3) and ReLU activation function.
   - Smoothing layer sequence: Flatten which converts multidimensional data to one dimensional data.
   - Full-coherence layer: Dense with 64 neurons and ReLU activation function.
   - Output layer: Dense with num_classes neurons and Softmax activation function. Generates membership probabilities for each class.
   This model is designed to classify images into num_classes classes, and it uses the ReLU activation function for non-linearity and the Softmax activation function for generating probabilistic predictions.

       The general structure of the model is as follows:
       <table>
       <tr><td>Layer (type)</td><td>Output Shape</td><td>Param</td></tr>
       <tr><td>conv2d (Conv2D)</td><td>(None, 26, 26, 32)</td><td>320</td></tr>
       <tr><td>max_pooling2d (MaxPooling2D)</td><td>(None, 13, 13, 32)</td><td>0</td></tr>
       <tr><td>conv2d_1 (Conv2D)</td><td>(None, 11, 11, 64)</td><td>18496</td></tr>
       <tr><td>max_pooling2d_1 (MaxPooling2D)</td><td>(None, 5, 5, 64)</td><td>0</td></tr>
       <tr><td>conv2d_2 (Conv2D)</td><td>(None, 3, 3, 64)</td><td>36928</td></tr>
       <tr><td>flatten (Flatten)</td><td>(None, 576)</td><td>0</td></tr>
       <tr><td>dense (Dense)</td><td>(None, 64)</td><td>36928</td></tr>
       <tr><td>dense_1 (Dense)</td><td>(None, num_classes)</td><td>xxx</td></tr>
       <tr>Total params: xxx</tr>
       <tr>Trainable params: xxx</tr>
       <tr>Non-trainable params: xxx</tr>
       </table>
        Here None indicates that the batch size may vary depending on the size of the input data.
        
        The total number of model parameters (Total params) will depend on the selected filter size and the number of neurons in the full-coherent layers.
        The Total params of the model will depend on the selected filter size and the number of neurons in the fully connected layers.
        
        Once the model is created it can be compiled and trained on training data with suitable loss functions, optimisers and metrics.


#### 2. model.h5: Trained model saved in H5 format.
#### 3. inference.py: is a Python script that takes a single CLI argument representing the path to a directory containing image samples. It prints the output in CSV format and generates a plot of the accuracy of predicted characters.
#### 4. train.py: The script trains a convolutional neural network model on the MNIST dataset and saves the trained model. It also generates plots for the model's accuracy and loss during training.
#### 5. readme.md: Documentation describing the use of the model.
#### 6. requirements.txt: A file containing a list of Python dependencies.

## Conclusion
This neural network model is designed for handwritten digit recognition. It can be used for automatic character recognition in images and other tasks related to handwriting data processing.
