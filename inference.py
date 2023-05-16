import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import create_model

# Parsing command-line arguments
parser = argparse.ArgumentParser(description='Image samples print script')
parser.add_argument('input_dir', type=str, help='Path to the input directory')
args = parser.parse_args()

# Getting the list of image files in the specified directory
image_files = [file for file in os.listdir(args.input_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]

# Creating the model
input_shape = (28, 28, 1)  # Set the correct input image size
num_classes = 10  # Set the correct number of classes
model = create_model(input_shape, num_classes)

# Loading the pre-trained model
model.load_weights('mnist_model.h5')  # Replace 'mnist_model.h5' with the path to your model

# Opening the CSV file for writing the output
output_file = 'output.csv'
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Iterating over each image
    correct_predictions = 0

    for image_file in image_files:
        image_path = os.path.join(args.input_dir, image_file)
        true_label = int(image_file.split('_')[0])

        # Loading and preprocessing the image for the model
        image = Image.open(image_path).convert('L')
        image = image.resize(input_shape[:2])
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=-1)
        image_array = np.expand_dims(image_array, axis=0)

        # Predicting the character in the image using the model
        predicted_class = np.argmax(model.predict(image_array))

        # Writing the output to the CSV file
        writer.writerow([predicted_class, image_path])

        # Checking the correctness of the prediction
        if predicted_class == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(image_files)
    print('Accuracy:', accuracy)
    print('Correct Prediction:', correct_predictions)
    print('Sum of image files', len(image_files))

# Plotting the accuracy of predicted characters
plt.bar(['Accuracy'], [accuracy])
plt.title('Accuracy of Predicted Characters')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.savefig('accuracy_plot.png')  # Saving the plot as an image
plt.show()
