from keras import optimizers, losses
from keras.datasets import mnist
from keras.utils import to_categorical
from model import create_model
import matplotlib.pyplot as plt


# Loading the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocessing and normalizing the data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Creating the model
input_shape = (28, 28, 1)
num_classes = 10
model = create_model(input_shape, num_classes)

# Setting hyperparameters
learning_rate = 0.001
batch_size = 128
num_epochs = 10

# Compiling the model
model.compile(optimizer=optimizers.Adam(learning_rate),
              loss=losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# Training the model
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test, y_test), validation_split=0.1)

# Saving the model
model.save('mnist_model.h5')

# Plotting the accuracy and loss during training
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.savefig('training_metrics.png')  # Saving the plots as PNG images
plt.show()
