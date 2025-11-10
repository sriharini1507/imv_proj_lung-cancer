# Configure local dataset paths
import os
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
train_folder = os.path.join(DATASET_DIR, 'train')
validate_folder = os.path.join(DATASET_DIR, 'valid')
test_folder = os.path.join(DATASET_DIR, 'test')

if not all(os.path.isdir(path) for path in [train_folder, validate_folder, test_folder]):
    raise FileNotFoundError(
        "Expected dataset folders 'train', 'valid', and 'test' inside the 'dataset' directory."
    )

# Import necessary libraries
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout2D, Activation, Lambda, Flatten, LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import utils

print("Libraries Imported")

# Set the image size for resizing
IMAGE_SIZE = (350, 350)

# Initialize the image data generators for training and testing
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Define hyperparameters (override with environment variables if needed)
batch_size = int(os.environ.get('BATCH_SIZE', 8))
epochs = int(os.environ.get('EPOCHS', 5))

# Create the training data generator
train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    color_mode="rgb",
    class_mode='categorical'
)

# Create the validation data generator
validation_generator = test_datagen.flow_from_directory(
    validate_folder,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    color_mode="rgb",
    class_mode='categorical'
)

steps_per_epoch_default = math.ceil(train_generator.samples / batch_size) if train_generator.samples else 1
validation_steps_default = math.ceil(validation_generator.samples / batch_size) if validation_generator.samples else 1

steps_per_epoch = int(os.environ.get('STEPS_PER_EPOCH', steps_per_epoch_default))
validation_steps = int(os.environ.get('VALIDATION_STEPS', validation_steps_default))

# Set up callbacks for learning rate reduction, early stopping, and model checkpointing
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=5, verbose=2, factor=0.5, min_lr=0.000001)
early_stops = EarlyStopping(monitor='loss', min_delta=0, patience=6, verbose=2, mode='auto')
model_checkpoint_path = os.path.join(BASE_DIR, 'best_model.weights.h5')
checkpointer = ModelCheckpoint(filepath=model_checkpoint_path, verbose=2, save_best_only=True, save_weights_only=True)

# Define the number of output classes
OUTPUT_SIZE = 4

# Load a pre-trained model (Xception) without the top layers and freeze its weights
pretrained_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])
pretrained_model.trainable = False

# Create a new model with the pre-trained base and additional layers for classification
model = Sequential()
model.add(pretrained_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(OUTPUT_SIZE, activation='softmax'))

print("Pretrained model used:")
pretrained_model.summary()

print("Final model created:")
model.summary()

# Compile the model with an optimizer, loss function, and evaluation metric
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with the training and validation data generators
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[learning_rate_reduction, early_stops, checkpointer],
    validation_data=validation_generator,
    validation_steps=validation_steps
)

print("Final training accuracy =", history.history['accuracy'][-1])
print("Final testing accuracy =", history.history['val_accuracy'][-1])

# Function to display training curves for loss and accuracy
def display_training_curves(training, validation, title, subplot):
    if subplot % 10 == 1:
        plt.subplots(figsize=(10, 10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model ' + title)
    ax.set_ylabel(title)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])

# Display training curves for loss and accuracy
display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)
display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)

# Save the trained model
trained_model_path = os.path.join(BASE_DIR, 'trained_lung_cancer_model.h5')
model.save(trained_model_path)

# Function to load and preprocess an image for prediction
from tensorflow.keras.preprocessing import image
import numpy as np

def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image like the training images
    return img_array

class_labels = list(train_generator.class_indices.keys())

# Make sample predictions using a few images from the local test set
sample_images = []
for root, _, files in os.walk(test_folder):
    for file_name in files:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            sample_images.append(os.path.join(root, file_name))
            if len(sample_images) >= 4:
                break
    if len(sample_images) >= 4:
        break

for img_path in sample_images:
    img = load_and_preprocess_image(img_path, IMAGE_SIZE)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_class]
    actual_label = os.path.basename(os.path.dirname(img_path))
    print(f"Image: {img_path}")
    print(f"Actual folder: {actual_label}")
    print(f"Predicted class: {predicted_label}")

    plt.imshow(image.load_img(img_path, target_size=IMAGE_SIZE))
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()
