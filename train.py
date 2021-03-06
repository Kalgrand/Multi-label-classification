from keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import pickle
import nn

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-e", "--epoch", required=True, help="number of epoch")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

VERBOSE = 1
ROTATION_RANGE = 30
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
ZOOM_RANGE = 0.2
TEST_SIZE = 0.2

# Batch size
BS = 32

# Resize images to the dimension of RESIZE x RESIZE.
RESIZE = 28

# Images color depth (3 for RGB, 1 for grayscale)
IMG_DEPTH = 3
EPOCHS = int(args["epoch"])

# Grab the image paths and randomly shuffle them
image_paths = sorted(list(paths.list_images(args["dataset"])))

random.seed()
random.shuffle(image_paths)

# Initialize the data and labels
data = []
labels = []

# Loop over the input images
for image_path in image_paths:
    # Load the image and pre-process it
    image = cv2.imread(image_path)
    image = cv2.resize(image, (RESIZE, RESIZE))
    image = img_to_array(image)

    # Extract the class label from the image path
    label = image_path.split(os.path.sep)[-2].split("_")

    # Store image and labels in lists
    data.append(image)
    labels.append(label)

# Scale the raw pixel intensities to the [0, 1] range
data = np.array(data, dtype="float") / 255.0  # rgb (255,255,255)
labels = np.array(labels)

# Binarize the labels using scikit-learn's special multi-label
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# Save the text labels to disk as pickle
f = open(args["model"] + ".lbl", "wb")
f.write((pickle.dumps(mlb)))
f.close()

# Determine number of classes
no_classes = len(mlb.classes_)

# Partition the data into training and validating splits
(train_data, valid_data, train_labels, valid_labels) = train_test_split(data, labels, test_size=TEST_SIZE)
aug = ImageDataGenerator(rotation_range=ROTATION_RANGE, width_shift_range=WIDTH_SHIFT_RANGE, height_shift_range=HEIGHT_SHIFT_RANGE, zoom_range=ZOOM_RANGE,
                         horizontal_flip=True)

# Initialize the model
print("Compiling model...")
#model = nn.FullyConnectedForImageClassisfication.build(width=RESIZE, height=RESIZE, depth=IMG_DEPTH, hidden_units=HIDDEN_UNITS, classes=no_classes)
model = nn.SmallerVGGNet.build(width=RESIZE, height=RESIZE, depth=IMG_DEPTH, classes=no_classes)
#model = nn.LeNet5.build(width=RESIZE, height=RESIZE, depth=IMG_DEPTH, classes=no_classes)
model.summary()

# Purpose of loss functions is to compute the quantity that a model should seek to minimize during training
# Select the loss function
if no_classes == 2:
    loss = "binary_crossentropy"
else:
    loss = "categorical_crossentropy"

# Compile model
model.compile(loss=loss, optimizer="Adam", metrics=[metrics.mae, "accuracy"])

# Train the network
H = model.fit_generator(aug.flow(train_data, train_labels, batch_size=BS), validation_data=(valid_data, valid_labels),
                        epochs=EPOCHS, steps_per_epoch=len(train_data) // BS, verbose=VERBOSE)

# Save model to disk
model.save(args["model"] + ".h5")

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig(args["model"] + ".png")