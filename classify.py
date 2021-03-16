from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
from imutils import paths
import imutils
import cv2
import random
import pickle

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained model")
ap.add_argument("-t", "--testset", required=True, help="path to test images")
args = vars(ap.parse_args())

# Resize parmeters (RESIZE should be the same as used in training)
RESIZE = 28

# Grab the image paths and randomly shuffle them
image_paths = sorted(list(paths.list_images(args["testset"])))

random.seed()
random.shuffle(image_paths)

# Read labels for classes to recognize
mlb = pickle.loads(open(args["model"] + ".lbl", "rb").read())
labels = mlb.classes_

# Load the trained network
model = load_model(args["model"] + ".h5")
print(model.summary())
for image_path in image_paths:
    # Load the image
    image = cv2.imread(image_path)
    output = imutils.resize(image, width=400)

    # Preprocess the image
    image = cv2.resize(image, (RESIZE, RESIZE))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Classify the input image
    predict = model.predict(image)[0]

    # Find the winner classes and the probabilities
    probabilities = predict * 100
    top = np.argsort(probabilities)[::-1][:2]

    # loop over the indexes of the high confidence class labels
    for (i, j) in enumerate(top):
    
        # build the label and draw the label on the image
        label = "{}: {:.2f}%".format(labels[j], probabilities[j])
        cv2.putText(output, label, (10, (i * 30) + 25), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
    # show the probabilities for each of the individual labels
    for (label, p) in zip(labels, probabilities):
        print("{}: {:.2f}% " "".format(label, p))
        
    # show the output image
    cv2.imshow("Output", output)
    cv2.waitKey(0)