'''
Using LeNet model
Should run this on https://colab.research.google.com/
'''

# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D 
from keras.layers.core import Activation, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys

#-------------------------------------------------------

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 28, 28, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def preprocess_img(img):
  
    # Resize the image to the input shape used by the network model
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

    return img

def plot_classes_histogram(num_classes, arr_data):

    items = [i for i in range(num_classes)]
    counts = [0] * num_classes

    for item in arr_data:
        idx = items.index(item)
        counts[idx] = counts[idx] + 1

    fig_train = plt.figure()
    axes_train = fig_train.add_axes([0, 0, 1, 1])
    axes_train.bar(items, counts)

    plt.xlabel('degree')
    plt.ylabel('count')
    plt.title('Train data histogram')
    plt.show()

def LeNet_model(init_lr, epochs):
    model = Sequential(name="LeNet_Model")

    # first set of CONV => RELU => POOL layers
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=INPUT_SHAPE))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(3))
    model.add(Activation("softmax"))

    opt = Adam(lr=init_lr, decay=init_lr / epochs)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model

#-------------------------------------------------------

dataset = '/content/drive/MyDrive/Colab Notebooks/line_tracking_cnn/dataset' 
# initialize the data and labels
print("[INFO] Loading images...")
data = []
labels = []

# grab the image paths
imagePaths = list(paths.list_images(dataset))
imageCount = len(imagePaths)

# loop over the input images
for idx, imagePath in enumerate(imagePaths):

    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = preprocess_img(image)
    data.append(image)

    # extract the class label from the image path and update the labels list    
    imageFileName = imagePath.split(os.path.sep)[-1]
    imageName = imageFileName.split('.')[0]
    label = imageName.split('_')[-1]

    if int(label) < 80:
      lb = 1 # right
    elif int(label) > 110:
      lb = 2 # left
    else:
      lb = 0 # foward

    labels.append(lb)

    # Print progress status
    sys.stdout.write("\r\tProgress: {0}% ... ".format(round(100*idx/(imageCount - 1), 1)))
    sys.stdout.flush()

print("\n\tCompleted.")

#-------------------------------------------------------

# scale the raw pixel intensities to the range [0, 1]
data_arr = np.array(data, dtype=np.float) / 255.0
labels_arr = np.array(labels)
print(data_arr.shape)
 
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data_arr, labels_arr, test_size=0.2, random_state=42)
trainX, trainY = shuffle(trainX, trainY)
testX, testY = shuffle(testX, testY)

# plot train data histogram
plot_classes_histogram(3, trainY)

# one-hot encoding labels data
trainY = to_categorical(trainY, num_classes=3)
testY = to_categorical(testY, num_classes=3)

#-------------------------------------------------------

# initialize the number of epochs to train for, initial learning rate, and batch size
INIT_LR = 1e-3
EPOCHS = 15
BS = 32

# initialize the model
print("[INFO] Compiling model...")
model = LeNet_model(INIT_LR, EPOCHS)
model.summary() # view architecture of the model
 
# train the network
print("[INFO] Training network...")
H = model.fit(trainX, trainY, batch_size=BS,
              validation_data=(testX, testY),
              epochs=EPOCHS, verbose=2)

# summarize history for accuracy
plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# save the model to disk
print("[INFO] Serializing network...")
model.save('/content/drive/MyDrive/Colab Notebooks/line_tracking_cnn/model_class.h5')

print("Completed")

#-------------------------------------------------------

from keras import metrics 
score = model.evaluate(testX, testY, verbose=0)
print('Test loss: %.4f'% score[0])
print('Test accuracy %.4f'% score[1])

#-------------------------------------------------------

# test prediction, reference this to use the model
from keras.models import load_model
model = load_model('/content/drive/MyDrive/Colab Notebooks/line_tracking_cnn/model_class.h5')

dataset = '/content/drive/MyDrive/Colab Notebooks/line_tracking_cnn/dataset' 

# grab the image paths
imagePaths = list(paths.list_images(dataset))
imagePaths = imagePaths[0 : 100]

# loop over the input images
for idx, imagePath in enumerate(imagePaths):
    imageFileName = imagePath.split(os.path.sep)[-1]
    imageName = imageFileName.split('.')[0]
    angle_label = imageName.split('_')[-1]
    angle_label = int(angle_label)

    image = cv2.imread(imagePath)
    image = preprocess_img(image)
    image = np.array(image, dtype="float") / 255.0
    image = image.reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)

    prediction = np.argmax(model.predict(image))
    if (prediction == 1 and angle_label < 80) or \
        (prediction == 2 and angle_label > 110) or \
        (prediction == 0 and 80 <= angle_label <= 110):
        print("True")
    else:
        # print("----False----")
        print(angle_label)
        img = cv2.imread(imagePath)
        plt.imshow(img)
        plt.show()

print("Completed.")
