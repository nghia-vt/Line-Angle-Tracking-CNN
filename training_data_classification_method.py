'''
Using NVIDIA model
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

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def preprocess_image(img):
  
    # Resize the image to the input shape used by the network model
    scale = float(IMAGE_WIDTH) / img.shape[1]
    w_scale = int(img.shape[1] * scale)
    h_scale = int(img.shape[0] * scale)
    img = cv2.resize(img, (w_scale, h_scale), cv2.INTER_AREA)

    # Crop the image if neccesary
    img = img[(h_scale - IMAGE_HEIGHT) : h_scale, 0 : w_scale]

    # Convert the image from RGB to YUV (This is what the NVIDIA model does)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    return img

def plot_data_histogram(num_classes, arr_data):

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

def Nvidia_model():
    model = Sequential(name="Nvidia_Model")

    # elu: Exponential Linear Unit, similar to leaky Relu

    # Convolution layers (INPUT_SHAPE = (66, 200, 3) for NVIDIA model)
    model.add(Conv2D(24, (5,5), strides=(2,2), input_shape=INPUT_SHAPE, activation='elu'))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation='elu'))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation='elu'))
    model.add(Conv2D(64, (3,3), activation='elu'))
    model.add(Conv2D(64, (3,3), activation='elu'))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))

    # Output layer
    model.add(Dense(1))

    optimizer = Adam(lr=1e-4) # learning rate
    model.compile(loss='mse', optimizer=optimizer, metrics=["mean_squared_error"])

    return model

#-------------------------------------------------------

dataset = '/content/drive/MyDrive/Colab Notebooks/line_tracking_cnn/dataset' 
# initialize the data and labels
print("[INFO] Loading images...")
data = []
labels = []

# grab the image paths
imagePaths = list(paths.list_images(dataset))

# loop over the input images
for idx, imagePath in enumerate(imagePaths):

    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = preprocess_image(image)
    data.append(image)

    # extract the label from the image path and update the labels list    
    imageFileName = imagePath.split(os.path.sep)[-1]
    imageName = imageFileName.split('.')[0]
    label = imageName.split('_')[-1]
    labels.append(int(label))

    # Print progress status
    sys.stdout.write("\r\tProgress: {0}% ... ".format(round(100*idx/(len(imagePaths) - 1), 1)))
    sys.stdout.flush()

print("\n\tCompleted.")

#-------------------------------------------------------

data_arr = np.array(data, dtype=np.float)
labels_arr = np.array(labels, dtype=np.float)
print(data_arr.shape)
 
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data_arr, labels_arr, test_size=0.2, random_state=42)
trainX, trainY = shuffle(trainX, trainY)
testX, testY = shuffle(testX, testY)

# plot train data histogram
plot_data_histogram(181, trainY)

#-------------------------------------------------------

# initialize the number of epochs to train for, initial learning rate, and batch size
EPOCHS = 30
BATCH_SIZES = 16

# initialize the model
print("[INFO] Compiling model...")
model = Nvidia_model()

# view architecture of the model
model.summary()
 
# train the network
print("[INFO] Training network...")
H = model.fit(trainX, trainY, batch_size=BATCH_SIZES, validation_data=(testX, testY), epochs=EPOCHS, verbose=2)

# summarize history for loss
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# save the model to disk
print("[INFO] Saving model...")
model.save('/content/drive/MyDrive/Colab Notebooks/line_tracking_cnn/model.h5')

print("Completed")

#-------------------------------------------------------

# test prediction, reference this to use the model
from keras.models import load_model
model = load_model('/content/drive/MyDrive/Colab Notebooks/line_tracking_cnn/model.h5')

dataset = '/content/drive/MyDrive/Colab Notebooks/line_tracking_cnn/dataset' 

# grab the image paths
imagePaths = list(paths.list_images(dataset))
imagePaths = imagePaths[0 : 200] # test 201 image

error = []
# loop over the input images
for idx, imagePath in enumerate(imagePaths):
    imageFileName = imagePath.split(os.path.sep)[-1]
    imageName = imageFileName.split('.')[0]
    angle_label = imageName.split('_')[-1]
    angle_label = int(angle_label)

    image = cv2.imread(imagePath)
    image = preprocess_image(image)
    image = np.array(image, dtype=np.float)
    image = image.reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

    prediction = model.predict(image)
    error.append(np.abs(angle_label - prediction[0][0]))
    
x = [i for i in range(len(error))]
plt.plot(x, error)
plt.title('Error angle')
plt.ylabel('error')
plt.xlabel('x')
plt.legend('e', loc='upper right')
plt.show()
