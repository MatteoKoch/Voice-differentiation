import numpy as np
import cv2
import tensorflow as tf
import glob
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten


# 0: anas
# 1: matteo



w = 100
h = 300
color = 3

def train(inputs, output):
    print(inputs.shape, output.shape)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(inputs.shape[1], inputs.shape[2], color)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(output.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    model.fit(inputs, output, batch_size=32, epochs=50, verbose=1)

    return model



def createInputs():
    x_train = []
    y_train = []

    name = []
    outputs = []
    imgs = []
    name.append('a')
    name.append('m')
    j = 0
    # getting the number of images
    for file in glob.glob("daten/bilder_log/color/combined/*.jpg"):
        j += 1
    print(j)

    nj = j/2
    i = 0
    for file in glob.glob("daten/bilder_log/color/combined/*.jpg"):
        outputs.append(math.floor(i/nj))
        print(math.floor(i/nj))
        image = cv2.imread(file)
        image = cv2.resize(image, (w, h))
        imgs.append(image)
        i += 1

    index = np.arange(j)
    np.random.shuffle(index)

    zoo = [[1, 0], [0, 1]]
    for g in range(0, j):
        image = np.array(imgs[index[g]])
        image = np.reshape(image, (w, h, color))
        x_train.append(image)
        y_train.append(zoo[outputs[index[g]]])

    x_train = np.array(x_train)
    print(x_train.shape)
    x_train = x_train/255
    x_train = np.reshape(x_train, (j, w, h, color))

    y_train = np.array(y_train)
    print(y_train.shape)
    y_train = np.reshape(y_train, (j, 2))

    return x_train, y_train


def main():
    x, y = createInputs()
    model = train(x, y)
    model.save('models/cnn_log_color')
    cv2.waitKey(0)


if __name__ == '__main__':
    main()

