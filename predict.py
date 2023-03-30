import cv2
import numpy as np
from tensorflow import keras
import glob
import math

w = 100
h = 300

def predictedname(prediction):
    if prediction[0][0] == 1:
        return 'anas'
    if prediction[0][1] == 1:
        return 'david'
    if prediction[0][2] == 1:
        return 'jasin'
    return 'matteo'


def predict():
    model = keras.models.load_model('models/cnn_log_color_a_d_j_m')

    for file in glob.glob("daten/bilder_log/color/a/*.jpg"):
        image = cv2.imread(file)
        im = np.array(image)
        im = np.reshape(im, (1, w, h, 3))
        print(model.predict(im))
        print(predictedname(model.predict(im)))



def main():

    predict()


if __name__ == '__main__':
    main()