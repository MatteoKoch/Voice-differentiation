import pyscreenshot as ImageGrab
import cv2
import numpy as np
from tensorflow import keras


w = 100
h = 300


def predictedname(prediction):
    if prediction[0][0] == 1:
        return 'anas'
    return 'matteo'


def predict():
    model = keras.models.load_model('models/cnn_log_color')
    while True:
        im = ImageGrab.grab(bbox=(1050, 500, 1150, 800))
        im = np.array(im)
        im = np.reshape(im, (1, w, h, 3))
        print(model.predict(im))
        print(predictedname(model.predict(im)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    predict()


if __name__ == '__main__':
    main()