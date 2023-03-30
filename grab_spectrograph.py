import pyscreenshot as ImageGrab
import cv2
import numpy as np

def grab():
    i = 0;
    while True:
        im = ImageGrab.grab(bbox=(1050, 500, 1150, 800))
        im = np.array(im)
        cv2.imwrite('daten/bilder_log/color/j/j'+str(i)+'.jpg', im)
        i+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    grab()


if __name__ == '__main__':
    main()