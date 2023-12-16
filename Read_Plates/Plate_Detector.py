import numpy as np
import cv2
import pytesseract
import skimage

class Plate_Detector:
    def grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def apply_threshold(self, img):
        return cv2.threshold(img, 170, 255, cv2.THRESH_BINARY_INV)[1]