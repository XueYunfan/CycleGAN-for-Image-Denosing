from skimage.util import random_noise
from skimage.filters import gaussian
from skimage.exposure import adjust_gamma
import cv2
import os
import numpy as np

def synthetic(img):
	img = random_noise(img,var=0.0005)
	img = gaussian(img,sigma=0.8)
	img = adjust_gamma(img,0.8)
	return img*255

IMAGE_PATH = 'E:/deep learning files/GAN example/SMC/20/synthetic_test/tcps_tests/'
SAVE_PATH = 'E:/deep learning files/GAN example/SMC/20/synthetic_test/synthetic/'

img_names = os.listdir(IMAGE_PATH)
for name in img_names:
	img = cv2.imread(IMAGE_PATH+name)
	cv2.imwrite(SAVE_PATH+name, cropped, [int(cv2.IMWRITE_JPEG_QUALITY),100])
