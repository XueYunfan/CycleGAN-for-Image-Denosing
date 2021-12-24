import numpy as np
from PIL import Image
import tensorflow as tf
import GAN_models
import os
from tensorflow.keras.layers import *

IMAGE_PATH = 'E:/deep learning files/SEM enhancement/cropped input/train/'
SAVE_PATH = 'E:/deep learning files/SEM enhancement/cropped input/train_enhanced/'

img_names = os.listdir(IMAGE_PATH)
#img2 = Image.open('E:/Python files/cycleGAN/ep2.jpg')
#img2 = np.array(img2)
#img2 = img2.astype('float32')
#img2 = tf.reshape(img2,(1,160,160,1))

#print(tf.image.ssim(img,img2,max_val=1))
#a=input('a')

def model_A_to_B(g1, g2):
	inputs = Input(shape=(960,1280,1))
	generated_imgae = g1(inputs)
	output = g2(generated_imgae)
	model = tf.keras.Model(inputs = inputs, outputs = output)
	return model

gan = GAN_models.generator()
gan.load_weights('E:/Python files/cycleGAN/codes/weights/SEM_denosing/generator_1_49_4.2626e-03.h5')
gan2 = GAN_models.generator()
#gan2.load_weights('E:/Python files/cycleGAN/codes/weights/6_17/generator_2_9_2.6817e-03.h5')

model_D_F = model_A_to_B(gan,gan2)

def generate_img(path1,path2,size1=256,size2=256):
	
	img = Image.open(path1)
	#r, g, b = img.split()
	img = img.convert('L')
	img = np.array(img)
	#img = img[:256,:256]
	#print(img)
	#a=input()
	img = img.astype('float32')
	#img = img+10
	#img = img[:1600,:1600]
	img = tf.reshape(img,(1,size1,size2,1))
	
	prediction = gan.predict(img)
	#prediction = (prediction+1)*127.5
	prediction = np.reshape(prediction,(size1,size2))
	#print(prediction)

	#R = Image.fromarray(np.uint8(np.zeros((size1,size2))))
	#B = Image.fromarray(np.uint8(np.zeros((size1,size2))))
	G = Image.fromarray(np.uint8(prediction))
	#img = Image.merge('RGB', (R, G, B))

	G.save(path2, quality=95)
	#img.show()

paht1 = 'E:/deep learning files/SEM enhancement/test input/5_m01.tif'#'E:/deep learning files/GAN example/SMC/20/input/test/0-842.jpg')
paht2 = 'E:/Python files/cycleGAN/4_m01_SSIM2.jpg'
#generate_img(paht1, paht2)

for name in img_names:
	if name == 'new':
		continue
	generate_img(IMAGE_PATH+name,SAVE_PATH+name)
