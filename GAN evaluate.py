import numpy as np
from PIL import Image
import tensorflow as tf
import GAN_models
import os
import losses
from tensorflow.keras.layers import *

g1 = GAN_models.generator()
g1.load_weights('E:/Python files/cycleGAN/codes/weights/final/X40-processed-batchzise8-50-100ep/generator_1_49_9.3786e-03.h5')
g2 = GAN_models.generator()
g2.load_weights('E:/Python files/cycleGAN/codes/weights/final/X40-processed-batchzise8-50-100ep/generator_2_49_8.6672e-03.h5')

def model_A_to_B(g1, g2):
	inputs = Input(shape=(256,256,1))
	generated_imgae = g1(inputs)
	output = g2(generated_imgae)
	model = tf.keras.Model(inputs = inputs, outputs = output)
	return model

model_D_F = model_A_to_B(g1,g2)

IMAGE_PATH_TEST = 'E:/deep learning files/GAN example/SMC/40/processed_input/train/'
LABLE_PATH_TEST = 'E:/deep learning files/GAN example/SMC/40/processed_input/train/'

image_names_test = os.listdir(IMAGE_PATH_TEST)
label_names_test = os.listdir(LABLE_PATH_TEST)

test_file = []
test_label = []

for name in image_names_test:
	test_file.append(IMAGE_PATH_TEST+name)

for name in label_names_test:
	test_label.append(LABLE_PATH_TEST+name)
		
def parse_function(filename, labelname):
	image_contents = tf.io.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_contents)
	#image_converted = tf.image.rgb_to_grayscale(image_decoded)
	image_converted = tf.cast(image_decoded, tf.bfloat16)
	#image_converted = image_converted+20

	label_contents = tf.io.read_file(labelname)
	label_decoded = tf.image.decode_jpeg(label_contents)
	label_converted = tf.cast(label_decoded, tf.bfloat16)
	
	return image_converted, image_converted

test_filenames = tf.constant(test_file)
test_labels = tf.constant(test_label)
test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
test_dataset = test_dataset.map(parse_function)
test_dataset = test_dataset.batch(4)

model_D_F.compile(loss=losses.SSIM_loss, optimizer='Nadam', metrics=[losses.SSIM_loss,'mae'])
model_D_F.evaluate(test_dataset)
