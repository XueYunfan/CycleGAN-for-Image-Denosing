import os
from PIL import Image
import numpy as np
import tensorflow as tf
import random

def list_image_files(directory):
	files = sorted(os.listdir(directory))
	return [os.path.join(directory, f) for f in files]

def load_label(path):
	img = Image.open(path)
	img = img.convert('L')
	img = np.array(img)
	img = img.reshape(256, 256, 1)
	#img = (img/127.5)-1
	#print(img)
	#a=input('pause')
	return img.astype('float32')

def load_input(path):
	img = Image.open(path)
	img = img.convert('L')
	img = np.array(img)
	img = img.reshape(256, 256, 1)
	#img = img+40
	#img = (img/127.5)-1
	#print(img)
	#a=input('pause')
	return img.astype('float32')

def save_image(np_arr, path):
	img = np_arr * 127.5 + 127.5
	im = Image.fromarray(img)
	im.save(path)


def Load_images(path_input, path_label, n_images):
	if n_images < 0:
		n_images = float("inf")
	all_A_paths, all_B_paths = list_image_files(path_input), list_image_files(path_label)
	#random.shuffle(all_A_paths)###########
	#random.shuffle(all_B_paths)###########
	images_A, images_B = [], []
	images_A_paths, images_B_paths = [], []
	for path_A, path_B in zip(all_A_paths, all_B_paths):
		img_A, img_B = load_input(path_A), load_label(path_B)
		images_A.append(img_A)
		images_B.append(img_B)
		images_A_paths.append(path_A)
		images_B_paths.append(path_B)
		if len(images_A) > n_images - 1: break

	return {
		'Input': np.array(images_A),
		'Input_paths': np.array(images_A_paths),
		'Label': np.array(images_B),
		'Label_paths': np.array(images_B_paths)
	}

def write_log(callback, names, logs, batch_no):
	"""
	Util to write callback for Keras training
	"""
	for name, value in zip(names, logs):
		summary = tf.Summary()
		summary_value = summary.value.add()
		summary_value.simple_value = value
		summary_value.tag = name
		callback.writer.add_summary(summary, batch_no)
		callback.writer.flush()
