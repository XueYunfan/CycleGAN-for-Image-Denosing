import os
import datetime
import numpy as np
import tqdm
import pandas as pd

import load_images
import losses
import GAN_models

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import utils
from tensorflow.keras.optimizers import RMSprop

BASE_DIR = 'weights/'

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

def save_all_weights(d, g, epoch_number, g_current_loss, d_current_loss, name):
	now = datetime.datetime.now()
	save_dir = os.path.join(BASE_DIR, 'SEM_{}_{}_res_SSIM_paired_3'.format(now.month, now.day))
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	g.save_weights(os.path.join(save_dir, 'generator_{}_{}_{:.4e}.h5'.format(name,epoch_number, g_current_loss)), True)
	d.save_weights(os.path.join(save_dir, 'discriminator_{}_{}_{:.4e}.h5'.format(name,epoch_number, d_current_loss)), True)

IMAGE_PATH_TRAIN = 'E:/deep learning files/SEM enhancement/cropped input/train/'#'E:/deep learning files/GAN example/FITC/SMC/input/'
LABLE_PATH_TRAIN = 'E:/deep learning files/SEM enhancement/cropped label/train/'#'E:/deep learning files/GAN example/FITC/SMC/label/'

def train_multiple_outputs(n_images, batch_size, log_dir, epoch_num, critic_updates):
	
	g1 = GAN_models.generator()
	g2 = GAN_models.generator()
	#g1.load_weights('E:/Python files/cycleGAN/codes/weights/SEM_7_6_res_SSIM/generator_1_4_1.2590e-03.h5')
	#g2.load_weights('E:/Python files/cycleGAN/codes/weights/SEM_7_6_res_SSIM/generator_2_4_2.0025e-03.h5')
	d1 = GAN_models.discriminator()
	d2 = GAN_models.discriminator()
	#d1.load_weights('E:/Python files/cycleGAN/codes/weights/SEM_7_6_res_SSIM/discriminator_1_4_5.2786e-02.h5')
	#d2.load_weights('E:/Python files/cycleGAN/codes/weights/SEM_7_6_res_SSIM/discriminator_2_4_6.4358e-02.h5')
	
	def model_GAN(g, d):
		inputs = Input(shape=(256,256,1))
		generated_imgae = g(inputs)
		output = d(generated_imgae)
		model = tf.keras.Model(inputs = inputs, outputs = output)
		return model
	
	def model_A_to_B(g1, g2):
		inputs = Input(shape=(256,256,1))
		generated_imgae = g1(inputs)
		output = g2(generated_imgae)
		model = tf.keras.Model(inputs = inputs, outputs = output)
		return model
	
	model_GAN1 = model_GAN(g1, d1)
	model_D_F = model_A_to_B(g1,g2)
	model_GAN2 = model_GAN(g2, d2)
	model_F_D = model_A_to_B(g2,g1)
	
	d1.trainable = True
	d1.compile(optimizer=RMSprop(learning_rate=0.0005), loss=losses.wasserstein_loss)
	d1.trainable = False
	model_GAN1.compile(optimizer=RMSprop(learning_rate=0.0001), loss=losses.wasserstein_loss)
	#g1.compile(optimizer='Nadam', loss=losses.SSIM_loss)
	model_D_F.compile(optimizer='Nadam', loss=losses.SSIM_loss)#losses.SSIM_loss
	d1.trainable = True
	
	d2.trainable = True
	d2.compile(optimizer=RMSprop(learning_rate=0.0005), loss=losses.wasserstein_loss)
	d2.trainable = False
	model_GAN2.compile(optimizer=RMSprop(learning_rate=0.0001), loss=losses.wasserstein_loss)#losses.wasserstein_loss
	#g2.compile(optimizer=RMSprop(learning_rate=0.0002), loss=losses.SSIM_loss)
	model_F_D.compile(optimizer='Nadam', loss=losses.SSIM_loss)
	d2.trainable = True
	
	#output_true_batch, output_false_batch = np.ones((batch_size,1)), np.zeros((batch_size,1))
	#output_true_batch = utils.to_categorical(output_true_batch, 2, dtype='float32')
	#output_false_batch = utils.to_categorical(output_false_batch, 2, dtype='float32')
	
	log_path = 'E:/Python files/cycleGAN/codes/logs/'
	tensorboard_callback = TensorBoard(log_path)
	
	
	loss_of_d1=[]
	loss_of_d2=[]
	loss_of_g1=[]
	loss_of_g2=[]
	loss_of_GAN1=[]
	loss_of_GAN2=[]
	ep = 1
	for epoch in tqdm.tqdm(range(epoch_num)):
		
		data = load_images.Load_images(IMAGE_PATH_TRAIN, LABLE_PATH_TRAIN, n_images)
		x_train = data['Input']
		y_train = data['Label']
		
		permutated_indexes = np.random.permutation(y_train.shape[0])

		d1_losses = []
		g1_losses = []
		A_to_B_losses = []
		d2_losses = []
		g2_losses = []
		B_to_A_losses = []
		iters = 1
		for index in range(int(y_train.shape[0] / batch_size)):
			
			output_true_batch, output_false_batch = -np.ones((batch_size,1)), np.ones((batch_size,1))
			#label_noises1 = 0.1*np.random.rand(batch_size,1)
			#label_noises2 = 0.1*np.random.rand(batch_size,1)
			#output_true_batch = output_true_batch - label_noises1
			#output_false_batch = output_false_batch + label_noises2
			
			batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
			
			inputs = x_train[batch_indexes]
			labels = y_train[batch_indexes]

			generated_labels = g1.predict(x=inputs, batch_size=batch_size)
			generated_inputs = g2.predict(x=labels, batch_size=batch_size)
			#restored_labels = model_F_D.predict(x=labels, batch_size=batch_size)
			#restored_inputs = model_D_F.predict(x=inputs, batch_size=batch_size)
			
			if ep==1 and iters==1:
				critic = 20
			else:
				critic = critic_updates
#=======================================================================	
			#train discriminator
			for _ in range(critic):
				d1_loss_real = d1.train_on_batch(labels, output_true_batch)
				d1_loss_fake = d1.train_on_batch(generated_labels, output_false_batch)
				d1_loss = 0.5 * np.add(d1_loss_fake, d1_loss_real)
				d1_losses.append(d1_loss)
				d2_loss_real = d2.train_on_batch(inputs, output_true_batch)
				d2_loss_fake = d2.train_on_batch(generated_inputs, output_false_batch)
				d2_loss = 0.5 * np.add(d2_loss_fake, d2_loss_real)
				d2_losses.append(d2_loss)
			d1.trainable = False
			d2.trainable = False
#=======================================================================
			#train GAN
			A_to_B_loss = model_GAN1.train_on_batch(inputs, output_true_batch)
			A_to_B_losses.append(A_to_B_loss)		
			B_to_A_loss = model_GAN2.train_on_batch(labels, output_true_batch)
			B_to_A_losses.append(B_to_A_loss)
#=======================================================================	
			#train cycleloss
			#g1.trainable = False
			
			#gth1_loss = g1.train_on_batch(inputs,labels)
			#gth2_loss = g2.train_on_batch(labels,inputs)
			
			#for _ in range(2):
			g1_loss = model_D_F.train_on_batch(inputs,inputs)
			#g1_loss_2 = model_D_F.train_on_batch(restored_inputs,inputs)
			#g1_loss = 0.5 * np.add(g1_loss_1, g1_loss_2)
			g1_losses.append(g1_loss)	
			#g1.trainable = True
			
			#g2.trainable = False
			#for _ in range(2):
			g2_loss = model_F_D.train_on_batch(labels,labels)
			#g2_loss_2 = model_F_D.train_on_batch(restored_labels,labels)
			#g2_loss = 0.5 * np.add(g2_loss_1, g2_loss_2)
			g2_losses.append(g2_loss)	
			#g2.trainable = True
			
			d1.trainable = True
			d2.trainable = True
			
			print('epoch={}, iters={}, \nloss_A_to_B={:.5f}, d1_loss={:.5f}, g1_loss={:.5f}\nloss_B_to_A={:.5f}, d2_loss={:.5f}, g2_loss={:.5f}'.format(ep, iters, A_to_B_loss, d1_loss, g1_loss, B_to_A_loss, d2_loss, g2_loss))
			iters += 1
			
			loss_of_d1.append(d1_loss)
			loss_of_d2.append(d2_loss)
			loss_of_g1.append(g1_loss)
			loss_of_g2.append(g2_loss)
			loss_of_GAN1.append(A_to_B_loss)
			loss_of_GAN2.append(B_to_A_loss)
			
#=======================================================================
		# write_log(tensorboard_callback, ['g_loss', 'd_on_g_loss'], [np.mean(d_losses), np.mean(d_on_g_losses)], epoch_num)
		#print(np.mean(d1_losses),np.mean(d2_losses),np.mean(A_to_B_losses),np.mean(B_to_A_losses))
		#with open('log.txt', 'a+') as f:
		#	f.write('{} - {} - {}\n'.format(epoch, np.mean(d1_losses), np.mean(g1_losses)))
		
		save_all_weights(d1, g1, epoch, float(np.mean(g1_losses)), float(np.mean(d1_losses)),'1')
		save_all_weights(d2, g2, epoch, float(np.mean(g2_losses)), float(np.mean(d2_losses)),'2')
		ep += 1
		
	training_his = np.array([loss_of_d1,loss_of_d2,loss_of_g1,loss_of_g2,loss_of_GAN1,loss_of_GAN2])
	training_his = pd.DataFrame(data=training_his)
	training_his.to_csv('E:/Python files/cycleGAN/codes/training_his/training_his_SEM_1.csv',encoding='gbk')
#@click.command()
#@click.option('--n_images', default=-1, help='Number of images to load for training')
#@click.option('--batch_size', default=16, help='Size of batch')
#@click.option('--log_dir', required=True, help='Path to the log_dir for Tensorboard')
#@click.option('--epoch_num', default=4, help='Number of epochs for training')
#@click.option('--critic_updates', default=5, help='Number of discriminator training')
def train_command(n_images, batch_size, log_dir, epoch_num, critic_updates):
	return train_multiple_outputs(n_images, batch_size, log_dir, epoch_num, critic_updates)


if __name__ == '__main__':
	train_command(743, 8, True, 50, 1)
