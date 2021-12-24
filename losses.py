import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import *
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import utils
from tensorflow.keras.constraints import min_max_norm
import tensorflow as tf

pre_model = load_model('E:/Python files/focus/Three Cells/no validation/best model fold-1 134 no-validation.hdf5')
inputs = Input((388,388,3))
x = inputs
for layer in pre_model.layers[1:5]:
	x = layer(x)
loss_model = Model(inputs=inputs, outputs=x)
loss_model.trainable = False

def l1_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

def l2_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def perceptual_loss_100(y_true, y_pred):
    return 100 * perceptual_loss(y_true, y_pred)

def perceptual_loss(y_true, y_pred):
	return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

def SSIM_loss(y_true, y_pred):
	#loss = tf.nn.relu(0.95-tf.image.ssim(y_true,y_pred,max_val=1))
	loss = 1-tf.image.ssim(y_true,y_pred,max_val=255.0)
	return loss

def threshold_loss1(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	y_true_f = K.cast(K.greater_equal(y_true_f,100), dtype=K.floatx())
	y_pred_f = K.sigmoid(y_pred_f-100)
	intersection = K.sum(y_true_f * y_pred_f)
	smooth = 1
	return 1.-(2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

def threshold_loss2(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	y_true_f = K.cast(K.greater_equal(y_true_f,80), dtype=K.floatx())
	y_pred_f = K.sigmoid(y_pred_f-80)
	intersection = K.sum(y_true_f * y_pred_f)
	smooth = 1
	return 1.-(2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

def combo_loss(y_true, y_pred):
	loss1 = l1_loss(y_true, y_pred)
	loss2 = SSIM_loss(y_true, y_pred)
	loss = loss1+loss2*100
	return loss
	
def GAN_output_loss(y_true, y_pred):
	PL = perceptual_loss(y_true, y_pred)
	L2 = l2_loss(y_true, y_pred)
	loss = PL+L1*0.1
	return loss

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def gradient_penalty_loss(y_true, y_pred):
    averaged_samples = merge_function(y_true, y_pred)
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))

    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = K.square(1 - gradient_l2_norm)

    return K.mean(gradient_penalty)

def merge_function(y_true, y_pred):
	#inputs = np.array([y_true, y_pred])
	#shape = inputs.shape
	#print(shape)
	alpha = K.random_uniform(shape=(2,356,426,1))
	return (alpha * y_true) + ((1 - alpha) * y_pred)
