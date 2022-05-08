from keras.models import load_model
import numpy as np
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Conv2DTranspose, Conv2D, BatchNormalization, \
						Activation, Concatenate, Input, MaxPool2D,\
						UpSampling2D, ZeroPadding2D, Lambda, Add

from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
import cv2
import os
import librosa
import scipy
from keras.utils import plot_model
import tensorflow as tf
from keras.utils import multi_gpu_model
from discriminator_D import contrastive_loss


def conv_block(x, num_filters, kernel_size=3, strides=1, padding='same', act=True):
	x = Conv2D(filters=num_filters, kernel_size= kernel_size, 
					strides=strides, padding=padding)(x)
	x = BatchNormalization(momentum=.8)(x)
	if act:
		x = Activation('relu')(x)
	return x

def conv_t_block(x, num_filters, kernel_size=3, strides=2, padding='same'):
	x = Conv2DTranspose(filters=num_filters, kernel_size= kernel_size, 
					strides=strides, padding=padding)(x)
	x = BatchNormalization(momentum=.8)(x)  #transformation to keep the mean output close to 0 and the output standard deviation close to 1
	x = Activation('relu')(x)

	return x

def create_model(args, mel_step_size):
	############# encoder for face/identity
	input_face = Input(shape=(args.img_size, args.img_size, 6), name="input_face")

	identity_mapping = conv_block(input_face, 32, kernel_size=11) # 96x96
	x1_face = conv_block(identity_mapping, 64, kernel_size=7, strides=2) # 48x48
	x2_face = conv_block(x1_face, 128, 5, 2) # 24x24
	x3_face = conv_block(x2_face, 256, 3, 2) #12x12
	x4_face = conv_block(x3_face, 512, 3, 2) #6x6
	x5_face = conv_block(x4_face, 512, 3, 2) #3x3
	x6_face = conv_block(x5_face, 512, 3, 1, padding='valid')
	x7_face = conv_block(x6_face, 256, 1, 1)

	############# encoder for audio
	input_audio = Input(shape=(80, mel_step_size, 1), name="input_audio")

	x = conv_block(input_audio, 32)
	x = conv_block(x, 32)
	x = conv_block(x, 32)

	x = conv_block(x, 64, strides=3)	#27X9
	x = conv_block(x, 64)
	x = conv_block(x, 64)

	x = conv_block(x, 128, strides=(3, 1)) 		#9X9
	x = conv_block(x, 128)
	x = conv_block(x, 128)

	x = conv_block(x, 256, strides=3)	#3X3
	x = conv_block(x, 256)
	x = conv_block(x, 256)

	x = conv_block(x, 512, strides=1, padding='valid')	#1X1
	x = conv_block(x, 512, 1, 1)

	embedding = Concatenate(axis=3)([x7_face, x])

	############# decoder
	x = conv_block(embedding, 512, 1)
	x = conv_t_block(embedding, 512, 3, 3)# 3x3
	x = Concatenate(axis=3) ([x5_face, x]) 

	x = conv_t_block(x, 512) #6x6
	x = Concatenate(axis=3) ([x4_face, x])

	x = conv_t_block(x, 256) #12x12
	x = Concatenate(axis=3) ([x3_face, x])

	x = conv_t_block(x, 128) #24x24
	x = Concatenate(axis=3) ([x2_face, x])

	x = conv_t_block(x, 64) #48x48
	x = Concatenate(axis=3) ([x1_face, x])

	x = conv_t_block(x, 32) #96x96
	x = Concatenate(axis=3) ([identity_mapping, x])

	x = conv_block(x, 16) #96x96
	x = conv_block(x, 16) #96x96
	x = Conv2D(filters=3, kernel_size=1, strides=1, padding="same") (x)
	prediction = Activation("sigmoid", name="prediction")(x)
	
	model = Model(inputs=[input_face, input_audio], outputs=prediction)
	model.summary()		


def create_combined_model(generator_G, discriminator_D, args, mel_step_size):
	input_face = Input(shape=(args.img_size, args.img_size, 6), name="input_face_comb")
	input_audio = Input(shape=(80, mel_step_size, 1), name="input_audio_comb")

	fake_face = generator_G([input_face, input_audio])
	discriminator_D.trainable = False
	d = discriminator_D([fake_face, input_audio])

	model = Model([input_face, input_audio], [fake_face, d])
	
