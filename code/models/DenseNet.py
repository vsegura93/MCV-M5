#-Parameters:
#	-growth rate -> number of layers for each convolutional layer inside the dense block
#	-l -> number of Composite functions in a dense block
#	-n_d -> number of dense blocks
#	-NC number of classes
#	-theta -> Compression Factor (n layers/filters on the transition layers)
#	-dpt -> Dropout rate


#Dense Block:
#	-opt1:  BN + Act(ReLu) + 3x3 Conv + Drpt
#	-opt2:  BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3) + Drpt
#
#Transition:
#	1x1 conv + 2x2 avg pooling

#- Architecture:

#Input-> Conv (7x7, stride 2) -> MaxPooling (3x3, stride 2) -> 
#	->(DenseBlock -> Transition ) * n_d ->
#	-> Classification (7x7 average pooling -> K Dim fully connected -> softmax)

#/********************************************************************************/
#/*****************************Densenet********************************************/
#/********************************************************************************/
import keras.backend as K
from keras.models import Model
from keras.layers import Flatten, Convolution2D, Input, Activation, Flatten, Dropout
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, merge, AveragePooling2D
from keras.regularizers import l2

def build_DenseNet(input_shape=(3, 224, 224), NC=1000, l2_reg=0., load_pretrained=None, n_filters=14, growth_rate=4, l=4, n_d=3,  theta=1, dpt=0.5, freeze_layers_from='base_model'):


	input_data = Input(shape=input_shape)
  
	X = Convolution2D(n_filters, 7, 7, subsample=(2,2), border_mode='same', name='First_convolution',W_regularizer=l2(l2_reg))(inputs)
	
	for index in range(n_d):
		X, n_filters = DenseBlock(X, growth_rate, dpt, l, n_filters, index, l2_reg)
		X, n_filters = Transition(X, theta, n_filters, index, l2_reg)
	X = GlobalAveragePooling2D(name='GlobalAveragePooling')(X)
	Predictions = Dense(NC, activation='softmax', W_regularizer=l2(l2_reg), name='softmax')(X)

	model = Model(input=inputs, output=Predictions)

  # Freeze some layers
  if freeze_layers_from is not None:
    if freeze_layers_from == 'base_model':
      print ('   Freezing base model layers')
        for layer in base_model.layers:
          layer.trainable = False
    else:
      for i, layer in enumerate(model.layers):
        print(i, layer.name)
        print ('   Freezing from layer 0 to ' + str(freeze_layers_from))
        for layer in model.layers[:freeze_layers_from]:
          layer.trainable = False
          for layer in model.layers[freeze_layers_from:]:
            layer.trainable = True

	#if weights_path:
        	#model.load_weights(weights_path)

	return model



#versio 1
def DenseConv(X, growth_rate, dpt, index, i, l2_reg):

	X = BatchNormalization(gamma_regularizer=l2(l2_reg),                         beta_regularizer=l2(l2_reg), name='BatchNorm_'+str(index)+ str(i))(X)
 
	X = Activation('relu', name='Relu_' + str(index)+str(i))(X)
 
	X = Convolution2D(growth_rate, 3, 3, border_mode='same', name='Convolution_'+str(index)+str(i))(X)
	X = Dropout(dpt, name='dropout_'+str(index)+str(i))(X)

	return X

#versio 2
def BottleConv(X, growth_rate, dpt, index, i, l2_reg):

	X = BatchNormalization(gamma_regularizer=l2(l2_reg),                          beta_regularizer=l2(l2_reg), name='First_BatchNorm_'+str(index)+str(i))(X)
 
	X = Activation('relu',name='Relu_' + str(index)+ str(i))(X)
 
	X = Convolution2D(growth_rate, 1,1, border_mode='same', W_regularizer=l2(l2_reg), name='1x1_Convolution_'+str(index)+ str(i))(X)
 
	X = BatchNormalization(gamma_regularizer=l2(l2_reg),                          beta_regularizer=l2(l2_reg), name='BatchNormalization_'+str(index)+ str(i))(X)
 
	X = Convolution2D(growth_rate, 3, 3, border_mode='same', W_regularizer=l2(l2_reg),name='Convolution_'+str(index)+ str(i))(X)
 
	X = Dropout(dpt, name='dropout_'+str(index)+ str(i))(X)

	return X	

def Transition(X, theta, n_filters, index, l2_reg):
	
	X = BatchNormalization(gamma_regularizer=l2(l2_reg), beta_regularizer=l2(l2_reg), name='Transition_BatchNorm_'+str(index))(X)
 
	n_filters  = int(theta * n_filters)
 
	X = Convolution2D(n_filters, 1,1, border_mode='same', W_regularizer=l2(l2_reg), name='Transition_Conv'+str(index))(X)
 
	X = AveragePooling2D(pool_size=(2, 2), name='Transition_Pooling'+str(index))(X)

	return X, n_filters 

def DenseBlock(X, growth_rate, dpt, l, n_filters, index, l2_reg, Conv_Mode=DenseConv):
	
	Features = [X]
	concat_axis = 1 if K.image_dim_ordering() == "th" else -1	

	for i in range(l):
		
		X = Conv_Mode(X, growth_rate, dpt, index, i,l2_reg)
		Features.append(X) 
		X = merge(Features, mode='concat', concat_axis=concat_axis)
		n_filters += growth_rate
	
	return X, n_filters 





