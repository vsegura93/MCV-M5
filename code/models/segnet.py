##########Segnet -> Encoding block // decoding block


# Keras imports
from keras.models import Model
from keras.layers import Input, merge, BatchNormalization
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D,UpSampling2D)
from keras.layers.core import Dropout, Activation, Reshape, Permute
from keras.regularizers import l2

# Custom layers import
from layers.ourlayers import (CropLayer2D, NdSoftmax)
from layers.deconv import Deconvolution2D

from keras import backend as K
dim_ordering = K.image_dim_ordering()




def build_segnet(img_shape=(3, None, None), nclasses=8, l2_reg=0., init='glorot_uniform', path_weights=None, freeze_layers_from=None):

  # Regularization warning
  if l2_reg > 0.:
    print ("Regularizing the weights: " + str(l2_reg))

  # Build network:
  
  #Encoding Block
  
  # Input layer
  inputs = Input(shape=img_shape)
  padded = ZeroPadding2D(padding=(10, 10), name='pad100')(inputs)
  
  #Enc Block 1
  X = batched_conv(padded, 64, 3, 3, l2_reg, 1, 'Enc')
  X = batched_conv(X, 64, 3, 3, l2_reg, 2, 'Enc')
  X = MaxPooling2D(name = 'Enc_MaxP_1')(X)
  
  #Enc Block 2
  X = batched_conv(X, 128, 3, 3, l2_reg, 3, 'Enc')
  X = batched_conv(X, 128, 3, 3, l2_reg, 4, 'Enc')
  X = MaxPooling2D(name = 'Enc_MaxP_2')(X)
  
  #Enc Block 3
  X = batched_conv(X, 256, 3, 3, l2_reg, 5, 'Enc')
  X = batched_conv(X, 256, 3, 3, l2_reg, 6, 'Enc')
  X = batched_conv(X, 256, 3, 3, l2_reg, 7, 'Enc')
  X = MaxPooling2D(name = 'Enc_MaxP_3')(X)
  
  #Enc Block 4
  X = batched_conv(X, 512, 3, 3, l2_reg, 8, 'Enc')
  X = batched_conv(X, 515, 3, 3, l2_reg, 9, 'Enc')
  X = batched_conv(X, 512, 3, 3, l2_reg, 10, 'Enc')
  X = MaxPooling2D(name = 'Enc_MaxP_4')(X)
  
  #Enc Block 5
  X = batched_conv(X, 512, 3, 3, l2_reg, 11, 'Enc')
  X = batched_conv(X, 512, 3, 3, l2_reg, 12, 'Enc')
  X = batched_conv(X, 512, 3, 3, l2_reg, 13, 'Enc')
  X = MaxPooling2D(name = 'Enc_MaxP_5')(X)
  
  #Dec Block 1
  X = UpSampling2D(name = 'Dec_Ups_1')(X)
  X = batched_conv(X, 512, 3, 3, l2_reg, 1, 'Dec')
  X = batched_conv(X, 512, 3, 3, l2_reg, 2, 'Dec')
  X = batched_conv(X, 512, 3, 3, l2_reg, 3, 'Dec')
  
  #Dec Block 2
  X = UpSampling2D(name = 'Dec_Ups_2')(X)
  X = batched_conv(X, 512, 3, 3, l2_reg, 4, 'Dec')
  X = batched_conv(X, 512, 3, 3, l2_reg, 5, 'Dec')
  X = batched_conv(X, 512, 3, 3, l2_reg, 6, 'Dec')

  #Dec Block 3  
  X = UpSampling2D(name = 'Dec_Ups_3')(X)
  X = batched_conv(X, 256, 3, 3, l2_reg, 7, 'Dec')
  X = batched_conv(X, 256, 3, 3, l2_reg, 8, 'Dec')
  X = batched_conv(X, 256, 3, 3, l2_reg, 9, 'Dec')
  
  #Dec Block 4  
  X = UpSampling2D(name = 'Dec_Ups_4')(X)
  X = batched_conv(X, 128, 3, 3, l2_reg, 10, 'Dec')
  X = batched_conv(X, 128, 3, 3, l2_reg, 11, 'Dec')

  #Dec Block 5  
  X = UpSampling2D(name = 'Dec_Ups_5')(X)
  X = batched_conv(X, 128, 3, 3, l2_reg, 12, 'Dec')
  X = batched_conv(X, 128, 3, 3, l2_reg, 13, 'Dec')
  
  # Softmax
  X = CropLayer2D(inputs, name='Crop')(X)
  #X = Reshape(img_shape)(X)
  
  softmax_segnet = NdSoftmax()(X)
  
  #print img_shape
    
  #X = Reshape((5, 288 * 384))(X)
  #X = Permute((2, 1))(X)
  #softmax_segnet = Activation('softmax')(X)

  # Complete model
  model = Model(input=inputs, output=softmax_segnet)   
  
  # Load pretrained Model
  if path_weights:
    load_matcovnet(model, path_weights, n_classes=nclasses)

  # Freeze some layers
  if freeze_layers_from is not None:
    freeze_layers(model, freeze_layers_from)

  return model
  
  
def batched_conv(X, channels, kernel_w, kernel_h, l2_reg, index, mode):
  
  X = Convolution2D(channels, kernel_w,kernel_h, border_mode= 'same', W_regularizer=l2(l2_reg), name=mode +'_Conv_'+str(index))(X)
  
  X = BatchNormalization(gamma_regularizer=l2(l2_reg), beta_regularizer=l2(l2_reg), name=mode + '_BN_'+str(index))(X)
  
  X = Activation('relu',name=mode + '_Relu_' + str(index))(X)
  
  return X
  
  
  
               
               