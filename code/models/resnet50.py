# Keras imports
from __future__ import print_function
from keras.layers import Dense
from keras.layers import Activation
from keras.models import Model
from keras.applications.resnet50 import ResNet50

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

def build_resnet50(in_shape=(3, 224, 224), n_classes=1000, weight_decay=0.0001, 
                   load_pretrained=False, freeze_layers_from='base_model'):
    # Decide if load pretrained weights from imagenet
    if load_pretrained:
        weights = 'imagenet'
    else:
        weights = None
    
    #base_model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
    
    # Add final layers
    #x = base_model.output
    #x = Dense(n_classes, activation='softmax', name='fc1000b')(x)
    #predictions = Activation("softmax", name="softmax")(x)
    
    # This is the model we will train
    model = ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=None, classes=n_classes)#Model(input=base_model.input, output=predictions) #

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

    return model