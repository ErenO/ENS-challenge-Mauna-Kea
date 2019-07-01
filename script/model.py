import tensorflow as tf
import keras
import keras.backend as K

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

from keras.applications.nasnet import NASNetMobile, NASNetLarge, preprocess_input
from keras.optimizers import Adam, RMSprop
from keras.losses import binary_crossentropy, categorical_crossentropy

import matplotlib.image as mpimg

from keras.models import Sequential, Model
from keras.layers import *

from keras.optimizers import *

# def get_model_res(pretrained_model, nbClasses):
#     base_model = pretrained_model

#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(1024, activation='relu')(x)
#     x = Dropout(0.3)(x)
#     x = Dense(512, activation='relu')(x)
#     x = Dropout(0.3)(x)

#     predictions = Dense(nbClasses, activation='softmax')(x)
#     model = Model(inputs=base_model.input, outputs=predictions)
#     model.compile(optimizer=Adam(0.0001), loss=keras.losses.categorical_crossentropy, metrics=['acc'])
#     model.summary()
#     return (model)

# def get_model_classif_nasnet(num_classes=2):
#     inputs = Input((224, 224, 3))
#     base_model = NASNetMobile(include_top=False, input_shape=(224, 224, 3))#, weights=None
#     x = base_model(inputs)
#     out1 = GlobalMaxPooling2D()(x)
#     out2 = GlobalAveragePooling2D()(x)
#     out3 = Flatten()(x)
#     out = Concatenate(axis=-1)([out1, out2, out3])
#     out = Dropout(0.5)(out)
#     out = Dense(num_classes, activation="softmax", name="3_")(out)
#     model = Model(inputs, out)
#     for layer in base_model.layers[-4]:
#         layer.trainable = False
#     model.compile(optimizer=Adam(0.0001), loss=categorical_crossentropy, metrics=['acc'])
#     model.summary()

#     return (model)

def get_model_classif_nasnet(pretrain_model, num_classes=2, trainable=5, lossFunc=categorical_crossentropy):
    '''
        trainable : number of layers trainable
        num_classes : number of classes to classify
        pretrain_model : model with pretrain weight loaded
    '''
#     base_model = NASNetMobile(include_top=False, input_shape=(224, 224, 3))#, weights=None
    inputs = Input((224, 224, 3))
    base_model = pretrain_model
    x = base_model(inputs)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(num_classes, activation="softmax", name="3_")(out)
    model = Model(inputs, out)
#     for layer in base_model.layers[:-trainable]:
#         layer.trainable = False
    model.compile(optimizer=Adam(0.0001), loss=lossFunc, metrics=['acc'])
#     model.summary()

    return (model)

def get_model_1(pretrained_model, nbClasses, trainable=10, lossFunc=categorical_crossentropy):
    '''
        trainable : number of layers trainable
        num_classes : number of classes to classify
        pretrain_model : model with pretrain weight loaded
    '''
    base_model = pretrained_model

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)

    predictions = Dense(nbClasses, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(0.0001), loss=lossFunc, metrics=['acc'])
    
    for layer in base_model.layers[:-trainable]:
        layer.trainable = False
#     model.summary()
    return (model)

def get_model_2(pretrained_model,numclasses, trainable=10, lossFunc=categorical_crossentropy):
    '''
        trainable : number of layers trainable
        num_classes : number of classes to classify
        pretrain_model : model with pretrain weight loaded
    '''
    base_model = pretrained_model # Topless
    # Add top layer
    x = base_model.output
    x = Conv2D(512, kernel_size = (3,3), padding = 'valid')(x)
    x = GlobalAveragePooling2D()(x)

    predictions = Dense(numclasses, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers[:-trainable]:
        layer.trainable = False
    
    model.compile(optimizer=Adam(0.0001), loss=lossFunc, metrics=['acc'])
    model.summary()
    return (model)