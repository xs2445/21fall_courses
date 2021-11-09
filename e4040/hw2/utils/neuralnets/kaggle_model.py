import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, BatchNormalization, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input



def create_model(lr = 1e-3):
    #*****************************
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='valid', input_shape=(128,128,3)))
    model.add(BatchNormalization())
    # model.add(Conv2D(32, kernel_size=3, activation='relu', padding='valid'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.5))

    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='valid'))
    model.add(BatchNormalization())
    # model.add(Conv2D(256, kernel_size=3, activation='relu', padding='valid'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='valid'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='valid'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    # model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    # model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='valid'))
    model.add(BatchNormalization())
    # model.add(MaxPooling2D())
    # model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.4))
    model.add(Dense(5, activation='softmax'))
    #*****************************

    
    # for layers in model_efn.layers:
    # layers.trainable = True


    # optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True) # Adam(lr=lr, decay=0.01) 
    optimizer = Adam(learning_rate=lr, decay=0.01)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


def create_model2(lr=1e-3):
    
    #********************
    # inputs = tf.keras.Input(shape=(128,128,3))
    base_model=EfficientNetB0(weights='imagenet',include_top=False, input_shape=(128,128,3)) 
    x=base_model.output
    
    # x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)

    x = Dropout(0.5)(x)

    x = BatchNormalization()(x)

    x=Dense(512,activation='relu')(x) #dense layer 1
    x = BatchNormalization()(x)
    
    x=Dense(512,activation='relu')(x) #dense layer 2
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # x=Dense(64,activation='relu')(x) #dense layer 3
    preds=Dense(5,activation='softmax')(x) #final layer with softmax activation
    model=Model(inputs=base_model.input,outputs=preds)
    #********************

    
    # for layer in model.layers[:20]:
    #     layer.trainable=False
    # for layer in model.layers[20:]:
    #     layer.trainable=True
    
    for layers in model.layers:
        layers.trainable = True   
        
    optimizer = Adam(learning_rate=lr, decay=0.01)
        
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
    return model


def create_model3(lr=1e-3):
    
    #********************
    base_model=ResNet50(weights='imagenet',include_top=False, input_shape=(128,128,3)) 
    x=base_model.output
    
    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)

    x = BatchNormalization()(x)
    
    x=Dense(1024,activation='relu')(x) #dense layer 1
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)


    x=Dense(512,activation='relu')(x) #dense layer 3
    preds=Dense(5,activation='softmax')(x) #final layer with softmax activation
    model=Model(inputs=base_model.input,outputs=preds)
    #********************

    trainable_layers = 18
    
    for layer in model.layers[:trainable_layers]:
        layer.trainable=False
    for layer in model.layers[trainable_layers:]:
        layer.trainable=True
    
    # for layers in model.layers:
    #     layers.trainable = True   
        
    optimizer = Adam(learning_rate=lr, decay=0.01)
        
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
    return model


def create_model4(lr=1e-3):
    
    #********************
    base_model=EfficientNetB7(weights='imagenet',include_top=False, input_shape=(128,128,3)) 
    x=base_model.output
    
    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)

    x = BatchNormalization()(x)

    x=Dense(512,activation='relu')(x) #dense layer 1
    x = BatchNormalization()(x)
    
    x=Dense(512,activation='relu')(x) #dense layer 2
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x=Dense(64,activation='relu')(x) #dense layer 3
    preds=Dense(5,activation='softmax')(x) #final layer with softmax activation
    
    model=Model(inputs=base_model.input,outputs=preds)
    #********************

    trainable_layers = 18
    
    for layer in model.layers[:trainable_layers]:
        layer.trainable=False
    for layer in model.layers[trainable_layers:]:
        layer.trainable=True
    
    # for layers in model.layers:
    #     layers.trainable = True   
        
    optimizer = Adam(learning_rate=lr, decay=0.01)
        
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
    return model



def create_model5(lr=1e-3):
    
    #********************
    base_model=VGG16(weights='imagenet',include_top=False, input_shape=(128,128,3)) 
    x=base_model.output
    
    x = Flatten()(x)
    
    # x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)

    x = BatchNormalization()(x)

    x=Dense(512,activation='relu')(x) #dense layer 1
    x = BatchNormalization()(x)
    
    x=Dense(512,activation='relu')(x) #dense layer 2
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x=Dense(64,activation='relu')(x) #dense layer 3
    preds=Dense(5,activation='softmax')(x) #final layer with softmax activation
    
    model=Model(inputs=base_model.input,outputs=preds)
    #********************

    untrainable_layers = 5
    
    for layer in model.layers[:untrainable_layers]:
        layer.trainable=False
    for layer in model.layers[untrainable_layers:]:
        layer.trainable=True
    
    # for layers in model.layers:
    #     layers.trainable = True   
        
    optimizer = Adam(learning_rate=lr, decay=0.01)
        
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
    return model


def create_model6(lr=1e-3):
    
    #********************
    base_model=EfficientNetB0(weights='imagenet',include_top=False, input_shape=(128,128,3)) 
    x=base_model.output
    
    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)

    preds=Dense(5,activation='softmax')(x) #final layer with softmax activation
    
    model=Model(inputs=base_model.input,outputs=preds)
    #********************

    untrainable_layers = 5
    
    for layer in model.layers[:untrainable_layers]:
        layer.trainable=False
    for layer in model.layers[untrainable_layers:]:
        layer.trainable=True
    
    # for layers in model.layers:
    #     layers.trainable = True   
        
    optimizer = Adam(learning_rate=lr, decay=0.01)
        
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
    return model


def create_model7(lr=1e-3):
    
    #********************
    base_model=ResNet50(weights='imagenet',include_top=False, input_shape=(128,128,3)) 
    x=base_model.output
    
    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)

    x = BatchNormalization()(x)

    preds=Dense(5,activation='softmax')(x) #final layer with softmax activation
    
    model=Model(inputs=base_model.input,outputs=preds)
    #********************

    trainable_layers = 0
    
    for layer in model.layers[:trainable_layers]:
        layer.trainable=False
    for layer in model.layers[trainable_layers:]:
        layer.trainable=True
    
    # for layers in model.layers:
    #     layers.trainable = True   
        
    optimizer = Adam(learning_rate=lr, decay=0.01)
        
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
    return model