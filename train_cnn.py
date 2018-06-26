from PIL import Image
import numpy as np
import glob, os.path
import matplotlib.pyplot as plt
import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, Activation, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

def createModel(n_classes, input_shape):
    '''
    #--------------------------------------------------------------------------------------------#
    |  Based on: Leaf Identification Using a Deep Convolutional Neural Network                   |
    |                   Christoph Wick and Frank Puppe, Germany, 2017  			         |
    #--------------------------------------------------------------------------------------------#
    '''
    model = Sequential()
    model.add(Conv2D(40, (3, 3), padding='same', activation='relu', strides = 1, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(4,4), strides = 2))

    model.add(Conv2D(40, (2, 2), padding='same', activation='relu', strides = 1)) 
    model.add(MaxPooling2D(pool_size=(4,4), strides = 2))
    
    model.add(Conv2D(80, (2, 2), padding='same', activation='relu', strides = 1))
    model.add(MaxPooling2D(pool_size=(4,4), strides = 2))
    
    model.add(Conv2D(160, (2, 2), padding='same', activation='relu', strides = 1))
    model.add(MaxPooling2D(pool_size=(4,4), strides = 2))
    
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(n_classes, activation='softmax'))
     
    return model
 

#Some important parameters
train_data_dir = 'dataset/images/lab/train'
validation_data_dir = 'dataset/images/lab/val'
input_shape=(300, 300, 3) #Size of the input images
img_height, img_width = input_shape[0], input_shape[1]
nb_train_samples = 19743
nb_validation_samples = 4940
n_classes = len(os.listdir(train_data_dir))
batch_size = 64	#Size of the batch
epochs = 50	#Number of epochs in training

# Initiate the train and test generators with data Augumentation 
#The augmentation for the train data
train_datagen = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip = True,
    zoom_range=0.2
    )
#The augmentation for the test data - Only flips
test_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip = True)

#Get the images in batches from the directories
train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size)

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width))

#Creates the model
#model = createModel(n_classes, input_shape) #Create a new model
model = load_model("model.h5") #Continue training a model

Adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False) #creating Adam Optimizer
model.compile(loss = "categorical_crossentropy", optimizer = Adam, metrics=["accuracy"]) #compile the model 

#Save the model according to the conditions 
#Checkpoint will verify if the validation accuracy has improved
checkpoint = ModelCheckpoint("model.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1) 
#Early will stop if the model do not improve after 10 epochs
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

#Training the model
model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs=epochs,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size,
    callbacks = [checkpoint, early])
