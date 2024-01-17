#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:05:28 2024

@author: toluojo
"""

#VGG-16 Implementation  - CLASSIFICATION APPLICATION

# Importing the needed packages 
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import preprocess_input 
from tensorflow.keras.applications.vgg16 import VGG16
from PIL import Image
from tensorflow.keras import layers 

# to download the VGG16 Weights 
import keras
print(keras.utils.get_file('', ''))


# Now to preprocess the MTF Image for VGG-16
from Case1DataMTFv1 import MTF_images_paths 
from Case1DataMTFv1 import MTF_images_paths_AE_T
from Case1DataMTFv1 import MTF_images_paths_SMCAC
from Case1DataMTFv1 import MTF_images_paths_SMDC
from Case1DataMTFv1 import MTF_images_paths_VIB_S
from Case1DataMTFv1 import MTF_images_paths_VIB_T



#TRAINING DATA

MTF_images = []

for image_path in MTF_images_paths:
    
    img = Image.open(image_path)
    
    # Resize the image to VGG-16 required size (224x224)
    img = img.resize((224, 224))
    
    # Convert the image to NumPy array and normalize pixel values
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
    
    # Preprocess the image for VGG-16
    img_preprocessed = preprocess_input(img_array)
    
    MTF_images.append(img_preprocessed)

for image_path in MTF_images_paths_AE_T:
    
    img = Image.open(image_path)
    
    # Resize the image to VGG-16 required size (224x224)
    img = img.resize((224, 224))
    
    # Convert the image to NumPy array and normalize pixel values
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
    
    # Preprocess the image for VGG-16
    img_preprocessed = preprocess_input(img_array)
    
    MTF_images.append(img_preprocessed)
    
for image_path in MTF_images_paths_SMCAC:
    
    img = Image.open(image_path)
    
    # Resize the image to VGG-16 required size (224x224)
    img = img.resize((224, 224))
    
    # Convert the image to NumPy array and normalize pixel values
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
    
    # Preprocess the image for VGG-16
    img_preprocessed = preprocess_input(img_array)
    
    MTF_images.append(img_preprocessed)
  
    
for image_path in MTF_images_paths_SMDC:
    
    img = Image.open(image_path)
    
    # Resize the image to VGG-16 required size (224x224)
    img = img.resize((224, 224))
    
    # Convert the image to NumPy array and normalize pixel values
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
    
    # Preprocess the image for VGG-16
    img_preprocessed = preprocess_input(img_array)
    
    MTF_images.append(img_preprocessed)


for image_path in MTF_images_paths_VIB_S:
    
    img = Image.open(image_path)
    
    # Resize the image to VGG-16 required size (224x224)
    img = img.resize((224, 224))
    
    # Convert the image to NumPy array and normalize pixel values
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
    
    # Preprocess the image for VGG-16
    img_preprocessed = preprocess_input(img_array)
    
    MTF_images.append(img_preprocessed)


for image_path in MTF_images_paths_VIB_T:
    
    img = Image.open(image_path)
    
    # Resize the image to VGG-16 required size (224x224)
    img = img.resize((224, 224))
    
    # Convert the image to NumPy array and normalize pixel values
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
    
    # Preprocess the image for VGG-16
    img_preprocessed = preprocess_input(img_array)
    
    MTF_images.append(img_preprocessed)


# TESTING DATA 

from CASE9 import MTF_images_paths_AE_S_C9
from CASE9 import MTF_images_paths_AE_T_C9
from CASE9 import MTF_images_paths_SMCAC_C9
from CASE9 import MTF_images_paths_SMDC_C9
from CASE9 import MTF_images_paths_VIB_S_C9
from CASE9 import MTF_images_paths_VIB_T_C9

MTF_images_test = []

for image_path in MTF_images_paths_AE_S_C9:
    
    img = Image.open(image_path)
    
    # Resize the image to VGG-16 required size (224x224)
    img = img.resize((224, 224))
    
    # Convert the image to NumPy array and normalize pixel values
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
    
    # Preprocess the image for VGG-16
    img_preprocessed = preprocess_input(img_array)
    
    MTF_images_test.append(img_preprocessed)

for image_path in MTF_images_paths_AE_T_C9:
    
    img = Image.open(image_path)
    
    # Resize the image to VGG-16 required size (224x224)
    img = img.resize((224, 224))
    
    # Convert the image to NumPy array and normalize pixel values
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
    
    # Preprocess the image for VGG-16
    img_preprocessed = preprocess_input(img_array)
    
    MTF_images_test.append(img_preprocessed)
    
for image_path in MTF_images_paths_SMCAC_C9:
    
    img = Image.open(image_path)
    
    # Resize the image to VGG-16 required size (224x224)
    img = img.resize((224, 224))
    
    # Convert the image to NumPy array and normalize pixel values
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
    
    # Preprocess the image for VGG-16
    img_preprocessed = preprocess_input(img_array)
    
    MTF_images_test.append(img_preprocessed)
  
    
for image_path in MTF_images_paths_SMDC_C9:
    
    img = Image.open(image_path)
    
    # Resize the image to VGG-16 required size (224x224)
    img = img.resize((224, 224))
    
    # Convert the image to NumPy array and normalize pixel values
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
    
    # Preprocess the image for VGG-16
    img_preprocessed = preprocess_input(img_array)
    
    MTF_images_test.append(img_preprocessed)


for image_path in MTF_images_paths_VIB_S_C9:
    
    img = Image.open(image_path)
    
    # Resize the image to VGG-16 required size (224x224)
    img = img.resize((224, 224))
    
    # Convert the image to NumPy array and normalize pixel values
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
    
    # Preprocess the image for VGG-16
    img_preprocessed = preprocess_input(img_array)
    
    MTF_images_test.append(img_preprocessed)


for image_path in MTF_images_paths_VIB_T_C9:
    
    img = Image.open(image_path)
    
    # Resize the image to VGG-16 required size (224x224)
    img = img.resize((224, 224))
    
    # Convert the image to NumPy array and normalize pixel values
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
    
    # Preprocess the image for VGG-16
    img_preprocessed = preprocess_input(img_array)
    
    MTF_images_test.append(img_preprocessed)


traindata = MTF_images
testdata = MTF_images_test

# Path to the manually downloaded weights file
weights_path = '/path/to/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

#LOAD BASE MODEL 
base_model = VGG16(input_shape = (224, 224, 3), include_top = False, weights =weights_path)


for layer in base_model.layers:
    layer.trainable = False


# Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Add a final sigmoid layer with 1 node for classification output
x = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)

model.compile(optimizer= tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])



# to modify the last layer of the keras regression instead of classification  - will change the activation function 

vgghist = model.fit(traindata, testdata, steps_per_epoch = 100, epochs = 10)




#notes for next time almost have it working just need to find a way to download the weights directly from the VGG16 weights repository, save the dow 






































































