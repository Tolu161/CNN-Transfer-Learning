#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 11:03:40 2023

@author: toluojo
"""
# Issues with this sample code need to find another one 

# Importing the needed packages 
import numpy as np
import keras 
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from tensorflow.keras.applications.vgg16 import preprocess_input 
from PIL import Image

from keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from keras.models import Sequential 

from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

#loading the MTF image, checking the dimensions and normalise pixel values 


from Case1DataMTFv1 import MTF_images_paths 
from Case1DataMTFv1 import MTF_images_paths_AE_T
from Case1DataMTFv1 import MTF_images_paths_SMCAC
from Case1DataMTFv1 import MTF_images_paths_SMDC
from Case1DataMTFv1 import MTF_images_paths_VIB_S
from Case1DataMTFv1 import MTF_images_paths_VIB_T

# Now to preprocess the MTF Image for VGG-16

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



'''
# creating and object image data generator to pass training and test data 
trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="Users/toluojo/Documents/University of Nottingham /YEAR 5 /MMME 4086 - Indv Project /mill/Case1DataMTFTv1.py",target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="Users/toluojo/Documents/University of Nottingham /YEAR 5 /MMME 4086 - Indv Project /mill/Case1DataMTFv1.py", target_size=(224,224))
'''
traindata = MTF_images
testdata = MTF_images_test

# Model Structure 

# Generate the model
model = Sequential()
# Layer 1: Convolutional
model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3),
                 padding='same', activation='relu'))
# Layer 2: Convolutional
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 3: MaxPooling
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 4: Convolutional
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 5: Convolutional
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 6: MaxPooling
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 7: Convolutional
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 8: Convolutional
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 9: Convolutional
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 10: MaxPooling
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 11: Convolutional
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 12: Convolutional
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 13: Convolutional
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 14: MaxPooling
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 15: Convolutional
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 16: Convolutional
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 17: Convolutional
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 18: MaxPooling
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

#Defines classification layers 

# Layer 19: Flatten
model.add(Flatten())
# Layer 20: Fully Connected Layer
model.add(Dense(units=4096, activation='relu'))
# Layer 21: Fully Connected Layer
model.add(Dense(units=4096, activation='relu'))
# Layer 22: Softmax Layer
model.add(Dense(units=2, activation='softmax'))



#Adam Optimiser 
# Add Optimizer
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

# Check model summary
print(model.summary())



#Model Implementation 

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', 
                             verbose=1, save_best_only=True, 
                             save_weights_only=False, mode='auto', period=1)



# Early Stopping 

earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')


#Fit Generator

hist = model.fit_generator(steps_per_epoch=100, generator=traindata, validation_data=testdata,
                           validation_steps=10, epochs=100,
                           callbacks=[checkpoint, earlystop])



# Plot Visualisation
plt.plot(hist.history["acc"])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()


fig = plt.figure(figsize=(15, 6))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(1, 1),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.3,
                 )

#plt.imshow(GAF_sample[0], cmap='rainbow')
grid = grid[0]

im = grid.imshow(hist.history["acc"], cmap='rainbow', origin='lower')
grid.set_title('acc ', fontdict={'fontsize': 12})
grid.cax.colorbar(im)

g1 = grid.imshow(hist.history["val_acc"], cmap='rainbow', origin='lower')
grid.set_title('val_acc ', fontdict={'fontsize': 12})
grid.cax.colorbar(g1)

g2 = grid.imshow(hist.history["loss"], cmap='rainbow', origin='lower')
grid.set_title('loss ', fontdict={'fontsize': 12})
grid.cax.colorbar(g2)

g3 = grid.imshow(hist.history["val_loss"], cmap='rainbow', origin='lower')
grid.set_title('val_loss ', fontdict={'fontsize': 12})
grid.cax.colorbar(g3)

grid.cax.toggle_label(True)


plt.show()









'''
# Testing the model 


img = image.load_img("../mill/MTF_images_C9.jpg",target_size=(224,224))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)
saved_model = load_model("vgg16_1.h5")
output = saved_model.predict(img)
if output[0][0] > output[0][1]:
    print("cat")
else:
    print('dog')
'''




