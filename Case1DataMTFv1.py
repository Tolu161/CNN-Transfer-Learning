#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:15:27 2023

@author: toluojo
"""
# CONVERTING THE CASE 1 DATA INTO MTF - FOR TRAINING 


import pandas as pd 
import numpy as np 
from pyts.image import MarkovTransitionField
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pywt
from PIL import Image 
import os 
# was trying to see if it would work with case 1 and a few issues with the bin number 
# now polan is to reshape dataframe before it is converted to an array 


#computing a dataset of Gramian Angular Field

#8/12/23
Case_1 = pd.read_csv('Mill_Case1_gma.csv')

'''

# length of case 1 - 153000 ,split 80% - train  ,20% - test data 

#calculate the index to split the data 
train_size = int(len(Case1)*0.8)

#splitting data into test and train data 
train_data = Case1.iloc[:train_size]
test_data = Case1.iloc[train_size:]

# Resetting indices if needed
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

#rename train data to Case_1

Case_1 = train_data
'''

print(len(Case_1))


C1smcAC = Case_1.smcAC

# reshape the dataframe to a 2d dataframe 
C1smcAC = C1smcAC.values.reshape(-1)
C1smcAC_2D = C1smcAC.reshape(-1,1)

# convert case 1 dataframe into a matrix/ array 
#C1smcAC_2D = C1smcAC_2D.to_numpy()

C1smcAC_2D = np.asfarray(C1smcAC_2D)
print(C1smcAC_2D.shape)


#define the step size for samples
sample_size = 250

# splitting the case 1 dataframe into the different samples 
samples = [C1smcAC_2D[i:i+sample_size] for i in range(0,len(C1smcAC_2D), sample_size)]

#Get recurence plots for all time series 
MTF = MarkovTransitionField(n_bins=30)



#storing the file paths
MTF_images_paths_SMCAC = []


# for loop to append each Markvov transition field to a list of path files and convert it to an image 
for i, sample in enumerate(samples) :
    
    sample= sample.flatten()
    sample = np.array([sample]) 
    
    MTF_sample = MTF.fit_transform(sample)
    print(MTF_sample)
   
   #convert MTF to image , mtf values scaled between 0 and 224 using max scaling  to fit, the data type is converted after scaling to ensure its compatible  with image generation  
    MTF_image = (MTF_sample - MTF_sample.min()) / (MTF_sample.max() - MTF_sample.min()) * 224.0
    MTF_image = MTF_image.astype(np.uint8)
    img = Image.fromarray(MTF_image[0])  # Assuming MTF_sample shape is (1, 1, 250)   

    # Define the output folder
    output_folder = 'MTF_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the image with a unique filename
    image_filename = f'{output_folder}/MTF_image_SMCAC{i}.png'
    img.save(image_filename, 'PNG')


    # Append the file path to the list
    MTF_images_paths_SMCAC.append(image_filename)
    
 

    
'''
    # To display the grid  
    
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

    im = grid.imshow(MTF_sample[0], cmap='rainbow', origin='lower')
    grid.set_title('MTF ', fontdict={'fontsize': 12})
    grid.cax.colorbar(im)
    grid.cax.toggle_label(True)
    
    plt.suptitle('Markov transition fields for time series smcAC data', y=0.98, fontsize=16)
    
    plt.show()
    '''
   
#SPINDLE MOTOR DIRECT CURRENT 
   
# Do the same for spindle motor direct current   - smcDC - but not displaying MTF filed 
C1smcDC = Case_1.smcDC

#reshape the dataframe to a 2d dataframe 
C1smcDC = C1smcDC.values.reshape(-1)
C1smcDC_2D = C1smcDC.reshape(-1,1)


C1smcDC_2D = np.asfarray(C1smcDC_2D)
print(C1smcDC_2D.shape)


#define the step size for samples
sample_size = 250

# splitting the case 1 dataframe into the different samples 
samples = [C1smcDC_2D[i:i+sample_size] for i in range(0,len(C1smcDC_2D), sample_size)]

#Get recurence plots for all time series 
MTF = MarkovTransitionField(n_bins=30)

#storing the file paths
MTF_images_paths_SMDC = []


# for loop to append each Markvov transition field to a list of path files and convert it to an image 
for i, sample in enumerate(samples) :
    
    sample= sample.flatten()
    sample = np.array([sample]) 
    
    MTF_sample = MTF.fit_transform(sample)
    print(MTF_sample)
   
   #convert MTF to image , mtf values scaled between 0 and 224 using max scaling  to fit, the data type is converted after scaling to ensure its compatible  with image generation  
    MTF_image = (MTF_sample - MTF_sample.min()) / (MTF_sample.max() - MTF_sample.min()) * 224.0
    MTF_image = MTF_image.astype(np.uint8)
    img = Image.fromarray(MTF_image[0])  # Assuming MTF_sample shape is (1, 1, 250)   

    # Define the output folder
    output_folder = 'MTF_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the image with a unique filename
    image_filename = f'{output_folder}/MTF_image_SMDC{i}.png'
    img.save(image_filename, 'PNG')


    # Append the file path to the list
    MTF_images_paths_SMDC.append(image_filename)
  
    
  
    
  
    
  
    
  
#TABLE VIBRATION 
    
#now adding vib_table     
C1vibt = Case_1.vib_table

#reshape the dataframe to a 2d dataframe 
C1vibt  = C1vibt.values.reshape(-1)
C1vibt_2D = C1vibt.reshape(-1,1)


C1vibt_2D = np.asfarray(C1vibt_2D)
print(C1vibt_2D.shape)


#define the step size for samples
sample_size = 250

# splitting the case 1 dataframe into the different samples 
samples = [C1vibt_2D[i:i+sample_size] for i in range(0,len(C1vibt_2D), sample_size)]

#Get recurence plots for all time series 
MTF = MarkovTransitionField(n_bins=30)

#storing the file paths
MTF_images_paths_VIB_T = []


# for loop to append each Markvov transition field to a list of path files and convert it to an image 
for i, sample in enumerate(samples) :
    
    sample= sample.flatten()
    sample = np.array([sample]) 
    
    MTF_sample = MTF.fit_transform(sample)
    print(MTF_sample)
   
   #convert MTF to image , mtf values scaled between 0 and 224 using max scaling  to fit, the data type is converted after scaling to ensure its compatible  with image generation  
    MTF_image = (MTF_sample - MTF_sample.min()) / (MTF_sample.max() - MTF_sample.min()) * 224.0
    MTF_image = MTF_image.astype(np.uint8)
    img = Image.fromarray(MTF_image[0])  # Assuming MTF_sample shape is (1, 1, 250)   

    # Define the output folder
    output_folder = 'MTF_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the image with a unique filename
    image_filename = f'{output_folder}/MTF_image_VIB_T{i}.png'
    img.save(image_filename, 'PNG')


    # Append the file path to the list
    MTF_images_paths_VIB_T.append(image_filename)
    










    
    
#SPINDLE VIBRATION 
        
C1vibsp = Case_1.vib_spindle

#reshape the dataframe to a 2d dataframe 
C1vibsp = C1vibsp.values.reshape(-1)
C1vibsp_2D = C1vibsp.reshape(-1,1)


C1vibsp_2D = np.asfarray(C1vibsp_2D)
print(C1vibsp_2D.shape)


#define the step size for samples
sample_size = 250

# splitting the case 1 dataframe into the different samples 
samples = [C1vibsp_2D[i:i+sample_size] for i in range(0,len(C1vibsp_2D), sample_size)]

#Get recurence plots for all time series 
MTF = MarkovTransitionField(n_bins=30)


    
#storing the file paths

MTF_images_paths_VIB_S = []


# for loop to append each Markvov transition field to a list of path files and convert it to an image 
for i, sample in enumerate(samples) :
    
    sample= sample.flatten()
    sample = np.array([sample]) 
    
    MTF_sample = MTF.fit_transform(sample)
    print(MTF_sample)
   
   #convert MTF to image , mtf values scaled between 0 and 224 using max scaling  to fit, the data type is converted after scaling to ensure its compatible  with image generation  
    MTF_image = (MTF_sample - MTF_sample.min()) / (MTF_sample.max() - MTF_sample.min()) * 224.0
    MTF_image = MTF_image.astype(np.uint8)
    img = Image.fromarray(MTF_image[0])  # Assuming MTF_sample shape is (1, 1, 250)   

    # Define the output folder
    output_folder = 'MTF_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the image with a unique filename
    image_filename = f'{output_folder}/MTF_image_VIB_S{i}.png'
    img.save(image_filename, 'PNG')


    # Append the file path to the list
    MTF_images_paths_VIB_S.append(image_filename)
    
    
    
    
    
    
    
  

#TABLE ACOUSTIC EMISSION 


#now adding vib_table     
C1AE_T = Case_1.AE_Table 

#reshape the dataframe to a 2d dataframe 
C1AE_T  = C1AE_T.values.reshape(-1)
C1AE_T_2D = C1AE_T.reshape(-1,1)


C1AE_T_2D = np.asfarray(C1AE_T_2D)
print(C1AE_T_2D.shape)


#define the step size for samples
sample_size = 250

# splitting the case 1 dataframe into the different samples 
samples = [C1AE_T_2D[i:i+sample_size] for i in range(0,len(C1AE_T_2D), sample_size)]

#Get recurence plots for all time series 
MTF = MarkovTransitionField(n_bins=30)


#storing the file paths

MTF_images_paths_AE_T = []


# for loop to append each gramian angular field to a list 
for i, sample in enumerate(samples) :
    
    sample= sample.flatten()
    sample = np.array([sample]) 
    
    MTF_sample = MTF.fit_transform(sample)
    print(MTF_sample)
   
   #convert MTF to image , mtf values scaled between 0 and 224 using max scaling  to fit, the data type is converted after scaling to ensure its compatible  with image generation  
    MTF_image = (MTF_sample - MTF_sample.min()) / (MTF_sample.max() - MTF_sample.min()) * 224.0
    MTF_image = MTF_image.astype(np.uint8)
    img = Image.fromarray(MTF_image[0])  # Assuming MTF_sample shape is (1, 1, 250)   

    # Define the output folder
    output_folder = 'MTF_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the image with a unique filename
    image_filename = f'{output_folder}/MTF_image_AE_T{i}.png'
    img.save(image_filename, 'PNG')


    # Append the file path to the list
    MTF_images_paths_AE_T.append(image_filename)






#SPINDLE ACOUSTIC EMISSION 
    
#Updating the code to save the images and store their file paths 
    
#Spindle Acoustic Emission 

#Now adding vib_table     
C1AE_S = Case_1.AE_Spindle 

#reshape the dataframe to a 2D dataframe 
C1AE_S  = C1AE_S.values.reshape(-1)
C1AE_S_2D = C1AE_S.reshape(-1,1)


C1AE_S_2D = np.asfarray(C1AE_S_2D)
print(C1AE_S_2D.shape)


#define the step size for samples
sample_size = 250

#splitting the case 1 dataframe into the different samples 
samples = [C1AE_S_2D[i:i+sample_size] for i in range(0,len(C1AE_S_2D), sample_size)]

#Get recurence plots for all time series 
MTF = MarkovTransitionField(n_bins=30)

#storing the file paths

MTF_images_paths = []

# for loop to append each gramian angular field to a list 
for i, sample in enumerate(samples) :
    
    sample= sample.flatten()
    sample = np.array([sample]) 
    
    MTF_sample = MTF.fit_transform(sample)
    print(MTF_sample)
   
   #convert MTF to image , mtf values scaled between 0 and 224 using max scaling  to fit, the data type is converted after scaling to ensure its compatible  with image generation  
    MTF_image = (MTF_sample - MTF_sample.min()) / (MTF_sample.max() - MTF_sample.min()) * 224.0
    MTF_image = MTF_image.astype(np.uint8)
    img = Image.fromarray(MTF_image[0])  # Assuming MTF_sample shape is (1, 1, 250)   

    # Define the output folder
    output_folder = 'MTF_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the image with a unique filename
    image_filename = f'{output_folder}/MTF_image_{i}.png'
    img.save(image_filename, 'PNG')


    # Append the file path to the list
    MTF_images_paths.append(image_filename)
  
 

  
      
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
'''


##### other method for plotting mtf outisde the loop 
# attempting to plot mtf in one figure 

fig, axes = plt.subplots(nrows=len(MTF_images), figsize=(8, 4 * len(MTF_images)))

for i, MTF_sample in enumerate(MTF_images):
    ax = axes[i] if len(MTF_images) > 1 else axes  # Handling multiple subplots
    ax.imshow(MTF_sample[0], cmap='rainbow', origin='lower')
    ax.set_title(f'Sample {i+1}')

plt.tight_layout()
plt.show()



####
'''



''' old stuff before 8/12/23

# import case 1 
Case_1 = pd.read_csv('Mill_Case1_gma.csv')
C1smcAC = Case_1.smcAC

# reshape the dataframe to a 2d dataframe 
C1smcAC = C1smcAC.values.reshape(-1)
C1smcAC_2D = C1smcAC.reshape(-1,1)

# convert case 1 dataframe into a matrix/ array 
#C1smcAC_2D = C1smcAC_2D.to_numpy()

C1smcAC_2D = np.asfarray(C1smcAC_2D)
print(C1smcAC_2D.shape)
#C1 = Case_1.to_numpy()

#convert to 
#C1smcAC_reshaped = np.reshape(C1smcAC_2D,(500,306))



sample_size = 1000

# splitting the case 1 dataframe into the different samples 
samples = [C1smcAC_2D[i:i+sample_size] for i in range(0,len(C1smcAC_2D), sample_size)]

# to experiment with different bin sizes 
mtf = MarkovTransitionField(n_bins= 100)


MTF_Images = []

for sample in samples: 
    
    MTF_sample = mtf.fit_transform(sample)
    MTF_Images.append(MTF_sample)


NUM_images = len(MTF_Images)

#creating a figure with the width 12 and height equal to 6 times the num images,  

fig = plt.figure(figsize=(10,10))

grid = ImageGrid(fig, 111, nrows_ncols=(10, 10), axes_pad=0.1, share_all=True,cbar_mode='single')
for i, ax in enumerate(grid):
    im = ax.imshow(MTF_sample[i], cmap='rainbow', origin='lower', vmin=0., vmax=1.)
grid[0].get_yaxis().set_ticks([])
grid[0].get_xaxis().set_ticks([])
plt.colorbar(im, cax=grid.cbar_axes[0])
ax.cax.toggle_label(True)

fig.suptitle("Markov transition fields for the 50 time series in the "
             "'GunPoint' dataset", y=0.92)

plt.show()


'''