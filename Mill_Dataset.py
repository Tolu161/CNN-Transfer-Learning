# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Data Experimentation  - uploading matlab data file , splitting the data and generating statistical features from the data , 
# Using loadmat function from scipy library to load the dataset from matlab ,then use pandas dataframe to store the data in python 
#import loadmat
import scipy.io as spio
from scipy.io import loadmat

import pandas as pd
import statistics

#dataset is loaded into varible 
mill_dataset = spio.loadmat('mill.mat')

#access the 3d array from loaded data
data = mill_dataset['mill']



#since dataset is a dictionary to display the keys : 
#print(mill_dataset)
print(mill_dataset.keys())
print(mill_dataset['__header__'])
print(mill_dataset['__version__'])
print(mill_dataset['__globals__'])



data = mill_dataset['mill']


# Create an array datalist 
dataList = []

# Section the data for each case 
dataList.append(data[0][0:17])  # - case 1 /

'''
dataList.append(data[0][17:31]) # - case 2 /


dataList.append(data[0][31:45]) # - case 3 /
dataList.append(data[0][45:52]) # - case 4  
dataList.append(data[0][109:115]) # - case 5
dataList.append(data[0][115:116])     # - case 6
dataList.append(data[0][116:124])  # - case 7
dataList.append(data[0][124:130])    # - case 8/
dataList.append(data[0][52:61])      # - case 9
dataList.append(data[0][61:71])     # - case 10
dataList.append(data[0][71:94])    # - case 11
dataList.append(data[0][94:109])    # - case 12
dataList.append(data[0][130:145])   # - case 13
dataList.append(data[0][145:154])   # - case 14
dataList.append(data[0][154:161])   # - case 15
dataList.append(data[0][161:167])   # - case 16

'''


#create an empty list to store the tempdataframe dictionary 

temp_dataframe =[]
caseSamples = []

#the number of columns 
# to loop through each case loop through integer i in variable  case-i-data and append data to temporary dataframe 

    
    # for loop to loop through index CaseData[j] from 0 which is run 1 to maximum run for each case, determined by the length of each case 
for j in range(len(dataList)): 
        print(j)
        # looping through each case of the dataset case 1 to case 16
        caseSamples = dataList[j]
        print(len(caseSamples))
        #loop through the rows in the casesamples 
        for row in range(len(caseSamples)): 
            
            #print(row)
            # loop through the samples the values that have 9000 for smcAC to AESpindle  , extracting the sample data 
            for sample in range(9000): 
                
                #print(sample)
                # now append the array into the temporary dataframe. 
                 
                temp_dataframe.append([caseSamples[row][0][0], caseSamples[row][1][0], caseSamples[row][2][0], caseSamples[row][3][0], caseSamples[row][4][0], caseSamples[row][5][0], caseSamples[row][6][0], caseSamples[row][7][sample], caseSamples[row][8][sample], caseSamples[row][9][sample], caseSamples[row][10][sample], caseSamples[row][11][sample], caseSamples[row][12][sample]])
                

# create an empty dataframe with the columns  : 
print(len(temp_dataframe))
#use this one 
mill_df = pd.DataFrame(temp_dataframe, columns = ['case', 'run', 'VB', 'time', 'DOC', 'feed', 'material', 'smcAC','smcDC','vib_table','vib_spindle', 'AE_Table', 'AE_Spindle' ] )
print(len(mill_df))


                
# now to  append the tempdataframe  to the original total dataframe 


#use this : 

#print(mill_df)


#print(mill_df.VB)

#declare wear variable 

'''

for i in  range(1,len(mill_df.VB)): 
    value = mill_df.VB[i]
    previous_value = mill_df.VB[i-
    if value =='nan':
        if previous_value != 'nan': 
                mill_df.VB[i] = previous_value 
                mill_df.VB.replace(value, previous_value, inplace=True)
                print(previous_value)
    
    else:
        previous_value = value 

'''
#print(mill_df)         
      

print(len(mill_df.VB))

for i in range(1, len(mill_df.VB)):
    
    value = mill_df.VB[i]
    
 
    
    previous_value = mill_df.VB[i - 1]

    if pd.isnull(value):
       
        
        if not pd.isnull(previous_value):
                
            mill_df.loc[i,'VB'] = previous_value
            
            
     

    #alternative approach:
#mill_df['VB'] = mill_df['VB'].fillna(method='ffill')


print(mill_df.VB[152999])


# add a command to save datafram to csv file- for case 1 file 

# use while loops more for unknown lengths 
# use for loop with certain number of increasing incremeents put that value in the main statmenet 
# confirm the use of statistical features in pretrained networks 
#look at best format to prepare the model for deep learning - refer to papers 
 

# exporting the dataframe to a csv : 

mill_df.to_csv('Mill_Case1.csv')




# for testing code for doing case 9 

















