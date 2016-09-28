#Rescaling the pixel intensity from 0-255 to 0-1
#Creating a data set reduced to w=8,  h= 8. 
#Reshaping it an image per row: 8x8x3 = 1x192
#Create a data set with images from both 'Dad' and 'Mom'

from os import listdir
from os.path import join
import numpy as np
import cv2

#1: Rescale pixel intensity
def rescalePx(img_array):
    return img_array*1./255

#2: resize
def resizePx(img_array):
    return cv2.resize(img_array,size,interpolation=cv2.INTER_AREA)

#3: reshaping
def reshapePx(img_array):
    return img_array.reshape((1,-1))

# 1-3: Combine all three
def aggTransform(img_array):
    return reshapePx(resizePx(rescalePx(img_array)))

                
# Create an array for the images:
def imgDataset(file_list,dir_path,label):
    nrows = len(file_list)
    images_array = np.full((nrows,(size[0]*size[1]*3)+1),label,dtype=np.float32)
    for ind,afile in enumerate(file_list):
        full_path = join(dir_path,afile)        
        img = cv2.imread(full_path)
        images_array[ind][:-1] = aggTransform(img)
        
    return images_array 

def createArray(apath,folder_name,label):
    cropped_path = join(apath,folder_name) + '/'
    files = listdir(cropped_path)
    return imgDataset(files,cropped_path,label)
# Save the image to 8x8 (wxh) pixels
size = (8,8)

cropped_path = 'D:/Pictures/'

dad_array = createArray(cropped_path,'Dad_cropped',0)
mom_array = createArray(cropped_path,'Mom_cropped',1)
child_array = createArray(cropped_path,'Child_cropped',2)

print dad_array.shape, mom_array.shape, child_array.shape
# Stack the arrays and save it to disk:
all_arrays = np.vstack((dad_array,mom_array,child_array))
np.savetxt('Dad_Mom_Child_large.csv',all_arrays,delimiter=',')

