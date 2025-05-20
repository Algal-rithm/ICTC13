import cv2 as cv2 # tried using 4.10
import skimage as skimage
from skimage import io, data, color, filters
#from skimage.color import rgb2ycbcr
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os
from skimage.exposure import rescale_intensity
import argparse
import shutil

print (cv2.__version__)

def createFileList(myDir, formats=['.tif', '.png', '.tiff']):
    fileList = []
    #print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            for format in formats:
                if name.endswith(format):
                    fullName = os.path.join(root, name)
                    fileList.append(fullName)
    return fileList

    
def hp_filter(img):
    
    # convert image to gray scale and float32
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
    
    # centered
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = rows//2 , cols//2

    # create a mask first, center square is 1 (now 0), remaining all zeros
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 0

    # apply mask and inverse DFT
    fshift = dft_shift*mask
    
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    
    # magnitude specturm
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1]) 
    
    # normalize
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)
    
    # fly
    # print(f"Grayscale Image Shape: {img.shape}")
    # print(f"DFT Shape: {dft.shape}")
    # print(f"DFT Shift Shape: {dft_shift.shape}")
    # print(f"Mask Shape: {mask.shape}")
    # print(f"Filtered Image Max Value: {np.max(img_back)}")
    # print(f"Filtered Image Min Value: {np.min(img_back)}")
    
    # beetle
    if np.max(img_back) == 0:
        print("*warning completely black*")
    else:
        print("is fine")
    
    return img_back

def customFilter(img):
    img_back = hp_filter(img)
    
    myData = img_back.flatten()
    myMean = np.mean(myData)
    myStd = np.std(myData)
    threshold = 3.5e+08
    
    myThres = myMean - (2*myStd)
    threshold = myThres
    upper = 0
    img_f = np.where(img_back>threshold, upper, img_back)
    
    return img_f
    
def isItBlank(img):
    #Here's the place to customise. A blank file here is where every value in the array is the same
      
    img_f = customFilter(img)
    img_f = (img_f * 255).astype(np.uint8)
    
    if img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
        
    img_f_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # threshold based on the pixel values
    threshold = 2 
    num_unique = len(np.unique(img_f_rgb))
    
    # grasshopper
    print(f"Number of unique pixel values in filtered image: {num_unique}")
    print(f"Max pixel value in filtered image: {np.max(img_f_rgb)}")
    
    # Check if the number of unique pixel values is low or if the maximum value is below the threshold
    if num_unique < 100 or np.max(img_f_rgb) < threshold:
        print("but nobody came ...")
        return True
    else:
        print("wow, many algae")
        return False
        

if __name__ == "__main__":
    pwidth = 448
    pheight = 448
    hstride = 0
    vstride = 0
    
    # needs TWO folders
    parser = argparse.ArgumentParser(description="Takes a folder of images and creates patches from them")
    parser.add_argument("-b", "--blankPatches", help="the source directory containing the patches that will be checked")
    parser.add_argument("-c", "--correspondingPatches",   help="the source directory that will also have files removed")

    args = parser.parse_args()

    if args.blankPatches:
        print("Getting pictures".format(args.blankPatches))
        imageNames = createFileList(args.blankPatches)
        print("*****************************************")
        print("Getting patches from the following files:")
        print(imageNames)
        print("*****************************************")
    if args.correspondingPatches:
        destDir = args.correspondingPatches
        print("Dependent folder that will *NOT* be modified {}.".format(args.correspondingPatches))

    for imageName in imageNames:
        imageBaseName = os.path.split(os.path.basename(imageName))[1]
        dependentFileName = os.path.join(destDir, imageBaseName)
        print(imageName)
        print(dependentFileName)
        img_mat = cv2.imread(imageName)
        img = np.asarray(img_mat)
        iheight, iwidth = img.shape[:2]
        # print("Image {} is {} by {}".format(imageName, iwidth, iheight))

        blank = isItBlank(img)
        if isItBlank(img):
            # remove the file
            print ("deletus")
            os.remove(imageName)
            
            # stopped the deletion of the corresponding file
            # os.remove(dependentFileName)
        else:
            print ("this is a good one")
            
