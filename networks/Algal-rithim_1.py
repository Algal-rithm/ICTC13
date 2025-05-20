import numpy as np
import os
import random
import shutil
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score

from tabulate import tabulate
from PIL import Image
import argparse


def modelArchitecture(input_shape, num_classes, architectureNumber):
    print(num_classes)
    if architectureNumber == 0:
        modelName = "ThreeConv4x4"
        model = Sequential()
        model.add(Conv2D(32, (4, 4), input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    if architectureNumber == 1:
        modelName = "ThreeConv3x3"
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    if architectureNumber == 2:
        modelName = "AlexNet_1"
        model = Sequential()
        model.add(Conv2D(96, (11, 11), strides=(4, 4), padding='valid', input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization())

        model.add(Conv2D(256, (11, 11), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization())

        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))


        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Model dropout to prevent overfitting
        model.add(Dropout(0.4))
        model.add(BatchNormalization())
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())
        model.add(Dense(1000))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))     
    if architectureNumber == 3:
        modelName = "VGG-16"
        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(512, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(512, (3, 3)))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(512, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(512, (3, 3)))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    if architectureNumber == 4:
        # previous stride of 1 too big, changed to stride 4
        modelName = "TwoConv3x3_1"
        model = Sequential()
        #model.add(Conv2D(32, kernel_size=(3, 3),
        #         activation='relu',
        #         input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), strides=(2, 2), input_shape=input_shape))
        model.add(Activation('relu'))

        #model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        #model.add(BatchNormalization())
        model.add(Dense(num_classes, activation='softmax'))
    if architectureNumber == 5:
        # previous stride of 1 too big, changed to stride 4
        modelName = "TwoConv3x3_2"
        model = Sequential()
        #model.add(Conv2D(32, kernel_size=(3, 3),
        #         activation='relu',
        #         input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), strides=(4, 4), input_shape=input_shape))
        model.add(Activation('relu'))

        #model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        #model.add(BatchNormalization())
        model.add(Dense(num_classes, activation='softmax'))
        
    if architectureNumber == 6:
        # previous stride of 1 too big, changed to stride 4
        modelName = "TwoConv3x3_3"
        model = Sequential()
        #model.add(Conv2D(32, kernel_size=(3, 3),
        #         activation='relu',
        #         input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), strides=(1, 1), input_shape=input_shape))
        model.add(Activation('relu'))

        #model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        #model.add(BatchNormalization())
        model.add(Dense(num_classes, activation='softmax'))
    if architectureNumber == 7:
        modelName = "AlexNet_2"
        model = Sequential()
   
        # less filters
        model.add(Conv2D(64, (11, 11), strides=(4, 4), padding='valid', input_shape=input_shape))
        model.add(Activation('relu'))
    
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization())

        # less filters and smaller kernel size
        model.add(Conv2D(128, (5, 5), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
    
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization())

        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        # less fully connected units and removed some dense layers
        model.add(Flatten())
        model.add(Dense(2048))  # reduced from 4096
        model.add(Activation('relu'))
        model.add(Dropout(0.5))  # increased dropout
        model.add(BatchNormalization())
        model.add(Dense(1024))  # reduced from 4096
        model.add(Activation('relu'))
        model.add(Dropout(0.5))  # increased dropout
        model.add(BatchNormalization())
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    if architectureNumber == 8: 
    # model from: www.kaggle.com/code/yashmahajan11/efficientnet-b7
    ## pre-trained on ImageNet
        modelName = "EfficientNetB0"  
        
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=input_shape, pooling='max') 
        x = base_model.output 
        x = BatchNormalization(axis=-1)(x) #, momentum=0.99, epsilon=0.001)(x) 
        x = Dense(64, kernel_regularizer=regularizers.l2(0.02), activity_regularizer=regularizers.l1(0.006), bias_regularizer=regularizers.l1(0.006), activation='relu')(x) # regularization decreased, was 256, 0.016, 0.006
        x = Dropout(rate=0.6)(x) #, seed=123)(x) # dropout decreased, was 0.45
        output = Dense(num_classes, activation='softmax')(x) 
        model = Model(inputs=base_model.input, outputs=output) 
        
        # to freeze the base model layers
        for layer in base_model.layers[-40:]: 
            layer.trainable = True

    print(modelName)
    print(model.summary())
    print("Number of parameters")
    print(model.count_params())
    return model, modelName

# change format '.tif' or '.png'
def createFileList(myDir, format='.png'):
    fileList = []
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

def processImage(inFile, outFile, processImageMethod, dim=224): # dim was 128
    if processImageMethod == 0:
        return processImage0(inFile, outFile)
    if processImageMethod == 1:
        return processImage1(inFile, outFile, dim)
    if processImageMethod == 2:
        return processImage2(inFile, outFile, dim)
    if processImageMethod == 3:
        return processImage3(inFile, outFile, dim)
    if processImageMethod == 4:
        return processImage4(inFile, outFile, dim)
    if processImageMethod == 5:
        return processImage5(inFile, outFile, dim)
    if processImageMethod == 6:
        return processImage6(inFile, outFile, dim)


def processImage0(inFile, outDir):
    #print("Straight Copy")
    shutil.copy2(inFile, outDir)

def processImage1(inFile, outDir, dim=224):
    #print("Copy and resize with Lanczos")
    im = Image.open(inFile)
    size = dim, dim
    img = im.resize(size, Image.LANCZOS)
    d, b = os.path.split(inFile)
    outName = os.path.join(outDir, b)
    img.save(outName)

def processImage2(inFile, outDir, dim=224):
    #print("Copy, normalize, resize")
    d, b = os.path.split(inFile)
    outName = os.path.join(outDir, b)
    im = cv2.imread(inFile, 0)
    im_norm = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX)
    size = dim, dim
    resized = cv2.resize(im_norm, size, interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(outName, resized)

def processImage3(inFile, outDir, dim=224):
    #print("High pass filter w/ custom filter")
    d, b = os.path.split(inFile)
    outName = os.path.join(outDir, b)
    im = cv2.imread(inFile, cv2.IMREAD_GRAYSCALE)
    im_norm = customFilter(im)
    size = dim, dim
    
    if im_norm is not None:
        resized = cv2.resize(im_norm, size, interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(outName, resized)

def processImage4(inFile, outDir, dim=224):
    #print("Custom filter 2 (morph. filter)")
    d, b = os.path.split(inFile)
    outName = os.path.join(outDir, b)
    im = cv2.imread(inFile, 0)
    im_norm = customFilter2(im)
    size = dim, dim
    resized = cv2.resize(im_norm, size, interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(outName, resized)

def processImage5(inFile, outDir, dim=224):
    # print("GrabCut")
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # image sizes ...
    # rect = (52, 52, 150, 150) # This was Ismael's for resized images
    # rect = (480, 270, 960, 540)
    rect = (1, 1, 448, 448)


    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 8, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    return img
    
def processImage6(inFiles, outDir, dim = 224):
    # print("Watershed")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)

    fg = cv2.erode(thresh, kernel, iterations=1)

    bgt = cv2.dilate(thresh, kernel, iterations=1)
    ret, bg = cv2.threshold(bgt, 1, 128, 1)

    marker = cv2.add(fg, bg)

    marker32 = np.int32(marker)
    cv2.watershed(img, marker32)
    m = cv2.convertScaleAbs(marker32)
    ret, thresh = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res = cv2.bitwise_and(img, img, mask=thresh)
    return res
    

def hp_filter(img):

    # convert image to gray scale and float32
    img_f = np.float32(img)
    
    if len(img.shape) == 2:
        img = img_f
    elif len(img.shape) == 3:
            img = cv2.cvtColor(img_f, cv2.COLOR_BGR2GRAY) 
    else:
        print("error")
        return None
    
    ## alternative for gray scale and float32
    # img = np.full((448,448,3), 12, np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # centered
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = rows//2 , cols//2

    # create a mask first, center square is 1 (now 0), remaining all zeros
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow-10:crow+10, ccol-10:ccol+10] = 0

    # apply mask and inverse DFT
    fshift = dft_shift*mask
    
    #ladybug
    masked = cv2. magnitude(fshift[:,:,0], fshift[:,:,1])
    cv2.imwrite('masked_dft.png', masked)
    
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    
    # magnitude specturm
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1]) 
    cv2.imwrite('hp_filter_output.png', img_back)
    
    # normalize
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)
    
    return img_back

def customFilter(img):
    img_back = hp_filter(img)
    
    if img_back is None:
        return None
    
    # *keep just in case*
    # myData = img_back.flatten()
    # myMean = np.mean(myData)
    # myStd = np.std(myData)
    # threshold = 3.5e+08
    # myThres = myMean - (2 * myStd)
    
    # *threshold* increase = fewer pixels (suggested 0.6), decrease = more features (suggested 0.4)
    threshold = 0.3 * np.max(img_back) # adjust as needed
    upper = 0
    img_f = np.where(img_back>threshold, upper, img_back)
    
    cv2.imwrite('thresholded_image.png', img_f)
    
    return img_f

def customFilter2(img):
    gray = img
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    upper = 0
    img_filtered = np.where(sure_bg==0, upper, img)
    
    return img_filtered

def splitIntoTestTrain(src, dst, processImageMethod=0):
    print(src)
    d, classname = os.path.split(src)
    fileList = createFileList(src)
    #10% randomness
    numSamples = len(fileList) // 10
    # if numSamples < 10:
    # numSamples = 10
    shuffledFiles = fileList
    random.shuffle(shuffledFiles)
    print(shuffledFiles)
    test = shuffledFiles[0:numSamples]
    train = shuffledFiles[numSamples:]

    testdir = "{}/test/{}".format(dst, classname)
    os.makedirs(testdir)
    traindir = "{}/train/{}".format(dst, classname)
    os.makedirs(traindir)

    for myFile in test:
        processImage(myFile, testdir, processImageMethod, img_w)

    for myFile in train:
        processImage(myFile, traindir, processImageMethod, img_w)
    return len(test), len(train)

def splitIntoTestTrainAndValidate(src, dst, processImageMethod=0):
    print(src)
    d, classname = os.path.split(src)
    fileList = createFileList(src)
    #10% randomness (10% test, 10% validate, the rest train)
    numSamples = len(fileList) // 10
    # numSamples < 10:
    # numSamples = 10
    numSamples2 = int(numSamples*2)
    shuffledFiles = fileList
    random.shuffle(shuffledFiles)
    print(shuffledFiles)
    test = shuffledFiles[0:numSamples]
    val = shuffledFiles[numSamples:numSamples2]
    train = shuffledFiles[numSamples2:]

    testdir = "{}/test/{}".format(dst, classname)
    os.makedirs(testdir)
    traindir = "{}/train/{}".format(dst, classname)
    os.makedirs(traindir)
    valdir = "{}/val/{}".format(dst, classname)
    os.makedirs(valdir)

    for myFile in test:
        processImage(myFile, testdir, processImageMethod, img_w)
    for myFile in val:
        processImage(myFile, valdir, processImageMethod, img_w)
    for myFile in train:
        processImage(myFile, traindir, processImageMethod, img_w)
    return len(test), len(train)


def decideDataGeneration(dataGenType=0):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip = True,
        zca_whitening = True,
        rotation_range = 180
    )
    if dataGenType == 1:
        datagen= ImageDataGenerator(
            preprocessing_function=preprocess_input,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            rotation_range=90,
            brightness_range=[1.0, 1.15],
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
        )


    return datagen

if __name__ == "__main__":
    batch_size = 64
    img_w, img_h = 224, 224 # was 128?

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print("Not enough GPU hardware devices available")
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
        #tf.keras.backend.set_session(tf.Session(config=config))
        

    ## I think this is just listing the obvious?
    ### it is NOT just stating the obvious apparently
    num_classes = 2
    srcpics0 = "/home/researcher/exp1/NodHar_1"
    srcpics1 = "/home/researcher/exp1/NodHar_73104"
    datadir = "/home/researcher/exp1"
    trainSamples = 15051 # change
    testSamples = 1880 # change
    valSamples = 1880 # change
    doShuffle = False
    dataGenType = 0
    processImageMethod = 0
    architectureNumber = 0 
    optimiser = "sgd"

    parser = argparse.ArgumentParser(description='Tests a specific model with some options')
    parser.add_argument("-b", "--before_gen", help="the before treatment directory")
    parser.add_argument("-a", "--after_gen",  help="the after treatment directory")
    parser.add_argument("-d", "--data_dir",   help="the directory to store the data")
    parser.add_argument("-f", "--process_files",   help="0: existing; 1: resize only; 2: normalize; 3: hp filter 4: morph", type=int)
    parser.add_argument("-g", "--data_gen",   help="the datagen type (0 for mine, 1 for Ismaels)", type=int)
    parser.add_argument("-m", "--model_number",   help="network number: 0:4x4 network; 1:LeNet5ish (3x3); 2:AlexNet; 3:VGG16 4:MNIST_99", type=int)
    parser.add_argument("-o", "--optimiser",   help="one of sgd, rmsprop, adam, adagrad")
    parser.add_argument("-e", "--epochs",   help="Number of epochs to run for", type=int)
    args = parser.parse_args()

    if args.before_gen:
        srcpics0 = args.before_gen
    if args.after_gen:
        srcpics1 = args.after_gen
    if args.data_dir:
        datadir = args.data_dir
    if args.process_files:
        print("Shuggle")
        doShuffle = False
        processImageMethod = args.process_files
    if args.data_gen:
        dataGenType = args.data_gen
    if args.model_number:
        architectureNumber = args.model_number
    if args.optimiser:
        optimiser = args.optimiser
    if args.epochs:
        epochs = args.epochs

    if architectureNumber > 1:
        img_w = 224
        img_h = 224
        
        
    doShuffle = False
    trainSamples = 0
    testdir = "{}/test".format(datadir)
    traindir = "{}/train".format(datadir)
    valdir = "{}/val".format(datadir)   
    if doShuffle:
        print("Doing shuffle this might take ages")

        if os.path.exists(testdir):
            shutil.rmtree(testdir)
        os.makedirs(testdir)
        if os.path.exists(traindir):
            shutil.rmtree(traindir)
        os.makedirs(traindir)
        if os.path.exists(valdir):
            shutil.rmtree(valdir)
        os.makedirs(valdir)

        te, tr = splitIntoTestTrainAndValidate(srcpics0, datadir, processImageMethod)
        trainSamples += tr
        testSamples += te
        te, tr = splitIntoTestTrainAndValidate(srcpics1, datadir, processImageMethod)
        trainSamples += tr
        testSamples += te
    else:
        print("no shuffle because {} already has files".format(traindir))
        # trainSamples = len([name for name in os.listdir(os.path.join(traindir, srcpics0))])
        # trainSamples += len([name for name in os.listdir(os.path.join(traindir, srcpics1))])
        # testSamples = len([name for name in os.listdir(os.path.join(testdir, srcpics0))])
        # testSamples += len([name for name in os.listdir(os.path.join(testdir, srcpics1))])
        # valSamples = len([name for name in os.listdir(os.path.join(valdir, srcpics0))])
        # valSamples += len([name for name in os.listdir(os.path.join(valdir, srcpics1))])
        
        # *need to change every new dataset*
        trainSamples = 15051 # change
        testSamples = 1880 # change
        valSamples = 1880 # change
        
    print("The samples are {}, {}, {}".format(trainSamples, testSamples, valSamples))



    if K.image_data_format() == 'channels_first':
        print("channels first")
        input_shape = (3, img_w, img_h)
    else:
        print("channels last")
        input_shape = (img_w, img_h, 3)

    # Note that there isn't any data augmentation
    datagen = decideDataGeneration(dataGenType)

    train_it = datagen.flow_from_directory('{}/train'.format(datadir),
                                           color_mode='rgb',
                                           target_size=(img_w, img_h),
                                           batch_size=batch_size,
                                           class_mode="categorical",
                                           shuffle=True)

    test_it = datagen.flow_from_directory('{}/test'.format(datadir),
                                           color_mode='rgb',
                                           target_size=(img_w, img_h),
                                           batch_size=batch_size,
                                           class_mode="categorical",
                                           shuffle=False) # was false?

    val_it = datagen.flow_from_directory('{}/val'.format(datadir),
                                           color_mode='rgb',
                                           target_size=(img_w, img_h),
                                           batch_size=batch_size,
                                           class_mode="categorical",
                                           shuffle=True)


    # And now the model....
    modelArchitectures = [0, 1, 2, 7] # 0: 4x4, 1: LeNet5ish, 2: AlexNet_modified, 7: AlexNet_modified_less
    epochNumbers = [100]
    #optimisers = ["rmsprop", "sgd", "adam", "adagrad"]
    optimisers = ["sgd", "adam"]

    bestf1 = 0
    bestNetwork = "unknown"
    bestMCC = 0
    bestAccuracy = 0
    bestf1model = 0
    bestMCCmodel = 0
    bestAccuracyModel = 0

    listOfTests = [ [0, 100, "sgd" ],
                    [0, 100, "adam" ],

                    [1, 100, "sgd" ],
                    [1, 100, "adam" ],

                    [2, 100, "sgd" ],
                    [2, 100, "adam" ],


                    [7, 100, "sgd" ],
                    [7, 100, "adam" ],

     ]
    
    doTheSaving = False
    listOfTests = [[architectureNumber, epochs, optimiser],]
    resultsList = []

    #resultsList.append(("archNo", "epochs", "opt", "f1", "acc", "mcc"))
    #for architectureNumber in modelArchitectures:
        #for epochs in epochNumbers:
            #for optimiser in optimisers:
    
    for test in listOfTests:

        architectureNumber = test[0]
        epochs = test[1]
        optimiser = test[2]

        print("{}".format(optimiser))
        
        if "sgd1" in optimiser:
            # trying something with SGD.
            print("switching to optimer SGD1")
            mySgd = optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False)
            optimiser = mySgd

        model, modelName = modelArchitecture(input_shape, num_classes, architectureNumber)

        # catagorical cross entrapy 
        print("Compiling the model: {}".format(modelName))
        model.compile(loss='categorical_crossentropy', # was mse
                      optimizer=optimiser,
                      metrics=[metrics.categorical_accuracy])

        # calculate stepsPerEpoch
        stepsPerEpoch = trainSamples // batch_size
        if stepsPerEpoch < 10:
            stepsPerEpoch = 235 # change
        
        print("Calculating batch size {} {} {}".format(trainSamples, batch_size, stepsPerEpoch))

        valSteps = valSamples // batch_size

        early = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0.01, patience=5, verbose=1, mode='auto')

        print("Fitting the model: {}".format(modelName))
        # was fit_generator
        model.fit_generator(
            train_it,
            steps_per_epoch=stepsPerEpoch,
            epochs=epochs,
            validation_data=test_it,
            validation_steps=valSteps,
            #callbacks=[early]
            )
        
        # predictions
        probabilities = model.predict(test_it)
        y_pred = np.argmax(probabilities, axis=-1)
        y_true = test_it.classes

        # confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("The stats for {} after {} epochs with {} opt:".format(modelName, epochs, optimiser))
        f1 = f1_score(y_true, y_pred, average='micro')
        f1_all = f1_score(y_true, y_pred, average=None)
        mcc = matthews_corrcoef(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred, normalize=True)
        
        # printing metrics
        print(cm)
        print("f1 micro = {} and all {} ".format(f1, f1_all))
        print("accuracy = {}".format(acc))
        print("mcc = {}".format(mcc))
        
        # the results ...
        myResults = (architectureNumber, epochs, optimiser, f1, acc, mcc)
        resultsList.append(myResults)
        
        # print out the results as we go ...
        print(tabulate(resultsList, headers=["archNo", "epochs", "opt", "f1", "acc", "mcc"]))
        print(cm[0][0])

        theseResults = [(datadir, dataGenType, modelName, epochs, optimiser, cm[0][0], cm[0][1], cm[1][0], cm[1][1], f1, f1_all, acc, mcc)]
        print(theseResults)
        
        #tryit = tabulate(theseResults, headers=["datadir", "augmentation", "Model", "epochs", "opt", "cm00", "cm01", "cm10", "cm11", "f1"])
        
        # saving the results to .txt
        with open('output.txt', 'a') as f:
            print(tabulate(theseResults, headers=["datadir", "augmentation", "Model", "epochs", "opt", "cm00", "cm01", "cm10", "cm11", "f1", "f1_all", "acc", "mcc"]), file=f)
            print("\n", file=f)
        with open('output_alt.txt', 'a') as f:
            print(theseResults)
            print("\n", file=f)


        if doTheSaving:
            saveModel = False

            if f1 > bestf1:
                bestf1 = f1
                bestf1model = model
                saveModel = True

            if mcc > bestMCC:
                bestMCC = mcc
                bestMCCmodel = model
                saveModel = True

            if acc > bestAccuracy:
                bestAccuracy = acc
                bestAccuracyModel = model
                saveModel = True

            # save model to file
            if saveModel:
                modelBaseFilename = "arch{}_epochs{}_opt{}".format(architectureNumber, epochs, optimiser)
                print("Saving to {}".format(modelBaseFilename))
                model_json = model.to_json()
                with open("{}.json".format(modelBaseFilename), "w") as json_file:
                    json_file.write(model_json)
                model.save_weights("{}.h5".format(modelBaseFilename))
        clear_session()

    print("The overall results:")
    print(resultsList)
    
