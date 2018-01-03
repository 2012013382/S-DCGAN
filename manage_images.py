from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.misc import imread, imresize
from os import  walk
from os.path import join
#CUB-200-2011 is a fine-grained data set, you can also use simpler datasets to get better generation.
CUB_image_path = 'CUB_200_2011/images/'
#For ease of understanding, I fix the input shape. 
IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = 3
def read_images_and_attributes(image_path, img_height=64, img_width=64, img_channels=3):
    image_set = []

    n = 0
    for root, _, file_names in walk(image_path):
        if len(file_names) == 0:
            continue
        images = np.zeros((len(file_names), img_height, img_width, img_channels), dtype=np.float32)
        for i in range(len(file_names)):
            img = imread(join(root, file_names[i]))
            img = imresize(img, (img_height, img_width))  # resize image to 224x224
            if len(img.shape) < 3:
                timg = img
                img = np.zeros((img_height, img_width, img_channels), dtype=np.float)
                img[:, :, 0] = timg
                img[:, :, 1] = timg
                img[:, :, 2] = timg
            img.astype(np.float32)
            images[i, :, :, :] = img
            images[i, :, :, :] = (images[i, :, :, :] - 127.5) / 127.5#Range[-1, 1] for the network input.
        image_set.append(images)
        print('Load set %d'%(n + 1))
        n += 1
    return image_set#Return shape is list(200, [len(images), img_height, img_width, img_channels])

def data_processing(batch_size=32):
    image_set = read_images_and_attributes(CUB_image_path, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    images = []
	#Convert data into numpy array
    for i in range(len(image_set)):
        for j in range(len(image_set[i])):
            images.append(image_set[i][j, :, :, :])
    np_image_set = np.array(images, np.float32)

    #Prepare data batches, I know there are better ways to get data batches, but I sample them directly for ease of understanding.
    pointer = 0
    batch_image_set = []
    for i in range(len(np_image_set) // batch_size):
        batch_images = np_image_set[pointer:pointer + batch_size, :, :, :]
        batch_image_set.append(batch_images)
        pointer = pointer + batch_size
    print('images num is %d'%(len(np_image_set)))
    return batch_image_set, len(np_image_set)#Default shape is list(len(np_image_set) // batch_size,[batch_size, img_height, img_width, img_channels])

def merge_images(images):
    #Merge batch size images in to one image.
    h = 8#image_num = 128 = (8*16)
    w = 16
    img = np.zeros((IMG_HEIGHT * h, IMG_WIDTH * w, IMG_CHANNELS), np.float32)
    count = 0
    for i in range(h):
        for j in range(w):
            img[IMG_HEIGHT * i : IMG_HEIGHT * (i+1), IMG_WIDTH * j : IMG_WIDTH * (j+1), :] = images[count, :, :, :]
            count += 1
    return img



