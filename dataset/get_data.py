import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

def get_image_path(
    data_root:str,                              # filr root of data to read
    )-> list:

    image_paths = []                            # initialize a list to store the image files' path
    for root,dirs,files in os.walk(data_root):  # walk on the data root
        for name in files:
            if name.endswith('jpg') or name.endswith('png') or name.endswith('JPEG'):
                image_path = os.path.join(root,name)   # get the file path of an image
                image_paths.append(image_path)  # store the file path of an image 

    return image_paths                          # return the list of the image files' path

def load_image(
    image_paths: list,                          # the list of image files' path to load
    )-> np.ndarray:

    X_list = []                                 # initialize a list to stor the images
    for image_path in image_paths:
        img = Image.open(image_path)            # load an image
        img = transforms.CenterCrop((224, 224))(img)
        arr = np.array(img)                     # change the data type into np.ndarray
        arr = np.transpose(arr,(2,0,1))         # transpose the dimension from (H,W,C) to (C,H,W)
        X_list.append(arr)                      # store an image
    X_train = np.array(X_list)                  # get total train images

    X_train = normalize(X_train)                # normalize the pixel values to [0,1]

    return X_train                              # return the train images

def normalize(
    X: np.ndarray,                              # images to normalize (N,3,H,W)
    )-> np.ndarray: 

    x_max = X.max()                             # get max value of the images
    x_min = X.min()                             # get min value of the images

    return ((X-x_min)/(x_max-x_min)).astype(np.float32)  # return the normalized images