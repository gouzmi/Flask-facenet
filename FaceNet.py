import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from imageio import imread
from skimage.transform import resize
from scipy.spatial import distance
from matplotlib import pyplot as plt
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances

cascade_path = 'model/cv2/haarcascade_frontalface_alt2.xml'


def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

images_error = []

def scale(img_path,margin=10):
    image_size = 160
    aligned_images=[]
    cascade = cv2.CascadeClassifier(cascade_path)
    img = imread(img_path)
    faces = cascade.detectMultiScale(img,
                                     scaleFactor=1.1,
                                      minNeighbors=3)
    (x, y, w, h) = faces[0]
    cropped = img[y-margin//2:y+h+margin//2,
                    x-margin//2:x+w+margin//2, :]
    aligned = resize(cropped, (image_size, image_size), mode='reflect')
    aligned_images.append(aligned)
    return np.array(aligned_images)



def facenet(img_path, model, margin,graph):
    
    # try:
        aligned_images = prewhiten(scale(img_path,margin))
        #embs = model.predict(aligned_images)
        print('------------')
        print(aligned_images.shape)
        if aligned_images.shape[-1]==4:
            aligned_images=aligned_images[:,:,:,:-1]
        with graph.as_default():
            embs = model.predict(aligned_images)
        embs = l2_normalize(embs)
        return embs
    # except:
    #     images_error.append(img_path)
    #     print(img_path,' : visage non détecté')


# values = pd.read_csv('notebook/features.csv')
# photo = '../data/images/dimsa.PNG'
# aligned_images = prewhiten(scale(photo))
# values['result']=euclidean_distances(values.iloc[:,1:],facenet(photo))
# # imread(values.sort_values(by='result').iloc[0,0])