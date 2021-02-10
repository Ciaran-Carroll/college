'''
RE4017 - Machine Vision

Kevin Burke (14155893)
Paul Lynch (16123778)
Ciaran Carroll (13113259)
Qicong Zhang (16069978)

Reconstruction of image from given sinogram
'''


import scipy.fftpack as fft
import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import rotate

import imutils     

def build_laminogram(radonT):
    "Generate a laminogram by simple backprojection using the Radon Transform of an image, 'radonT'."
    laminogram = np.zeros((radonT.shape[1],radonT.shape[1]))
    dTheta = 180.0 / radonT.shape[0]
    for i in range(radonT.shape[0]):
        temp = np.tile(radonT[i],(radonT.shape[1],1))
        temp = rotate(temp, dTheta*i)
        laminogram += temp
    return laminogram

sinogram = imutils.imread('sinogram.png')           #Read in the sinogram image
imutils.imshow(sinogram, title="Sinogram Image")    #Show the original Sinogram image
sino_lamino = build_laminogram(sinogram)            #Build the backprojections of sinogram
imutils.imshow(sino_lamino, title="Reconstructed Image from Backprojections") #Show the reconstructed image