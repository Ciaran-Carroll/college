'''
RE4017 - Machine Vision

Kevin Burke (14155893)
Paul Lynch (16123778)
Ciaran Carroll (13113259)
Qicong Zhang (16069978)

Reconstruction of image from sinogram with Hamming windowed ramp filter applied

'''


#%matplotlib
import scipy.fftpack as fft
import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import rotate

import imutils

sinogram = imutils.imread('sinogram.png')      #Read in the sinogram image

def build_proj_ffts(projs):
    return fft.rfft(projs, axis=1)

def ramp_filter_ffts(ffts):
    ramp = np.floor(np.arange(0.5, ffts.shape[1]//2 + 0.1, 0.5))
    result = ffts * ramp
    print("Length of result:")
    print(len(result))
    return result

def iffts(proj):
    return fft.irfft(proj, axis=1)

def build_laminogram(radonT):
    "Generate a laminogram by simple backprojection using the Radon Transform of an image, 'radonT'."
    laminogram = np.zeros((radonT.shape[1],radonT.shape[1]))
    dTheta = 180.0 / radonT.shape[0]
    for i in range(radonT.shape[0]):
        temp = np.tile(radonT[i],(radonT.shape[1],1))
        temp = rotate(temp, dTheta*i)
        laminogram += temp
    return laminogram

def hamming_window(ramp):
    hamming = np.hamming(len(ramp))
    hammingResult = ramp * hamming
    return hammingResult

fourier = build_proj_ffts(sinogram)            #Get the Fast Fourier Transform of the image (Frequency Domain)

ramp_filtered = ramp_filter_ffts(fourier)      #Filter the fourier transform by the ramp filter

inverse = iffts(ramp_filtered)                 #Take the inverse FFT to convert back to Spatial Domain

reconstructed = build_laminogram(inverse)      #Build the filtered image by backprojecting the filtered projections

hammingFilter = hamming_window(reconstructed)

imutils.imshow(hammingFilter, title="Backprojection with Hamming-windowed Ramp Filtering")