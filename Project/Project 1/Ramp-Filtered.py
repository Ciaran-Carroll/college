'''
RE4017 - Machine Vision

Kevin Burke (14155893)
Paul Lynch (16123778)
Ciaran Carroll (13113259)
Qicong Zhang (16069978)

Reconstruction of an image with ramp filter applied
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
    result =  fft.rfft(projs, axis=1)
    plt.plot(abs(np.real(result)))
    plt.grid()
    plt.show()
    return result

def ramp_filter_ffts(ffts):
    ramp = np.floor(np.arange(0.5, ffts.shape[1]//2 + 0.1, 0.5))
    plt.plot(ramp)
    plt.title('Ramp Filter')
    plt.grid()
    plt.show()
    return ffts * ramp

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

fourier = build_proj_ffts(sinogram)            #Get the Fast Fourier Transform of the image (Frequency Domain)
ramp_filtered = ramp_filter_ffts(fourier)      #Filter the fourier transform by the ramp filter
inverse = iffts(ramp_filtered)                 #Take the inverse FFT to convert back to Spatial Domain
reconstructed = build_laminogram(inverse)      #Build the filtered image by backprojecting the filtered projections

imutils.imshow(reconstructed, title="Backprojection with Ramp Filtering")