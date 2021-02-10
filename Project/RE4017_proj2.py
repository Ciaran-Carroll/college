'''

Student Name:   Ciaran Carroll
Student Id:     13113259

Project 2:
Research and Implement Harris Corner Detection using Python/Numpy Investigating
the behaviour of the algorithm.

Overall Project Steps:
(1) - Find Harris interest points (Hips) by thresholding Harris response images
      for image 1 and Image 2.
(2) - Form normalised patch descriptor vectors for all the Hips in image 1 and
      all the Hips in image 2.
(3) - Match these using inner product op & threshold for strong matches. Sort
      by match strength (strongest first). Result is a list of points
      correspondences.
(4) - 'Exhaustive RANSAC' to filter outliers from these and return best
      translation between the image (dr, dc).
(5) - Use the translation to make a composite image & return this.

#Steps:
#(1) - Find edges in image I(x,y) by convolving with derivative f Guassian x & y
#      kernels (sigma = 1) to give I_x(x,y) & I_y(x,y)

'''

import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage import filters
import matplotlib.pylab as plt
from PIL import Image
import scipy
from scipy import signal
#from pylab import *
# FInd Harris interest points (Hips) by thresholding Harris response images for
# image 1 and image 2


def imread(filename,greyscale=True):
    """Load an image, return as a Numpy array."""
    if greyscale:
        pil_im = Image.open(filename).convert('L')
    else:
        pil_im = Image.open(filename)
    return np.array(pil_im)


def imshow(im, autoscale=False,colourmap='gray', newfig=True, title=None):
    """Display an image, turning off autoscaling (unless explicitly required)
       and interpolation.

       (1) 8-bit greyscale images and 24-bit RGB are scaled in 0..255.
       (2) 0-1 binary images are scaled in 0..1.
       (3) Float images are scaled in 0.0..1.0 if their min values are >= 0
           and their max values <= 1.0
       (4) Float images are scaled in 0.0..255.0 if their min values are >= 0
           and their max values are > 1 and <= 255.0
       (5) Any image not covered by the above cases is autoscaled.  If
           autoscaling is explicitly requested, it is always turned on.

       A new figure is created by default.  "newfig=False" turns off this
       behaviour.

       Interpolation is always off (unless the backend stops this).
    """
    if newfig:
        if title != None: fig = plt.figure(title)
        else: fig = plt.figure()
    if autoscale:
        plt.imshow(im,interpolation='nearest',cmap=colourmap)
    else:
        maxval = im.max()
        if im.dtype == 'uint8':        ## 8-bit greyscale or 24-bit RGB
            if maxval > 1: maxval = 255
            plt.imshow(im,interpolation='nearest',vmin=0,vmax=maxval,cmap=colourmap)
        elif im.dtype == 'float32' or im.dtype == 'float64':
            minval = im.min()
            if minval >= 0.0:
                if maxval <= 1.0:  ## Looks like 0..1 float greyscale
                    minval, maxval = 0.0, 1.0
                elif maxval <= 255.0: ## Looks like a float 0 .. 255 image.
                    minval, maxval = 0.0, 255.0
            plt.imshow(im,interpolation='nearest',vmin=minval,vmax=maxval,cmap=colourmap)
        else:
            plt.imshow(im,interpolation='nearest',cmap=colourmap)
    plt.axis('image')
    ## plt.axis('off')
    plt.show()
    ##return fig


def compute_harris_response(image, sigma = 2):
    ''' Compute the Harris corner detector response function for each pixel
    in a gray level image. '''

    # Derivatives
    imagex = np.zeros(image.shape)
    filters.gaussian_filter(image, (sigma, sigma), (0, 1), imagex)
    imagey = np.zeros(image.shape)
    filters.gaussian_filter(image, (sigma, sigma), (1, 1), imagey)

    # Compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imagex*imagex, sigma)
    Wxy = filters.gaussian_filter(imagex*imagey, sigma)
    Wyy = filters.gaussian_filter(imagey*imagey, sigma)

    # Determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy

    return Wdet / Wtr

def get_harris_points(harris_im, min_d = 10, threshold = 0.1):
    ''' Return corners from a Harris response image min.dist is the minimum
     number of pixels seperating corners and image boundary. '''

     # Finds top corner canadates above a threshold
    corner_threshold = harris_im.max() * threshold
    harris_im_th = (harris_im > corner_threshold) * 1

    # Find the co-ordinates of these candidates, and their response values
    coords = np.array(harris_im_th.nonzero()).T
    candidate_values = np.array([harris_im[c[0],c[1]] for c in coords])

    # Find the indices into the ‘candidate_values’ array that sort it in order
    # of increasing response strength.
    index = np.argsort(candidate_values)

    # Store allowed point locatons in a Boolean image
    allowed_locations = np.zeros(harris_im.shape)
    allowed_locations[min_d:-min_d, min_d:-min_d] = 1

    # Select the best points taking the min_distance into account
    filtered_coords = []
    for i in index:
        #r,c = coords[i]
        if allowed_locations[coords[i,0], coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0] - min_d):(coords[i,0] + min_d), (coords[i,1] - min_d):(coords[i,1] + min_d)] = 0

    return filtered_coords

def plot_harris_points(image, filtered_coords):
    ''' Plots corners found in image '''

    plt.figure()
    plt.gray()
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords], '*')
    plt.axis('off')
    plt.show()

image = np.array(Image.open('arch1.png').convert('L'))
#image2 = np.zeros(image.shape)

sigma = 1
image2 = filters.gaussian_filter(image, sigma*2)
harris_image = compute_harris_response(image2)
filtered_coords = get_harris_points(harris_image, 10, 0.1)
print(len(filtered_coords))
plot_harris_points(image2, filtered_coords)
plt.show()
