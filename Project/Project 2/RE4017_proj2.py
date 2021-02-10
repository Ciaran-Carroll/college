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


def compute_harris_response(image, sigma = 5):
    ''' Compute the Harris corner detector algorithm for each pixel
    in a gray level image. '''

    # Derivatives
    imagex = np.zeros(image.shape)
    filters.gaussian_filter(image, (sigma, sigma), (0, 1), imagex)
    imagey = np.zeros(image.shape)
    filters.gaussian_filter(image, (sigma, sigma), (1, 0), imagey)

    # Compute components of the Harris matrix
    A = filters.gaussian_filter(imagex*imagex, sigma)
    B = filters.gaussian_filter(imagex*imagey, sigma)
    C = filters.gaussian_filter(imagey*imagey, sigma)

    # Determinant and trace
    Det_M = A*C - B**2
    Tr_M = A + C

    return Det_M / Tr_M

def get_harris_points(harris_im, min_d = 10, threshold = 0.1):
    ''' Return corners from a Harris response image min.dist is the minimum
     number of pixels seperating corners and image boundary. '''

    # Finds top corner canadates above a threshold
    corner_threshold = harris_im.max() * threshold
    harris_im_th = (harris_im > corner_threshold)

    # Find the co-ordinates of these candidates, and their response values
    coords = np.array(harris_im_th.nonzero()).T
    candidate_values = np.array([harris_im[c[0],c[1]] for c in coords])

    # Find the indices into the ‘candidate_values’ array that sort it in order
    # of increasing response strength.
    indices = np.argsort(candidate_values)

    # Store allowed point locatons in a Boolean image
    allowed_locations = np.zeros(harris_im.shape, dtype = 'bool')
    allowed_locations[min_d:-min_d, min_d:-min_d] = True

    # Select the best points taking the min_distance into account
    filtered_coords = []
    for i in indices[::-1]:
        r,c = coords[i]
        if allowed_locations[r, c]:
            filtered_coords.append((r,c))
            allowed_locations[r - min_d:r + min_d + 1, c - min_d:c + min_d + 1] = False

    return filtered_coords

def plot_harris_points(image, filtered_coords):
    ''' Plots corners found in image '''

    plt.figure()
    plt.gray()
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords], '.')
    plt.axis('off')
    plt.title('Plot Harris Corner')
    plt.show()

def get_descriptors(image, filtered_coords, wid = 5):
    ''' For each point return pixel values around the point using a neighbourhood
    of width 2_wid+1. '''

    desc =[]
    for coords in filtered_coords:
        patch = image[coords[0] - wid:coords[0] + wid + 1, coords[1] - wid:coords[0] + wid + 1].flatten()
        desc.append(patch)

    return desc

def match(desc1, desc2, threshold = 0.5):
    ''' For each of the corner descriptor in the first image, select its match
    to the second image using normaliazed cross correlation. '''

    n = len(desc1[0])

    # Pair-wise distances
    d = -np.ones((len(desc1),len(desc2)))

    # Calculating the normaliazed cross-correlation (ncc)
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
            print('d1 = ')
            print(d1)
            print(len(d1))
            d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
            print('d2 = ')
            print(d2)
            #print(len(d2)
            ncc_value = np.sum(d1 * d2) / (n - 1)
            if ncc_value > threshold:
                d[i,j] = ncc_value

    ndx = np.argsort(-d)
    print(ndx)
    matchscores = ndx[:,0]

    return matchscores

def match_twosided(desc1, desc2, threshold = 0.5):
    ''' Two-sided symmetric version of match(). '''
    matches_12 = match(desc1, desc2, threshold)
    matches_21 = match(desc2, desc1, threshold)

    ndx_12 = where(matches_12 >= 0)[0]

    # Remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1

    return matches_12

def appendimages(image1, image2):
    ''' Return a new image that appends the two images side-by-side. '''

    # Select the image with the fewest rows and fill in enough empty rowa
    rows1 = image1.shape[0]
    rows2 = image2.shape[0]

    if rows1 < rows2:
        image1 = concatenate((image1, np.zeros((rows2-rows1, image1.shape[1]))), axis = 0)
    elif rows1 > rows2:
        image2 = concatenate((image2, np.zeros((rows1-rows2, image1.shape[1]))), axis = 0)
    # If none of these cases are true, no filling needed

    return concatenate((image1,image2), axis = 1)

def plot_matches(image1, image2, locations1, locations2, matchscores, show_below = True):
    ''' Show a figure with lines joining the accepted matches
     imput: image1, image2 (images as arrays), locations1,locations2 (feature locations),
     matchscores (as output from 'match()'), show below (if images should be shown below
     matches) '''

    image3 = appendimages(image1, image2)
    if show_below:
        image3 = vstack(image1, image2)

    imshow(image3)

    cols1 = image.shape[1]
    for i,m in enumerate(matchscores):
        if m>0:
            plot([locations1[i][1],locations2[m][1]+cols1],[locations1[i][0],locations2[m][0]], 'c')
        plt.axis('off')


image = np.array(Image.open('arch1.png').convert('L'))

harris_image = compute_harris_response(image)
filtered_coords = get_harris_points(harris_image, 6)
print('No. of Harris Interest points in image: ', len(filtered_coords))
plot_harris_points(image, filtered_coords)
plt.show()

# imshow(filtered_coords)

image1 =  np.array(Image.open('arch1.png').convert('L'))
print(image1.size)
image2 =  np.array(Image.open('arch2.png').convert('L'))
print(image2.size)

wid = 5
harris_image1 = compute_harris_response(image1)
filtered_coords1 = get_harris_points(harris_image1, wid+1)
d1 = get_descriptors(image1, filtered_coords1, wid)

harris_image2 = compute_harris_response(image2)
filtered_coords2 = get_harris_points(harris_image2, wid+1)
d2 = get_descriptors(image2, filtered_coords2, wid)

print('Start matching')
matches = match_twosided(d1, d2)
plt.figure()
plt.gray()
plot_matches(image1, image2, filtered_coords1, filtered_coords2, matches)
plt.show()
