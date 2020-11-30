import skimage.io as io

# Show the figures / plots inside the notebook
%matplotlib inline
from skimage.color import rgb2gray,rgb2hsv
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import random_noise
import numpy as np

from skimage.exposure import histogram
from matplotlib.pyplot import bar
from scipy import misc
import imageio
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()
    # Ex: imread and imshow 
pyramids = io.imread('pyramids.jpeg')
io.imshow(pyramids)
io.show()

# Use this function to show a histogram
# The image should be gray-scale and should range from 0 to 1
def showHist(img,histogramImg):
    plt.figure()
    bar(histogramImg[1]*255, histogramImg[0], width=0.8, align='center')
    '''
@TODO:
Requirement #1 
-Read and print image 'coffee'  
-Show and print half of the  image
-----hint :use the attribute shape of numpy to get the image object dimentions
'''
coffee = io.imread('coffee.jpeg')
io.imshow(coffee)
io.show()
width=coffee.shape[1]
half_width = width //2 
new_coffee_image = coffee[:, :half_width]
io.imshow(new_coffee_image)
io.show()
'''
@TODO:
Requirement #2 
RGB to gray and HSV:
1- Write a function 'gray_image' that takes an image as input , then
    -get the gray scale of the image, then
    -display original image and the gray scale one side by side (subplot).
    -hint: use rgb2gray to get the graylevel of the image
- test your function with the image 'pyramids'

2- Write a function 'HSV_image' that takes an image as input , then
   -show the RGB image and the (3 channels of HSV image each channel in separated form ) one side by side (subplot).
   -hint: -use rgb2hsv (to get the hsv representation of the image).
          -To separately get the Hue, Saturation and Value channels, use hsvImg[:,:,X], 
           where hsvImg is the hsv representation of the image. 
           Hue is the first channel, Saturation is the second and value is the last channel.
- test your function for the images in HSV Folder. And comment on the results. 
'''
def gray_image(image):
    grayImg = rgb2gray(image)
    arr= [image,grayImg]
    title= ["RGB", "gray Image"]
    show_images( arr,title)
    
pyramids = io.imread('pyramids.jpeg')
gray_image(pyramids)

def HSV_image(image):
    hsvImg = rgb2hsv(image)
    arr=[image,hsvImg[:,:,0],hsvImg[:,:,1] ,hsvImg[:,:,2]]
    title = ["RGB", "Hue", "Saturation", "Value"]
    show_images( arr,title)

Image1 = io.imread('./hsv/ex1.png')
Image2 = io.imread('./hsv/ex2.jpg')
Image3 = io.imread('./hsv/ex3.jpg')

HSV_image(Image1)   
HSV_image(Image2)   
HSV_image(Image3)   
    
    '''
Requirement 2 (Noise):
http://scikit-image.org/docs/dev/api/skimage.util.html#skimage.util.random_noise

1.For an image of your choice (the effect of noise must be obvious): 
    Read the image.
    Convert it to greyscale.
    Apply salt & pepper noise with 
    amount=0.05, 0.5 and 0.9
   
2. From the other images. Recommend one image that won’t be greatly affected by the noise and state why.    
    '''
image = io.imread('coffee.jpeg')
grayImg = rgb2gray(image)
io.imshow(random_noise(grayImg, mode='s&p', amount=0.05))
io.show()
io.imshow(random_noise(grayImg, mode='s&p', amount=0.5))
io.show()
io.imshow(random_noise(grayImg, mode='s&p', amount=0.9))
io.show()
Image1 = io.imread('./hsv/ex1.png')
Image2 = io.imread('./hsv/ex2.jpg')
Image3 = io.imread('./hsv/ex3.jpg')
io.imshow(random_noise(rgb2gray(Image1), mode='s&p', amount=0.1))
io.show()
io.imshow(random_noise(rgb2gray(Image2), mode='s&p', amount=0.1))
io.show()
io.imshow(random_noise(rgb2gray(Image3), mode='s&p', amount=0.1))
io.show()
'''
Requirement 3 (Histogram):

1- For the given images ( in histogram folder): 
    Read the image.
    Apply histogram and show it.
Hint
    A) Use histogram (image) to get histogram. Try different values for nbins (256,64,8), What does it mean?
    B) and function(showHist) to draw it.
** 2- Draw a grey-scale image that has uniform histogram 
same number of pixels for all intensity levels) using code only. Let the size of the image be 256x256.
use np.ones to draw image with ones.
'''
image = io.imread('pyramids.jpeg')
gImg=rgb2gray(Image1)
histImg= histogram(gImg,256)
showHist(image,histImg)
histImg= histogram(gImg,64)
showHist(image,histImg)
histImg= histogram(gImg,8)
showHist(image,histImg)