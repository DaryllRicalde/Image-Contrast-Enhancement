import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure as ex
import imageio
import sys
import cv2 as cv

def he(img):
    if(len(img.shape)==2):      #gray
        outImg = ex.equalize_hist(img[:,:])*255 
    elif(len(img.shape)==3):    #RGB
        outImg = np.zeros((img.shape[0],img.shape[1],3))
        for channel in range(img.shape[2]):
            outImg[:, :, channel] = ex.equalize_hist(img[:, :, channel])*255

    outImg[outImg>255] = 255
    outImg[outImg<0] = 0
    return outImg.astype(np.uint8)

def main():
    img_name = sys.argv[1] #img as arguments
    img2 = sys.argv[2]
    img3 = sys.argv[3]

    img = imageio.imread(img_name)
    read_img2 = imageio.imread(img2)
    read_img3 = imageio.imread(img3)

    result1 = he(img)
    result2 = he(read_img2)
    result3 = he(read_img3)

    plt.imshow(result1)
    plt.imshow(result2)
    plt.imshow(result3)

    plt.show()
    cv.imwrite('enhanced1.png',result1)
    cv.imwrite('enhanced2.png',result2)
    cv.imwrite('enhanced3.png',result3)

if __name__ == '__main__':
    main()