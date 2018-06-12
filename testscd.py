import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import sys, os
import cv2

sys.path.insert(0,"scd")
from scd import get_tenors, process_image, imread_as_jpg 

def main():
    isess = tf.InteractiveSession()
    ckpt_filename = './ssd_300_kitti/model.ckpt-89992'
    ckpt_filename = './ssd_300_kitti/ssd_model.ckpt'
    tensors = get_tenors(ckpt_filename,isess)
# Load a sample image.
    path = 'test_images/'
    path =  '../kitti/training/image_2/'
    path =  '../kitti/voc_format/VOC2012/JPEGImages/'
#path = "../PhotographicImageSynthesis/result_512p/final/"
    outpath = 'output_images/'
    image_names = sorted(os.listdir(path))
    for name in image_names:
        img = imread_as_jpg(path + name)
        img = cv2.resize(img, (463, 150))
        """
        img = cv2.copyMakeBorder(img, left=0, right=0, top=75, bottom=75, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
        """
        #img = cv2.resize(img, (500, 300))
        """
        img = mpimg.imread("320.jpg")
        img = cv2.resize(img, dsize=(0,0), fx=720./img.shape[0], fy=720./img.shape[0])
        img = cv2.resize(img, (img.shape[1], 720))
        left =int((img.shape[1]  -1280)/2.)
        img = img[:,left:left+1280,:]
        print(img.shape)
        """
        img = process_image(img, tensors,isess, select_threshold=0.8, nms_threshold=0.5)
        mpimg.imsave(outpath + name, img, format='jpg')
if __name__ == "__main__":
    main()
