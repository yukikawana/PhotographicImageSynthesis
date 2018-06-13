import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import sys, os
import cv2

sys.path.insert(0,"scd")
from scd import get_tenors, process_image, imread_as_jpg 

def main():
    ckpt_filename = './ssd_300_kitti/ssd_model.ckpt'
    isess = tf.InteractiveSession()
    tensors = get_tenors(ckpt_filename,isess)
    saver = tf.train.Saver()
    saver.restore(isess, ckpt_filename)

# Load a sample image.
    path = 'test_images/'
    path =  '../kitti/voc_format/VOC2012/JPEGImages/'
    path =  '../kitti/testing/image_2/'
#path = "../PhotographicImageSynthesis/result_512p/final/"
    outpath = 'output_images/'
    image_names = sorted(os.listdir(path))
    for idx, name in enumerate(image_names):
        idx+=7481
        print("%06d.png"%idx)
        img = imread_as_jpg(path + "%06d.png"%idx)
        img = cv2.resize(img, (463, 150))
    
        img = process_image(img, tensors,isess, select_threshold=0.8, nms_threshold=0.5)
        #mpimg.imsave(outpath + name, img, format='jpg')
if __name__ == "__main__":
    main()
