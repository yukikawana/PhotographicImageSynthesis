import numpy as np
import re
import tensorflow as tf
import matplotlib.image as mpimg
import sys, os
import cv2
import dill

sys.path.insert(0,"scd")
from scd import get_tenors, process_image, imread_as_jpg 

def main():
    #CUDA_VISIBLE_DEVICES=""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    isess = tf.InteractiveSession()
    ckpt_filename = './ssd_300_kitti/ssd_model.ckpt'
    tensors = get_tenors(ckpt_filename,isess)
    predictions, localisations, logits, end_points, img_input, ssd  = tensors

    rs =r".*\/conv[0-9]\/conv[0-9]_[0-9]/Relu$"
    rc = re.compile(rs)
    new_end_points = {}
    for op in tf.get_default_graph().as_graph_def().node:
        gr = rc.match(op.name)
        if gr:
            print(op.name)
            new_end_points[op.name.split("/")[-2]] = tf.get_default_graph().get_tensor_by_name(op.name+":0")
    """
    for n in new_end_points:
        print(n,new_end_points[n])
    """
    path =  '../kitti/voc_format/VOC2012/JPEGImages/'
    outpath = 'output_images/'
    image_names = sorted(os.listdir(path))
    dimpkl = {}
    for name in image_names:
        img = imread_as_jpg(path + name)
        img = cv2.resize(img, (993//2, 300))
        print(img.shape, name)
        #img = process_image(img, tensors,isess, select_threshold=0.8, nms_threshold=0.5)
        img = process_image(img, tensors,isess, select_threshold=0.8, nms_threshold=0.5)
        #mpimg.imsave(outpath + name, img, format='jpg')
        for n in new_end_points:
            val = isess.run([new_end_points[n]], feed_dict={img_input: img})[0]
            print(n, val.shape[1:3])
            dimpkl[n] = val.shape[1:3]

        dill.dump(dimpkl, open("dim_300.pkl", "wb"))
        assert(False)
        mpimg.imsave(outpath + name, img, format='jpg')
if __name__ == "__main__":
    main()
