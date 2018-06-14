from __future__ import division
import math
import re
import os,helper,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import cv2
import os, sys
import dill
sys.path.insert(0,"scd")
from scd import get_tenors, process_image, imread_as_jpg 
NUM_TRAINING_IMAGES = 7480
#NUM_TRAINING_IMAGES = 30
ckpt_filename = './ssd_300_kitti/ssd_model.ckpt'
restore_epoch=9

def lrelu(x):
    return tf.maximum(0.2*x,x)

def build_scd(input,sess,reuse=None):
    #with tf.variable_scope("scd300"):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    #end_points= get_tenors(ckpt_filename,sess, input=input, reuse=reuse)
    end_points= get_tenors(ckpt_filename,sess, input=input)
    rs =r"(.*\/conv[0-9]\/conv[0-9]_[0-9]/Relu$|.*ssd_300_vgg\/pool5\/MaxPool$)"
    rc = re.compile(rs)
    new_end_points = {}
    for op in tf.get_default_graph().as_graph_def().node:
        gr = rc.match(op.name)
        if gr:
            new_end_points[op.name.split("/")[-2]] = tf.get_default_graph().get_tensor_by_name(op.name+":0")
    new_end_points["input"] = end_points["input"]
    return new_end_points

def recursive_generator(label,sp, reuse=None):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    dim=512 if sp>=75 else 1024
    if sp==5:
        input=label
    else:
        spss2 = math.ceil(sp/2.)
        downsampled=tf.image.resize_area(label,(spss2,scaledic[spss2]),align_corners=False)
        input=tf.concat([tf.image.resize_bilinear(recursive_generator(downsampled,spss2, reuse=reuse),(sp,scaledic[sp]),align_corners=True),label],3)
    net=slim.conv2d(input,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
    if sp==150:
        net=slim.conv2d(net,3,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv100')
        net=(net+1.0)/2.0*255.0

        """
        net=slim.conv2d(net,27,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv100')
        net=(net+1.0)/2.0*255.0
        split0,split1,split2=tf.split(tf.transpose(net,perm=[3,1,2,0]),num_or_size_splits=3,axis=0)
        net=tf.concat([split0,split1,split2],3)
        """

    return net

def compute_error(real,fake,label):
    #return tf.reduce_mean(label*tf.expand_dims(tf.reduce_mean(tf.abs(fake-real),reduction_indices=[3]),-1),reduction_indices=[1,2])#diversity loss
    return tf.expand_dims(tf.reduce_mean(tf.abs(fake-real)), -1)#simple loss

dimdic = dill.load(open("dim_150.pkl", "rb"))
scaledic = dill.load(open("scaledic.pkl", "rb"))
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess=tf.Session(config=config)
is_training=True
sp=150#spatial resolution: 256x512
with tf.variable_scope(tf.get_variable_scope()):
    #label=tf.placeholder(tf.float32,[None,None,None,513])
    #label2 =tf.image.resize_bilinear(label,(150,496),align_corners=False)
    real_image=tf.placeholder(tf.float32,[None,None,None,3])
    vgg_real=build_scd(real_image,sess)

    pool5 = vgg_real["pool5"]
    label2_small = tf.concat((pool5,tf.expand_dims(1-tf.reduce_sum(pool5,axis=3),axis=3)),3)
    label2 = tf.image.resize_bilinear(label2_small,(150,496),align_corners=True)

    generator=recursive_generator(label2,sp)
    fake_image=tf.placeholder(tf.float32,[None,None,None,3])
    vgg_fake=build_scd(generator,sess, reuse=True)

    train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.contrib.framework.get_name_scope())
    ssd_var_list=[v for v in train_vars if v.name.split("/")[0] == "ssd_300_vgg"]
    saver = tf.train.Saver(var_list=ssd_var_list)
    saver.restore(sess, ckpt_filename)
    p0=compute_error(vgg_real['input'],vgg_fake['input'],label2)
    p1=compute_error(vgg_real['conv1_2'],vgg_fake['conv1_2'],label2)/1.6
    p2=compute_error(vgg_real['conv2_2'],vgg_fake['conv2_2'],tf.image.resize_area(label2,dimdic['conv2_2']))/2.3
    p3=compute_error(vgg_real['conv3_2'],vgg_fake['conv3_2'],tf.image.resize_area(label2,dimdic['conv3_2']))/1.8
    p4=compute_error(vgg_real['conv4_2'],vgg_fake['conv4_2'],tf.image.resize_area(label2,dimdic['conv4_2']))/2.8
    p5=compute_error(vgg_real['conv5_2'],vgg_fake['conv5_2'],tf.image.resize_area(label2,dimdic['conv5_2']))*10/0.8#weights lambda are collected at 100th epoch
    content_loss=p0+p1+p2+p3+p4+p5
    G_loss=tf.reduce_sum(tf.reduce_min(content_loss,reduction_indices=0))*0.999+tf.reduce_sum(tf.reduce_mean(content_loss,reduction_indices=0))*0.001

lr=tf.placeholder(tf.float32)
G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss,var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])
all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
init_vars = list(set(all_vars) - set(ssd_var_list))
saver=tf.train.Saver(max_to_keep=1000, var_list=init_vars)
sess.run(tf.initialize_variables(init_vars))


if is_training:
    g_loss=np.zeros(NUM_TRAINING_IMAGES,dtype=float)
    input_images=[None]*(NUM_TRAINING_IMAGES+100)
    #label_images=[None]*NUM_TRAINING_IMAGES
    for epoch in range(1,101):
        if os.path.isdir("result_kitti256p_2/%04d"%epoch):
            continue
        saver.restore(sess, "result_kitti256p_2/%04d/model.ckpt"%restore_epoch)
        cnt=0
        for ind in np.random.permutation(NUM_TRAINING_IMAGES - 25)+1:
            st=time.time()
            cnt+=1
            if input_images[ind] is None:
                #label_images[ind]=np.load("hmpool5/%06d.npz"%ind)["arr_0"]#training label
               
                path =  '../kitti/training/image_2/%06d.png'%ind
                img = imread_as_jpg(path)
                img = cv2.resize(img, (496,150)) 
                input_images[ind]=np.expand_dims(np.float32(img),axis=0)#training image
            #_,G_current,l0,l1,l2,l3,l4,l5=sess.run([G_opt,G_loss,p0,p1,p2,p3,p4,p5],feed_dict={label:np.concatenate((label_images[ind],np.expand_dims(1-np.sum(label_images[ind],axis=3),axis=3)),axis=3),real_image:input_images[ind],lr:min(1e-6*np.power(1.1,epoch-1),1e-4 if epoch>100 else 1e-3)})
            _,G_current,l0,l1,l2,l3,l4,l5=sess.run([G_opt,G_loss,p0,p1,p2,p3,p4,p5],feed_dict={real_image:input_images[ind],lr:1e-4})#may try lr:min(1e-6*np.power(1.1,epoch-1),1e-4 if epoch>100 else 1e-3) in case lr:1e-4 is not good
            g_loss[ind]=G_current
            print("%d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f"%(epoch,cnt,np.mean(g_loss[np.where(g_loss)]),np.mean(l0),np.mean(l1),np.mean(l2),np.mean(l3),np.mean(l4),np.mean(l5),time.time()-st))
        os.makedirs("result_kitti256p_2/%04d"%epoch)
        target=open("result_kitti256p_2/%04d/score.txt"%epoch,'w')
        target.write("%f"%np.mean(g_loss[np.where(g_loss)]))
        target.close()
        saver.save(sess,"result_kitti256p_2/model.ckpt")

        if epoch%3==0:
            saver.save(sess,"result_kitti256p_2/%04d/model.ckpt"%epoch)

        for ind in range(NUM_TRAINING_IMAGES+1,NUM_TRAINING_IMAGES+51):
            path =  '../kitti/testing/image_2/%06d.png'%ind
            #path =  '../kitti/training/image_2/%06d.png'%ind
            #semantic = np.load("hmpool5/%06d.npz"%ind)["arr_0"]#training label
            if input_images[ind] is None:
                img = imread_as_jpg(path)
                img = cv2.resize(img, (496,150)) 
                input_images[ind]=np.expand_dims(np.float32(img),axis=0)#training image
            output=sess.run(generator,feed_dict={real_image:input_images[ind]})
            #output=sess.run(generator,feed_dict={real_image:input_images[ind]})
            output=np.minimum(np.maximum(output,0.0),255.0)
            """
            upper=np.concatenate((output[0,:,:,:],output[1,:,:,:],output[2,:,:,:]),axis=1)
            middle=np.concatenate((output[3,:,:,:],output[4,:,:,:],output[5,:,:,:]),axis=1)
            bottom=np.concatenate((output[6,:,:,:],output[7,:,:,:],output[8,:,:,:]),axis=1)
            scipy.misc.toimage(np.concatenate((upper,middle,bottom),axis=0),cmin=0,cmax=255).save("result_kitti256p_2/%04d/%06d_output.jpg"%(epoch,ind))
            """
            scipy.misc.toimage(output[0,:,:,:],cmin=0,cmax=255).save("result_kitti256p_2/%04d/%06d_output.jpg"%(epoch,ind))
else:
    input_images=[None]*(NUM_TRAINING_IMAGES+100)
    saver.restore(sess, "result_kitti256p_2/%04d/model.ckpt"%restore_epoch)
    if not os.path.isdir("result_kitti256p_2/eval"):
        os.makedirs("result_kitti256p_2/eval")
    for ind in range(NUM_TRAINING_IMAGES+1,NUM_TRAINING_IMAGES+51):
        path =  '../kitti/testing/image_2/%06d.png'%ind
        img = imread_as_jpg(path)
        img = cv2.resize(img, (496,150)) 
        input_images[ind]=np.expand_dims(np.float32(img),axis=0)#training image
        #semantic = np.load("hmpool5/%06d.npz"%ind)["arr_0"]#training label
        #output=sess.run(generator,feed_dict={label2_small:np.concatenate((semantic,np.expand_dims(1-np.sum(semantic,axis=3),axis=3)),axis=3)})
        output=sess.run(generator,feed_dict={real_image:input_images[ind]})
        output=np.minimum(np.maximum(output,0.0),255.0)
        output=np.minimum(np.maximum(output,0.0),255.0)
        scipy.misc.toimage(output[0,:,:,:],cmin=0,cmax=255).save("result_kitti256p_2/eval/%06d_output.jpg"%(ind))
