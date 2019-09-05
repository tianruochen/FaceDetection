# -*- coding: utf-8 -*-
import caffe
#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
#import Image
import sys
import os
from  math import pow
from PIL import Image, ImageDraw, ImageFont
import cv2
import math
import random
caffe_root = ‘/home/kobe/caffe/‘

sys.path.insert(0, caffe_root + ‘python‘)
os.environ[‘GLOG_minloglevel‘] = ‘2‘

caffe.set_mode_cpu()
#####################################################################################################
# Load the original network and extract the fully connected layers‘ parameters.
net = caffe.Net(‘/home/kobe/Desktop/faceDetect/project/deploy.prototxt‘, 
                ‘/home/kobe/Desktop/faceDetect/model/_iter_50000.caffemodel‘, 
                caffe.TEST)
params = [‘fc6‘, ‘fc7‘, ‘fc8_flickr‘]
# fc_params = {name: (weights, biases)}
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

for fc in params:
    print ‘{} weights are {} dimensional and biases are {} dimensional‘.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)
#######################################################################################################
# Load the fully convolutional network to transplant the parameters.
net_full_conv = caffe.Net(‘/home/kobe/Desktop/faceDetect/project/deploy_full_conv.prototxt‘, 
                          ‘/home/kobe/Desktop/faceDetect/model/_iter_50000.caffemodel‘,
                          caffe.TEST)
params_full_conv = [‘fc6-conv‘, ‘fc7-conv‘, ‘fc8-conv‘]
# conv_params = {name: (weights, biases)}
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

for conv in params_full_conv:
    print ‘{} weights are {} dimensional and biases are {} dimensional‘.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)
#############################################################################################################
for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
    conv_params[pr_conv][1][...] = fc_params[pr][1]
##############################################################################################################
net_full_conv.save(‘/home/kobe/Desktop/faceDetect/alexnet_iter_50000_full_conv.caffemodel‘)