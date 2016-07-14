import os
import cv2
import sys
import json
import time
import math
import random
import pickle
import subprocess
import numpy as np
import pandas as pd
from pprint import pprint
from glob import glob
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Add Faster R-CNN module to pythonpath
from params import get_params
params = get_params()
sys.path.insert(0, os.path.join(params['fast_rcnn_path'],'caffe-fast-rcnn', 'python'))
sys.path.insert(0, os.path.join(params['fast_rcnn_path'],'lib'))

import caffe
from fast_rcnn.config import cfg
import test as test_ops

# Init network
cfg.TEST.HAS_RPN = True
if params['gpu']:
    caffe.set_mode_gpu()
    caffe.set_device(0)
else:
    caffe.set_mode_cpu()

class Featurizer():
    
    def __init__(self,params):
        self.layer_dense = params['layer_dense']
        # self.pooling     = params['pooling']  
        self.pooling = 'sum'      
        self.net         = caffe.Net(params['net_proto'], params['net'], caffe.TEST)
        
    def pool_feats(self, feat):        
        pool_func = np.max if self.pooling == 'max' else np.sum
        return pool_func(pool_func(feat, axis=1), axis=1)
    
    def local_features(self, im, bbx=None):
        _          = test_ops.im_detect(self.net, im, boxes=None)
        feat       = self.net.blobs[self.layer_dense].data.squeeze()
        bbx        = bbx * np.array([feat.shape[2], feat.shape[1], feat.shape[2], feat.shape[1]])
        local_feat = feat[:,int(bbx[1]):int(bbx[3]),int(bbx[0]):int(bbx[2])]
        return self.pool_feats(local_feat)
        
    def global_features(self, im):
        _ = test_ops.im_detect(self.net, im, boxes=None)
        return self.net.blobs['fc7'].data.squeeze()

# --

dbpath  = './data/features_wbox/yemen/*'
# labpath = '/home/ubuntu/queries/flags/labels.json'

ft   = Featurizer(params)

db = [pickle.load(open(f)) for f in fs]
db = filter(None, db)
# labs = filter(lambda x: len(x['annotations']), json.load(open(labpath)))

# def kth_record(k):
#     qpath = '/home/ubuntu/queries/tanks/%s' % labs[k]['filename']
#     im    = cv2.imread(qpath)
    
#     bbx           = labs[k]['annotations'][0].copy()
#     bbx['x']      = max(0, bbx['x'] / im.shape[1])
#     bbx['y']      = max(0, bbx['y'] / im.shape[0])
#     bbx['width']  = min(1, bbx['width'] / im.shape[1])
#     bbx['height'] = min(1, bbx['height'] / im.shape[0])
    
#     bbx = np.array([bbx['x'], bbx['y'], bbx['x'] + bbx['width'], bbx['y'] + bbx['height']])
    
#     im = cv2.resize(im, (150, 150))
#     return im, bbx


# allfeats = np.vstack([d['feats']['pooled_roi'] for d in db])
# allfeats = normalize(allfeats)

# alllabs = reduce(lambda a,b: a+b, [[d['fname'] for _ in range(len(d['feats']['pooled_roi']))]for d in db])

# q = ft.local_features(*kth_record(0))
# q = normalize(q).squeeze()

# allsims = q.dot(allfeats.T)
# allsims.mean()
# allsims = sorted(zip(alllabs, allsims), key = lambda x: -x[1])
# pprint(allsims[0:5])

# paths = set([allsims[i][0] for i in range(5)])
# for p in paths:
#     subprocess.Popen(['rsub', p])

# --

densefeats = np.vstack([d['feats']['sum_dense'] for d in db])
densefeats = normalize(densefeats)

labs_ = [d['fname'] for d in db]

qpath = '/home/ubuntu/whereistheboom/%s' % '421207568_54473_7496901055679028316.jpg'
im = cv2.imread(qpath)
im = cv2.resize(im, (150, 150))

q = ft.local_features(im, np.array([0, 0, 1, 1]))
q = normalize(q).squeeze()

sims = q.dot(densefeats.T)
sims = sorted(zip(labs_, sims), key = lambda x: -x[1])
pprint(sims[0:10])

paths = [sims[i][0] for i in range(10)]
for p in paths:
    subprocess.Popen(['rsub', p])

