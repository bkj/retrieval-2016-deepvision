import os
import re
import sys
import cv2
import time
import pickle
import numpy as np
import argparse
from glob import glob
from params import get_params
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# --
# Setup

params = get_params()

sys.path.insert(0, os.path.join(params['fast_rcnn_path'],'caffe-fast-rcnn', 'python'))
sys.path.insert(0, os.path.join(params['fast_rcnn_path'],'lib'))

import caffe
import test as test_ops
from fast_rcnn.config import cfg

cfg.TEST.HAS_RPN = True

# --

class Extractor():
    
    def __init__(self, params):
        self.layer_roi   = params['layer_roi']
        self.layer_dense = params['layer_dense']
        self.dimension   = params['dimension']
        self.pooling     = params['pooling']
        
        self.net = caffe.Net(params['net_proto'], params['net'], caffe.TEST)
    
    def pool_feats(self, feat):        
        pool_func = np.max if self.pooling == 'max' else np.sum
        return pool_func(pool_func(feat, axis=2), axis=1)
    
    def image2features(self, im):
        scores, boxes = test_ops.im_detect(self.net, im, boxes=None)
        
        fc7_feats   = self.net.blobs['fc7'].data.squeeze()
        dense_feats = self.net.blobs[self.layer_dense].data.squeeze()
        roi_feats   = self.net.blobs[self.layer_roi].data.squeeze()
        
        return {
            "scores" : scores,
            "boxes"  : boxes,
            "feats"  : {
                "fc7"        : fc7_feats,
                "sum_dense"  : (dense_feats).sum(axis=2).sum(axis=1),
                "max_dense"  : (dense_feats).max(axis=2).max(axis=1),
                "pooled_roi" : roi_feats.max(axis=2).max(axis=2),
            }
        }
    
    def file2features(self, fname):        
        try:
            tmp = self.image2features(cv2.imread(fname))
            tmp.update({'fname' : fname})
            return tmp
        except:
            print >> sys.stderr, "error at %s" % fname
            return None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, action='store')
    parser.add_argument('--device', type=int, action='store', default=0)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ex   = Extractor(params)
    
    print >> sys.stderr, 'setting device :: %d' % args.device
    # caffe.set_device(args.device)
    caffe.set_mode_gpu()
    
    for inpath in sys.stdin:        
        inpath  = inpath.strip()
        outpath = os.path.join(args.db, re.sub('jpg', 'pkl', inpath.split('/')[-1]))
        
        print >> sys.stderr, "processing :: %s (dev %d)" % (inpath, args.device)
        features = ex.file2features(inpath)
        pickle.dump(features, open(outpath, 'wb'))
