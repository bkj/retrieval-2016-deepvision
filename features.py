import os
import sys
import cv2
import time
import pickle
import numpy as np
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
if params['gpu']:
    caffe.set_mode_gpu()
    caffe.set_device(0)
else:
    caffe.set_mode_cpu()

# --

class Extractor():
    
    def __init__(self, params):    
        self.dimension = params['dimension']
        self.dataset   = params['dataset']
        self.pooling   = params['pooling']
        self.layer     = params['layer']
        
        self.net = caffe.Net(params['net_proto'], params['net'], caffe.TEST)
    
    def pool_feats(self,feat):        
        pool_func = np.max if self.pooling == 'max' else np.sum
        return pool_func(pool_func(feat, axis=2), axis=1)
    
    def image2features(self,im):
        _     = test_ops.im_detect(self.net, im, boxes=None)
        feats = self.net.blobs[self.layer].data.squeeze()
        return self.pool_feats(feats)
    
    def dblist2pfeatures(self, dblist, PRINT_INTERVAL=5):        
        t0 = time.time()
        pooled_feats = np.zeros((len(dblist), self.dimension))
        for i,frame in enumerate(dblist):
            pooled_feats[i,:] = self.image2features(cv2.imread(frame))
            
            if not i % PRINT_INTERVAL:
                print '%d/%d in %f seconds' % (i, len(dblist), time.time() - t0)
        
        return pooled_feats


if __name__ == "__main__":    
    # Compute and save pooled_feats
    dblist = open(params['frame_list'],'r').read().splitlines()
    pooled_feats = Extractor(params).dblist2pfeatures(dblist)
    pickle.dump(pooled_feats, open(params['database_feats'], 'wb'))
    
    # Learn whitening transformation from pooled_feats
    normalize(pooled_feats)
    pca = PCA(params['dimension'], whiten=True)
    pca.fit(pooled_feats)
    pickle.dump(pca, open('%s_%s.pkl' % (params['pca_model'], params['dataset']), 'wb'))
