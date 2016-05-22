import os
import sys
import cv2
import time
import pickle
import numpy as np
from params import get_params
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

params = get_params()

# Add Faster R-CNN module to pythonpath
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

class Extractor():
    
    def __init__(self, params):    
        self.dimension      = params['dimension']
        self.dataset        = params['dataset']
        self.pooling        = params['pooling']
        self.layer          = params['layer']
        
        self.database_list = open(params['frame_list'],'r').read().splitlines()
        
        self.net = caffe.Net(params['net_proto'], params['net'], caffe.TEST)
    
    def pool_feats(self,feat):        
        pool_func = np.max if self.pooling == 'max' else np.sum
        return pool_func(pool_func(feat, axis=2), axis=1)
    
    def image2features(self,im):
        _     = test_ops.im_detect(self.net, im, boxes=None)
        feats = self.net.blobs[self.layer].data.squeeze()
        return self.pool_feats(feats)
    
    def save_feats_to_disk(self, PRINT_INTERVAL=5):
        pooled_feats = np.zeros((len(self.database_list), self.dimension))
        
        t0 = time.time()
        for i,frame in enumerate(self.database_list):
            pooled_feats[i,:] = self.image2features(cv2.imread(frame))
            
            if not counter % PRINT_INTERVAL:
                print counter, '/', len(self.database_list), time.time() - t0
        
        return pooled_feats
        

if __name__ == "__main__":    
    # Compute and save pooled_feats
    pooled_feats = Extractor(params).save_feats_to_disk()
    pickle.dump(pooled_feats, open(params['database_feats'], 'wb'))
    
    # Learn whitening transformation from pooled_feats
    normalize(pooled_feats)
    pca = PCA(params['dimension'], whiten=True)
    pca.fit(pooled_feats)
    pickle.dump(pca, open('%s_%s.pkl' % (params['pca_model'], params['dataset']), 'wb'))
