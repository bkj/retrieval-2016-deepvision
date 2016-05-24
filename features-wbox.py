import os
import sys
import cv2
import time
import pickle
import numpy as np
import argparse
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
    
    def pool_feats(self, feat):        
        pool_func = np.max if self.pooling == 'max' else np.sum
        return pool_func(pool_func(feat, axis=2), axis=1)
    
    def image2features(self, im):
        scores, boxes = test_ops.im_detect(self.net, im, boxes=None)
        feats         = self.net.blobs[self.layer].data.squeeze()
        return {
            'scores' : scores,
            'boxes'  : boxes, 
            'feats'  : self.pool_feats(feats)
        }
    
    def dblist2features(self, dblist, PRINT_INTERVAL=5):        
        t0 = time.time()
        
        out = []
        
        for i,fname in enumerate(dblist):
            
            out_ = self.image2features(cv2.imread(fname))
            out_.update({'fname' : fname})
            out.append(out_)
            
            if not i % PRINT_INTERVAL:
                print '%d/%d in %f seconds' % (i, len(dblist), time.time() - t0)
        
        return out


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--sparse', dest = 'sparse', action="store_true")
#     args = parser.parse_args()

if __name__ == "__main__":
    # Load list of files
    dblist = open(params['frame_list'], 'r').read().splitlines()
    
    # Compute all features
    all_features = Extractor(params).dblist2features(dblist[0:50])
        
    # Learn whitening transformation from embeddings
    all_embs = np.vstack([af['feats'] for af in all_features])
    normalize(all_embs)
    pca = PCA(params['dimension'], whiten=True)
    pca.fit(all_embs)
    
    pickle.dump(all_features, open(params['database_feats_wbox'], 'wb'))
    pickle.dump(pca, open('%s_%s.pkl' % (params['pca_model'], params['dataset']), 'wb'))
