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


def learn_transform(params, feats):
    normalize(feats)
    pca = PCA(params['dimension'], whiten=True)
    pca.fit(feats)
    pickle.dump(pca,open(params['pca_model'] + '_' + params['dataset'] + '.pkl','wb'))


class Extractor():

    def __init__(self,params):
        
        self.dimension = params['dimension']
        self.dataset   = params['dataset']
        self.pooling   = params['pooling']
        
        self.database_list = open(params['frame_list'],'r').read().splitlines()
        
        self.layer         = params['layer']
        self.save_db_feats = params['database_feats']
        
        print "Extracting from:", params['net_proto']
        self.net = caffe.Net(params['net_proto'], params['net'], caffe.TEST)

    def extract_feat_image(self,image):
        im = cv2.imread(image)
        scores, boxes = test_ops.im_detect(self.net, im, boxes=None)
        return self.net.blobs[self.layer].data.squeeze()

    def pool_feats(self,feat):        
        f = np.max if self.pooling == 'max' else np.sum
        return f(f(feat, axis=2), axis=1)
        
    def save_feats_to_disk(self):
        xfeats  = np.zeros((len(self.database_list), self.dimension))
        
        t0 = time.time()
        for i,frame in enumerate(self.database_list):
            feat = self.extract_feat_image(frame)
            feat = self.pool_feats(feat)
            xfeats[i,:] = feat
            
            if not counter % 5:
                print counter, '/', len(self.database_list), time.time() - t0
        
        pickle.dump(xfeats, open(self.save_db_feats, 'wb'))
        

if __name__ == "__main__":
    params = get_params()
    
    Extractor(params).save_feats_to_disk()
    
    feats = pickle.load(open(params['database_feats'],'rb'))
    learn_transform(params,feats)
