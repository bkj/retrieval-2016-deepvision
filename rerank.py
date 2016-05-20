import os
import cv2
import sys
import time
import math
import random
import pickle
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances

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

class Reranker():
    
    def __init__(self,params):
        
        self.layer_roi = 'pool5'
        
        self.dataset             = params['dataset']
        self.other_dataset       = 'paris' if self.dataset == 'oxford' else 'oxford'
        self.database_images     = params['database_images']
        self.dimension           = params['dimension']
        self.layer               = params['layer']
        self.num_rerank          = params['num_rerank']
        self.reranking_path      = params['reranking_path']
        self.use_regressed_boxes = params['use_regressed_boxes']
        self.pooling             = params['pooling']
        self.stage               = params['stage']
        self.N_QE                = params['N_QE']
        self.use_class_scores    = params['use_class_scores']
        self.distance            = params['distance']
        self.rankings_dir        = params['rankings_dir']
        self.queries             = params['query_names']
        
        self.database_list = open(params['frame_list'],'r').read().splitlines()      
        self.query_names   = open(params['query_list'],'r').read().splitlines()
        
        self.net = caffe.Net(params['net_proto'], params['net'], caffe.TEST)
        
        if self.pooling == 'sum':
            self.pca = pickle.load(open('%s_%s.pkl' % (params['pca_model'], self.other_dataset), 'rb'))
     
    def read_ranking(self,query):
        inpath = os.path.join(self.rankings_dir,os.path.basename(query.split('_query')[0]) +'.txt')
        return open(inpath,'r').read().splitlines()
    
    def query_info(self,filename):
        data = np.loadtxt(filename, dtype="str")
                
        bbx = data[1:].astype(float).astype(int)
        
        if self.dataset is 'paris':
            query = os.path.join(self.database_images,data[0].split('_')[1],data[0] + '.jpg')
        elif self.dataset is 'oxford':
            query = os.path.join(self.database_images,data[0].split('oxc1_')[1] + '.jpg')
    
        return query, bbx 

    def pool_feats(self, feat):        
        pool_func = np.max if self.pooling == 'max' else np.sum
        return pool_func(pool_func(feat, axis=1), axis=1)

    def get_query_local_feat(self, query, box=None):
        im   = cv2.imread(query)
        _    = test_ops.im_detect(self.net, im, boxes = None)
        feat = self.net.blobs[self.layer].data.squeeze()
        
        height, width = im.shape
        mult_h = float(np.shape(feat)[1]) / height
        mult_w = float(np.shape(feat)[2]) / width
        
        if box is None:
            query, bbx = self.query_info(query)
            bbx[0] *= mult_w
            bbx[2] *= mult_w
            bbx[1] *= mult_h
            bbx[3] *= mult_h
        else:
            xmin, ymin, xmax, ymax = box[0:4]
            bbx = [int(math.floor(xmin*mult_w)),int(math.floor(ymin*mult_h)),int(math.ceil(xmax*mult_w)),int(math.ceil(ymax*mult_h))]
        
        local_feat = feat[:,bbx[1]:bbx[3],bbx[0]:bbx[2]]
        return self.pool_feats(local_feat)
    
    def rerank_one_query(self,query):
        ranking = self.read_ranking(query)
        
        query_name = os.path.basename(query).rsplit('_',2)[0]
        
        query_feats = self.get_query_local_feat(query)
        query_feats = query_feats.reshape(-1, 1)
        
        # if self.stage is 'rerank2nd':
        #     # second stage of reranking. taking N locations at top N ranking as queries...
            
        #     with open(os.path.join(self.reranking_path,os.path.basename(query.split('_query')[0]) + '.pkl') ,'rb') as f:
        #         distances = pickle.load(f)
        #         locations = pickle.load(f)
        #         frames    = pickle.load(f)
        #         class_ids = pickle.load(f)
                
        #     frames_sorted    = np.array(frames)[np.argsort(distances)]
        #     locations_sorted = np.array(locations)[np.argsort(distances)]
            
        #     for i_qe in range(self.N_QE):
        #         query_feats += self.get_query_local_feat(frames_sorted[i_qe],locations_sorted[i_qe])
            
        #     query_feats/=(self.N_QE+1)
        
        query_feats = query_feats.T
        normalize(query_feats)
        
        if self.pooling is 'sum':
            query_feats = self.pca.transform(query_feats)
            normalize(query_feats)
        
        distances, locations, frames, class_ids = self.rerank_num_rerank(query_feats, ranking, query_name)
        
        outpath = os.path.join(self.reranking_path,os.path.basename(query.split('_query')[0]) + '.pkl')
        with open(outpath ,'wb') as f:
            pickle.dump(distances,f)
            pickle.dump(locations,f)
            pickle.dump(frames,f)
            pickle.dump(class_ids,f)
        
        return query, ranking, distances
    
    def image2features(self, im):
        scores, boxes = test_ops.im_detect(self.net, im, boxes=None, REG_BOXES=self.use_regressed_boxes)
        feat          = self.net.blobs[self.layer_roi].data
        return feat, boxes, scores
    
    def rerank_num_rerank(self,query_feats,ranking,query_name):
        distances = []
        locations = []
        frames    = []
        class_ids = []
        
        # query class (+1 because class 0 is the background)
        cls_ind = np.where(np.array(self.queries) == str(query_name))[0][0] + 1
        
        for im_ in ranking[0:self.num_rerank]:
            
            if self.dataset is 'paris':
                frame_to_read = os.path.join(self.database_images, im_.split('_')[1],im_ + '.jpg')
            elif self.dataset is 'oxford':
                frame_to_read = os.path.join(self.database_images, im_ + '.jpg')
            
            frames.append(frame_to_read)
            # Get features of current element
            feats, boxes, scores = self.image2features(cv2.imread(image))
            
            # we rank based on class scores 
            if self.use_class_scores:
                class_ids.append(cls_ind)
                
                scores         = feats[:,cls_ind]
                distances.append(np.max(scores))
                
                best_pos       = np.argmax(scores)
                best_box_array = boxes[best_pos,:]
                best_box       = best_box_array[4 * cls_ind:4 * (cls_ind + 1)]
                locations.append(best_box)
            else:
                if self.pooling is 'sum':
                    feats = np.sum(np.sum(feats,axis=2),axis=2)
                    normalize(feats)
                    feats = self.pca.transform(feats)
                    normalize(feats)
                else:
                    feats = np.max(np.max(feats,axis=2),axis=2)
                    normalize(feats)
                
                dist_array = pairwise_distances(query_feats, feats, self.distance, n_jobs=-1)
                distances.append(np.min(dist_array))
                
                # Array of boxes with min distance
                idx = np.argmin(dist_array)
                
                # Select array of locations with minimum distance
                best_box_array = boxes[idx,:]
                
                # Discard background score
                scores = scores[:,1:]
                
                # Class ID with max score . 
                cls_ind = np.argmax(scores[idx,:]) 
                class_ids.append(cls_ind+1)
            
                # Select the best box for the best class
                best_box = best_box_array[4*cls_ind:4*(cls_ind + 1)]
                locations.append(best_box)
                
        return distances, locations, frames, class_ids
          
    def rerank_all(self):
        for i,query in enumerate(self.query_names):
            print "Reranking for query", i, "out of", len(iter_), '...'
            query, ranking, distances = self.rerank_one_query(query)
            self.write_rankings(query, ranking, distances)
                        
    def write_rankings(self, query, ranking, distances):
        argdist = np.argsort(distances)
        if self.use_class_scores:
            argdist = argdist[::-1]
        
        ranking[0:self.num_rerank] = np.array(ranking[0:self.num_rerank])[argdist]
        
        savefile = open(os.path.join(self.rankings_dir,os.path.basename(query.split('_query')[0]) +'.txt'),'w')
        for rank in ranking:
            savefile.write(os.path.basename(rank).split('.jpg')[0] + '\n')
        
        savefile.close()
        
if __name__== '__main__':
    Reranker(params).rerank_all()
