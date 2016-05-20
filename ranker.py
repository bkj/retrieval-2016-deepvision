import os
import time
import random
import pickle

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize
from params import get_params

class Ranker():
    def __init__(self,params):
        
        self.dataset = params['dataset']
        self.other_dataset = 'paris' if self.dataset == 'oxford' else 'oxford'
        
        self.database_images = params['database_images']
        self.dimension       = params['dimension']
        self.pooling         = params['pooling']
        self.N_QE            = params['N_QE']
        self.stage           = params['stage']
        self.rankings_dir    = params['rankings_dir']
        self.distance        = params['distance']
                 
        self.database_list = np.array(open(params['frame_list'],'r').read().splitlines())
        self.query_names   = open(params['query_list'],'r').read().splitlines()
        
        self.pca      = pickle.load(open('%s_%s.pkl' % (params['pca_model'], self.other_dataset), 'rb'))
        self.db_feats = pickle.load(open(params['database_feats'],'rb'))
            
        normalize(self.db_feats)
        if self.pooling is 'sum':
            self.db_feats = self.pca.transform(self.db_feats)
            normalize(self.db_feats)
    
    def get_query_vectors(self):
        self.query_feats = np.zeros((len(self.query_names),self.dimension))
        
        for i,query in enumerate(self.query_names):
            
            query_file, box       = self.query_info(query)
            self.query_feats[i,:] = self.db_feats[np.where(self.database_list == query_file)]
            
            # add top elements of the ranking to the query
            if self.stage is 'QE':
                
                with open(os.path.join(self.rankings_dir,os.path.basename(query.split('_query')[0]) +'.txt'),'r') as f:
                    ranking = f.read().splitlines()
                
                for i_q in range(self.N_QE):
                    
                    imfile = ranking[i_q]
                    
                    if self.dataset is 'paris':
                        imname = os.path.join(self.database_images, imfile.split('_')[1], imfile + '.jpg')
                    elif self.dataset is 'oxford':
                        imname = os.path.join(self.database_images, imfile + '.jpg')
                    
                    # find feature and add to query
                    feat = self.db_feats[np.where(self.database_list == imname)].squeeze()
                    self.query_feats[i,:] += feat
                
                # find feature and add to query
        
        normalize(self.query_feats)
                
    def query_info(self,filename):
        data = np.loadtxt(filename, dtype="str")
        
        bbox = data[1:].astype(float).astype(int)
        
        if self.dataset == 'paris':
            query = os.path.join(self.database_images,query.split('_')[1], data[0] + '.jpg')
        elif self.dataset == 'oxford':
            query = os.path.join(self.database_images, data[0].split('oxc1_')[1] + '.jpg')
        
        return query, bbox 
    
    def write_rankings(self,distances):
        for i,query in enumerate(self.query_names):
            scores   = distances[i,:]
            rankings = self.database_list[np.argsort(scores)]
            savefile = open(os.path.join(self.rankings_dir,os.path.basename(query.split('_query')[0]) +'.txt'),'w')    
            for ranking in rankings:
                savefile.write(os.path.basename(rankings).split('.jpg')[0] + '\n')
            
            savefile.close()
    
    def rank(self):
        self.get_query_vectors()
        distances = pairwise_distances(self.query_feats,self.db_feats,self.distance, n_jobs=-1)
        self.write_rankings(distances)
      

if __name__ == "__main__":
    params = get_params()
    Ranker(params).rank()
