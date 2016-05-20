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
        
        self.dataset       = params['dataset']
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
        
    def query_info(self, filename):
        data = np.loadtxt(filename, dtype="str")
        
        if self.dataset == 'paris':
            query_file = os.path.join(self.database_images, data[0].split('_')[1], data[0] + '.jpg')
        elif self.dataset == 'oxford':
            query_file = os.path.join(self.database_images, data[0].split('oxc1_')[1] + '.jpg')
        
        return query_file
    
    def get_query_vectors(self, db_feats):
        query_feats = np.zeros((len(self.query_names),self.dimension))
        
        for i,filename in enumerate(self.query_names):
            query_file       = self.query_info(filename)
            query_feats[i,:] = db_feats[np.where(self.database_list == query_file)]
        
        normalize(query_feats)
        return query_feats
    
    def write_rankings(self, distances):
        for i,query in enumerate(self.query_names):
            scores   = distances[i,:]
            rankings = self.database_list[np.argsort(scores)]
            savefile = open(os.path.join(self.rankings_dir, os.path.basename(query.split('_query')[0]) +'.txt'),'w')    
            for ranking in rankings:
                savefile.write(os.path.basename(rankings).split('.jpg')[0] + '\n')
            
            savefile.close()
          

def load_database(params):
    pca      = pickle.load(open('%s_%s.pkl' % (params['pca_model'], self.other_dataset), 'rb'))
    db_feats = pickle.load(open(params['database_feats'], 'rb'))
    normalize(db_feats)
    if self.pooling is 'sum':
        db_feats = pca.transform(db_feats)
        normalize(db_feats)
    
    return db_feats


if __name__ == "__main__":
    params = get_params()

    db_feats    = load_database(params)
    query_feats = Ranker(params).get_query_vectors(db_feats)
    
    distances   = pairwise_distances(query_feats, db_feats, params['distance'], n_jobs=-1)
    ranker.write_rankings(distances)
