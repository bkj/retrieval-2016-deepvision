import os
import sys
import time
import random
import pickle
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize
from params import get_params

class Ranker():
    def __init__(self,params):
        
        self.dataset         = params['dataset']
        self.database_images = params['database_images']
        self.dimension       = params['dimension']
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
    
    def get_query_vectors(self, db):
        query_feats = np.zeros((len(self.query_names), self.dimension))
        
        for i,filename in enumerate(self.query_names):
            query_file = self.query_info(filename)
            ind = filter(lambda x: x['fname'] == query_file, db)
            if len(ind) > 0:
                query_feats[i,:] = db[ind[0]]['feats']['pooled_dense']
                print >> sys.stderr, 'matching %s' % query_file
            else:
                pass
        
        normalize(query_feats)
        return query_feats
    
    def write_rankings(self, distances):
        for i,query in enumerate(self.query_names):
            scores   = distances[i,:]
            rankings = self.database_list[np.argsort(scores)]
            savefile = open(os.path.join(self.rankings_dir, os.path.basename(query.split('_query')[0]) +'.txt'),'w')    
            for ranking in rankings:
                savefile.write(os.path.basename(ranking).split('.jpg')[0] + '\n')
            
            savefile.close()


def load_database(params):
    other_dataset = 'paris' if params['dataset'] == 'oxford' else 'oxford'
    pca = pickle.load(open('%s_%s.pkl' % (params['pca_model'], other_dataset), 'rb'))
    
    db = pickle.load(open(params['database_feats_wbox']))        
    
    db_feats = np.vstack([d['feats']['pooled_dense'] for d in db])
    normalize(db_feats)
    # if params['pooling'] is 'sum':
    #     db_feats = pca.transform(db_feats)
    #     normalize(db_feats)
    
    return db, db_feats


if __name__ == "__main__":
    params = get_params()
    ranker = Ranker(params)
    
    print >> sys.stderr, 'ranker :: load_database'
    db, db_feats = load_database(params)
    
    print >> sys.stderr, 'ranker :: get_query_vectors'
    query_feats = ranker.get_query_vectors(db)
    
    print >> sys.stderr, 'ranker :: pairwise_distances'
    distances = pairwise_distances(query_feats, db_feats, params['distance'], n_jobs=-1)
    
    print >> sys.stderr, 'ranker :: write_rankings'
    print distances[0:10,0:10]
    ranker.write_rankings(distances)
    