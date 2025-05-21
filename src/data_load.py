import os
import pickle as pkl
import numpy as np
import pandas as pd
from torch.utils import data


BASE_PATH = '../dataset'

def read_pkl(input_file):
    with open(input_file,'rb') as fr:
        temp_result = pkl.load(fr)
    
    return temp_result

class dataSet(data.Dataset):
    def __init__(self, esm2_file, label_file, protein_file):
        super(dataSet,self).__init__()
        self.esm2_feature = read_pkl(esm2_file)
        self.labels = read_pkl(label_file)
        self.proteins = np.array(list(pd.read_csv(protein_file)['proteins']))

    def __getitem__(self,index):
        
        protein = self.proteins[index]
        esm_feats = self.esm2_feature[protein]
        label_embedding = self.labels[protein]
    
        return protein, esm_feats, label_embedding
                

    def __len__(self):
        return len(self.proteins)

class dataSet_test(data.Dataset):
    def __init__(self, esm2_file, protein_file):
        super(dataSet_test,self).__init__()
        self.esm2_feature = read_pkl(esm2_file)
        self.proteins = np.array(list(pd.read_csv(protein_file)['proteins']))

    def __getitem__(self,index):
        protein = self.proteins[index]
        esm_feats = self.esm2_feature[protein]
    
        return protein, esm_feats
                

    def __len__(self):
        return len(self.proteins)
    
class dataSet_cafa(data.Dataset):
    def __init__(self, esm2_file=None, label_file=None, protein_file=None):
        super(dataSet_cafa,self).__init__()
        self.esm2_feature = read_pkl(esm2_file)
        if label_file != None:
            self.labels = read_pkl(label_file)
        else:
            self.labels = None
        self.proteins = np.array(list(pd.read_csv(protein_file)['proteins']))
        
    def __getitem__(self,index):
        
        protein = self.proteins[index]
        esm_feats = self.esm2_feature[protein]
        if self.labels != None:
            label_embedding = self.labels[protein]
        else:
            label_embedding = []
    
        return protein, esm_feats, label_embedding
                

    def __len__(self):
    
        return len(self.proteins)

                