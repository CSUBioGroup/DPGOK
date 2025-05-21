import os
import pickle as pkl
import numpy as np
import pandas as pd
from torch.utils import data


BASE_PATH = '/public/home/hpc244706074/myProject/dataset'

def read_pkl(input_file):
    with open(input_file,'rb') as fr:
        temp_result = pkl.load(fr)
    
    return temp_result

class dataSet(data.Dataset):
    def __init__(self, esm2_file=None, label_file=None, protein_file=None):
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
    
    def label_embed(self, label, gos):
        embed = np.zeros(len(gos),dtype = np.float32)
        l = [idx for idx, value in enumerate(gos) if value in label]
        embed[l] = 1
        return embed

class dataSet2(data.Dataset):
    def __init__(self, esm2_file=None, label_file=None, protein_file=None, pre_go_file=None):
        super(dataSet2,self).__init__()
        self.esm2_feature = read_pkl(esm2_file)
        self.labels = read_pkl(label_file)
        self.proteins = np.array(list(pd.read_csv(protein_file)['proteins']))
        self.pre_go = read_pkl(pre_go_file)
    def __getitem__(self,index):
        
        protein = self.proteins[index]
        esm_feats = self.esm2_feature[protein]
        label_embedding = self.labels[protein]
        pre_go = self.pre_go[protein]
    
        return protein, esm_feats, label_embedding, pre_go
                

    def __len__(self):
    
        return len(self.proteins)
    
    def label_embed(self, label, gos):
        embed = np.zeros(len(gos),dtype = np.float32)
        l = [idx for idx, value in enumerate(gos) if value in label]
        embed[l] = 1
        return embed
    
class dataSet_test(data.Dataset):
    def __init__(self, esm2_file=None, label_file=None, protein_file=None, gos_file=None):
        super(dataSet_test,self).__init__()
        self.esm2_feature = read_pkl(esm2_file)
        self.labels = read_pkl(label_file)
        self.proteins = np.array(list(pd.read_csv(protein_file)['proteins']))
        self.gos = read_pkl(gos_file)

    def __getitem__(self,index):
        
        protein = self.proteins[index]
        esm_feats = self.esm2_feature[protein]
        label_embedding = self.labels[protein]
        gos = [i for i in self.gos[protein]]

        return protein, esm_feats, label_embedding, gos
                

    def __len__(self):
    
        return len(self.proteins)
    
    def label_embed(self, label, gos):
        embed = np.zeros(len(gos),dtype = np.float32)
        l = [idx for idx, value in enumerate(gos) if value in label]
        embed[l] = 1
        return embed
    

class dataSet3(data.Dataset):
    def __init__(self, esm2_file=None, label_file=None, protein_file=None, interpro_file=None, pre_go_file2=None, pre_go_file3=None):
        super(dataSet3,self).__init__()
        self.esm2_feature = read_pkl(esm2_file)
        self.labels = read_pkl(label_file)
        self.proteins = np.array(list(pd.read_csv(protein_file)['proteins']))
        if interpro_file != None:
            self.interpro = read_pkl(interpro_file)
        else:
            self.interpro = None
        # self.pre_go2 = read_pkl(pre_go_file2)
        # self.pre_go3 = read_pkl(pre_go_file3)
    def __getitem__(self,index):
        
        protein = self.proteins[index]
        esm_feats = self.esm2_feature[protein]
        label_embedding = self.labels[protein]
        if self.interpro != None:
            interpros = self.interpro[protein]

        # pre_go2 = self.pre_go2[protein]
        # pre_go3 = self.pre_go3[protein]
    
        return protein, esm_feats, label_embedding
                

    def __len__(self):
    
        return len(self.proteins)
    
    def label_embed(self, label, gos):
        embed = np.zeros(len(gos),dtype = np.float32)
        l = [idx for idx, value in enumerate(gos) if value in label]
        embed[l] = 1
        return embed

class dataSet4(data.Dataset):
    def __init__(self, esm3_file=None, label_file=None, protein_file=None):
        super(dataSet4,self).__init__()
        self.esm3_feature = read_pkl(esm3_file)
        self.labels = read_pkl(label_file)
        self.proteins = np.array(list(pd.read_csv(protein_file)['proteins']))
    
    def __getitem__(self,index):
        
        protein = self.proteins[index]
        esm_feats = self.esm3_feature[protein]
        label_embedding = self.labels[protein]
    
        return protein, esm_feats, label_embedding
                

    def __len__(self):
    
        return len(self.proteins)
    
    def label_embed(self, label, gos):
        embed = np.zeros(len(gos),dtype = np.float32)
        l = [idx for idx, value in enumerate(gos) if value in label]
        embed[l] = 1
        return embed
    
class dataSet_cafa(data.Dataset):
    def __init__(self, esm2_file=None, label_file=None, protein_file=None):
        super(dataSet_cafa,self).__init__()
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
    