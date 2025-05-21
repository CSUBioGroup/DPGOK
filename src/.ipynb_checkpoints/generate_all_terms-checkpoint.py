import pickle
import warnings
import numpy as np
import scipy.sparse as ssp
# from sklearn.metrics import average_precision_score as aupr
import math
import pandas as pd
from collections import OrderedDict,deque,Counter
import math
import re
import pickle as pkl
import os
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='uniprot_API')
    parser.add_argument('--predict')
    parser.add_argument('--output_path')
    parser.add_argument('--true')
    parser.add_argument('--background')
    parser.add_argument('--go')
    parser.add_argument('--metrics')
    
    args = parser.parse_args()
    return args

__all__ = ['fmax', 'aupr', 'ROOT_GO_TERMS',  'read_pkl', 'save_pkl']
ROOT_GO_TERMS = {'GO:0003674', 'GO:0008150', 'GO:0005575'}


def read_pkl(pklfile):
    with open(pklfile,'rb') as fr:
        data=pkl.load(fr)
    return data

def save_pkl(pklfile, data):
    with open(pklfile,'wb') as fw:
        pkl.dump(data, fw)

BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'
FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS}

NAMESPACES = {
    'cc': 'cellular_component',
    'mf': 'molecular_function',
    'bp': 'biological_process'
}

NAMESPACES_REVERT={
    'cellular_component': 'cc',
    'molecular_function': 'mf',
    'biological_process': 'bp'
}

EXP_CODES = set(['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC',])
CAFA_TARGETS = set([
    '10090', '223283', '273057', '559292', '85962',
    '10116',  '224308', '284812', '7227', '9606',
    '160488', '237561', '321314', '7955', '99287',
    '170187', '243232', '3702', '83333', '208963',
    '243273', '44689', '8355'])

def is_cafa_target(org):
    return org in CAFA_TARGETS

def is_exp_code(code):
    return code in EXP_CODES

class Ontology(object):
    def __init__(self, filename, with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ic = None
        self.icdepth=None

    def has_term(self, term_id):
        return term_id in self.ont

    def calculate_ic(self, annots):
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        self.icdepth={}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])
            self.ic[go_id] = math.log(min_n / n, 2)
            self.icdepth[go_id]=math.log(self.get_depth(go_id,NAMESPACES_REVERT[self.get_namespace(go_id)]),2)*self.ic[go_id]
    
    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]
    
    def get_icdepth(self, go_id):
        if self.icdepth is None:
            raise Exception('Not yet calculated')
        if go_id not in self.icdepth:
            return 0.0
        return self.icdepth[go_id]

    def load(self, filename, with_rels):
        ont = dict()
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['regulates'] = list()
                    obj['alt_ids'] = list()
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        # add all types of relationships
                        if it[0] == 'part_of':
                            # obj['is_a'].append(it[1])
                            obj['part_of'].append(it[1])
                            
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
        if obj is not None:
            ont[obj['id']] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)
        return ont


    def get_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set


    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set
    
    def get_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set
    
    def get_part_of_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['part_of']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set

    def get_part_of_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['part_of']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set
    
    def get_depth(self,term_id,ont):
        q = deque()
        q.append(term_id)
        layer=1
        while(len(q) > 0):
            all_p=set()
            while(len(q)>0):
                t_id = q.popleft()
                p_id=self.get_parents(t_id)
                all_p.update(p_id)
            if all_p:
                layer+=1
                for item in all_p:
                    if item == FUNC_DICT[ont]:
                        return layer
                    q.append(item)
        return layer


    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']
    
    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set
    
if __name__ == "__main__":
    # for ont in ['mf','bp','cc']:
    #     print(ont)
    #     count = 0
    #     # add_go = []
    #     train_go = list(pd.read_csv(f'/public/home/hpc244706074/myProject/dataset/select_min_count_1_{ont}_labels.csv')['functions'])
    #     l1 = len(train_go)
    #     print(l1)
    #     test_go = read_pkl(f'/public/home/hpc244706074/myProject/dataset/{ont}/test_data_separate_{ont}_labels_goname.pkl')
    #     for p, gos in test_go.items():
    #         for g in gos:
    #             if g not in train_go:
    #                 count = count+1
    #                 train_go.append(g)
    #     print(count)
    #     print(len(train_go))
    #     df = pd.DataFrame({'functions':train_go})
    #     df.to_csv(f'/public/home/hpc244706074/myProject/dataset/train&test_{ont}_labels.csv')


    go_file = '/public/home/hpc244706074/myProject/dataset/go.obo'
    go = Ontology(go_file, with_rels=True)
    # for ont in ['mf','bp','cc']:
    #     print(ont)
        # go_set = go.get_namespace_terms(NAMESPACES[ont])
        # print(f'go_number:  {len(go_set)}')
        # file_path = f'/public/home/hpc244706074/myProject/dataset/select_min_count_1_{ont}_labels.csv'
        # train_go = list(pd.read_csv(file_path)['functions'])
        # difference = list(go_set-set(train_go))
        # all_terms = [*train_go, *difference]
        # print(len(all_terms))
        # df = pd.DataFrame({'functions':all_terms})
        # df.to_csv(f'/public/home/hpc244706074/myProject/dataset/all_go_{ont}_terms.csv')
        # all_terms = list(pd.read_csv(f'/public/home/hpc244706074/myProject/dataset/train&test_{ont}_labels.csv')['functions'])
        # go_dict = {g:i for i, g in enumerate(all_terms)}
        # sparse_m = {}
        # sparse_m2 = {}
        # sparse_m['is_a'] = {}
        # row = []
        # col = []
        # for i, g in enumerate(all_terms):
        #     gps = go.get_anchestors(g)
        #     for gp in gps:
        #         if gp in all_terms:
        #             row.append(i)
        #             col.append(go_dict[gp])
        # sparse_m['is_a']['row'] = row
        # sparse_m['is_a']['col'] = col
        # print(f'is_a_all_anchestors: {len(row)}')
        # sparse_m2['is_a'] = {}
        # row = []
        # col = []
        # for i, g in enumerate(all_terms):
        #     gps = go.get_parents(g)
        #     for gp in gps:
        #         if gp in all_terms:
        #             row.append(i)
        #             col.append(go_dict[gp])
        # sparse_m2['is_a']['row'] = row
        # sparse_m2['is_a']['col'] = col
        # print(f'is_a_dp: {len(row)}')

        # sparse_m['part_of'] = {}
        # row = []
        # col = []
        # for i, g in enumerate(all_terms):
        #     gps = go.get_part_of_anchestors(g)
        #     for gp in gps:
        #         if gp in all_terms:
        #             row.append(i)
        #             col.append(go_dict[gp])
        # sparse_m['part_of']['row'] = row
        # sparse_m['part_of']['col'] = col
        # print(f'part_of_all_anchestors: {len(row)}')
        # sparse_m2['part_of'] = {}
        # row = []
        # col = []
        # for i, g in enumerate(all_terms):
        #     gps = go.get_part_of_parents(g)
        #     for gp in gps:
        #         if gp in all_terms:
        #             row.append(i)
        #             col.append(go_dict[gp])
        # sparse_m2['part_of']['row'] = row
        # sparse_m2['part_of']['col'] = col
        # print(f'part_of_dp: {len(row)}')

        # save_pkl(f'/public/home/hpc244706074/myProject/dataset/train&test_{ont}_terms_direct_parents', sparse_m2)
        # save_pkl(f'/public/home/hpc244706074/myProject/dataset/{ont}_train_terms_direct_parents', sparse_m2)


    for ont in ['mf','bp','cc']:
        print(ont)
        # file_path = f'/public/home/hpc244706074/myProject/dataset/all_go_{ont}_terms.csv'
        # all_terms = list(pd.read_csv(file_path)['functions'])
        train_gos = list(pd.read_csv(f'/public/home/hpc244706074/myProject/dataset/select_min_count_1_{ont}_labels.csv')['functions'])
        test_train_gos = list(pd.read_csv(f'/public/home/hpc244706074/myProject/dataset/train&test_{ont}_labels.csv')['functions'])
        test_gos = list(set(test_train_gos)-set(train_gos))
        all_terms = test_train_gos
        go_dict = {g:i for i, g in enumerate(all_terms)}
        sparse_m = {}
        sparse_m2 = {}
        sparse_m['is_a'] = {}
        row = []
        col = []
        for g in test_gos:
            gps = go.get_anchestors(g)
            for gp in gps:
                if gp in all_terms:
                    row.append(go_dict[g])
                    col.append(go_dict[gp])
        sparse_m['is_a']['row'] = row
        sparse_m['is_a']['col'] = col
        print(f'is_a_all_anchestors: {len(row)}')

        sparse_m2['is_a'] = {}
        row = []
        col = []
        for g in test_gos:
            gps = go.get_parents(g)
            for gp in gps:
                if gp in all_terms:
                    row.append(go_dict[g])
                    col.append(go_dict[gp])
        sparse_m2['is_a']['row'] = row
        sparse_m2['is_a']['col'] = col
        print(f'is_a_dp: {len(row)}')

        sparse_m2['part_of'] = {}
        row = []
        col = []
        for g in test_gos:
            gps = go.get_part_of_parents(g)
            for gp in gps:
                if gp in all_terms:
                    row.append(go_dict[g])
                    col.append(go_dict[gp])
        sparse_m2['part_of']['row'] = row
        sparse_m2['part_of']['col'] = col
        print(f'part_of_dp: {len(row)}')

        sparse_m['part_of'] = {}
        row = []
        col = []
        for g in test_gos:
            gps = go.get_part_of_anchestors(g)
            for gp in gps:
                if gp in all_terms:
                    row.append(go_dict[g])
                    col.append(go_dict[gp])
        sparse_m['part_of']['row'] = row
        sparse_m['part_of']['col'] = col
        print(f'part_of_all_anchestors: {len(row)}')
        # save_pkl(f'/public/home/hpc244706074/myProject/dataset/test_{ont}_terms_anchestors', sparse_m)
        save_pkl(f'/public/home/hpc244706074/myProject/dataset/test_{ont}_terms_direct_parents.pkl', sparse_m2)

        # labels = list(go_set)
        # goid_idx = {}
        # idx_goid = {}
        # for idx, goid in enumerate(labels):
        #     goid_idx[goid] = idx
        #     idx_goid[idx] = goid



        
