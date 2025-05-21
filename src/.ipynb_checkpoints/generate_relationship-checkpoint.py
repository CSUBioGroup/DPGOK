import pickle as pkl
import pandas as pd
import numpy as np
import math
import os
import gzip
import json
from collections import deque, Counter,defaultdict


def read_pkl(input_file):
    with open(input_file,'rb') as fr:
        temp_result = pkl.load(fr)
    
    return temp_result

def save_pkl(output_file,data):
    with open(output_file,'wb') as fw:
        pkl.dump(data,fw)

class Ontology(object):
    def __init__(self, filename='./uniprot_sprot_train_test_data_oral/go.obo', with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ic = None

    def has_term(self, term_id):
        return term_id in self.ont

    def calculate_ic(self, annots):
        # print(annots[:10])
        # print(type(annots[0]))
        # sys.exit(0)
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])
            self.ic[go_id] = math.log(min_n / n, 2)
    
    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

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
                            obj['is_a'].append(it[1])  #---------------changed
                            # obj['part_of'].append(it[1])
                            
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


    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set

    def get_part_of_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['part_of']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set


    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        if term_id in self.ont:
            return self.ont[term_id]['namespace']
        else:
            return 'can not find'
    
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
    
def get_sparse_matrix_of_go(filepath, go, output):
    go_terms = list(pd.read_csv(filepath)['functions'])
    go_dict = {g:i for i, g in enumerate(go_terms)}
    row = []
    col = []
    for i, g in enumerate(go_terms):
        gps = go.get_anchestors(g)
        for gp in gps:
            if gp in go_terms:
                row.append(i)
                col.append(go_dict[gp])
    sparse_m = {}
    sparse_m['row'] = row
    sparse_m['col'] = col
    print(len(row))
    print(len(col))
    with open(output, 'wb') as fr:
        pkl.dump(sparse_m, fr)

if __name__ == "__main__":
    # go_file = '/public/home/hpc244706074/myProject/dataset/go.obo'
    # go = Ontology(go_file, with_rels=True)
    # for ont in ['mf','bp','cc']:
    #     output = f'/public/home/hpc244706074/myProject/dataset/{ont}_both_anchestors.pkl'
    #     get_sparse_matrix_of_go(f'/public/home/hpc244706074/myProject/dataset/select_min_count_1_{ont}_labels.csv', go, output)
    for ont in ['mf','bp','cc']:
        infile1 = f'/public/home/hpc244706074/myProject/dataset/{ont}_both_anchestors.pkl'
        infile2 = f'/public/home/hpc244706074/myProject/dataset/{ont}_is_a_anchestors.pkl'
        infile3 = f'/public/home/hpc244706074/myProject/dataset/{ont}_part_of_anchestors.pkl'
        data1 = read_pkl(infile1)
        data2 = read_pkl(infile2)
        data3 = read_pkl(infile3)
        tuples1 = set(zip(data1['row'], data1['col']))
        tuples2 = set(zip(data2['row'], data2['col']))
        tuples3 = set(zip(data3['row'], data3['col']))
        dif1 = tuples1-tuples2
        dif2 = tuples1-tuples3
        dif3 = dif1-tuples3
        print(len(dif1))
        print(len(dif2))
        print(len(dif3))
        hidden_rel_row, hidden_rel_col = map(list, zip(*dif3))
        sparse_m = {}
        sparse_m['is_a'] = data2
        sparse_m['part_of'] = data3
        sparse_m['both'] = {'row':hidden_rel_row, 'col':hidden_rel_col}
        outfile = f'/public/home/hpc244706074/myProject/dataset/{ont}_all_relationships_2.pkl'
        save_pkl(outfile, sparse_m)