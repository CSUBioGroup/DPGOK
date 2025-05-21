import pickle
import warnings
import numpy as np
import scipy.sparse as ssp
# from sklearn.metrics import average_precision_score as aupr
import math
import pandas as pd
from collections import OrderedDict,deque,Counter
import math
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

__all__ = ['fmax', 'aupr', 'ROOT_GO_TERMS', 'compute_performance', 'compute_performance_deepgoplus', 'read_pkl', 'save_pkl']
ROOT_GO_TERMS = {'GO:0003674', 'GO:0008150', 'GO:0005575'}

def fmax(go, targets, scores, idx_goid):
    targets = ssp.csr_matrix(targets)
    fmax_ = 0.0, 0.0
    precisions = []
    recalls = []
    
    for cut in (c / 100 for c in range(101)):
        cut_sc = ssp.csr_matrix((scores >= cut).astype(np.int32))
        correct = cut_sc.multiply(targets).sum(axis=1)
        correct_sc = cut_sc.multiply(targets)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p, r = correct / cut_sc.sum(axis=1), correct / targets.sum(axis=1)
            p, r = np.average(p[np.invert(np.isnan(p))]), np.average(r)
            
        if np.isnan(p):
            precisions.append(0.0)
            recalls.append(r)
        else:
            precisions.append(p)
            recalls.append(r)

        
        try:
            fmax_ = max(fmax_, (2 * p * r / (p + r) if p + r > 0.0 else 0.0, cut))
        except ZeroDivisionError:
            pass
    return fmax_[0], fmax_[1], precisions, recalls

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
                            obj['is_a'].append(it[1])
                            
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
        
def new_compute_performance(test_df, go, ont):

    go_set = go.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])
    labels = list(go_set)

    goid_idx = {}
    idx_goid = {}
    for idx, goid in enumerate(labels):
        goid_idx[goid] = idx
        idx_goid[idx] = goid

    pred_scores = []
    true_scores = []
    # Annotations
    for i, row in enumerate(test_df.itertuples()):
        # true
        vals = [0]*len(labels)
        annots = set()
        for go_id in row.gos:
            if go.has_term(go_id):
                annots |= go.get_anchestors(go_id)
        for go_id in annots:
            if go_id in go_set:
                vals[goid_idx[go_id]] = 1
        true_scores.append(vals)

        # pred
        vals = [-1]*len(labels)
        for items,score in row.predictions.items():
#             print(f'item:{items},score:{score}, vals:{vals[goid_idx[items]]}')
            score = float(score)
            if items in go_set:
                vals[goid_idx[items]] = max(score, vals[goid_idx[items]])
            go_parent = go.get_anchestors(items)
            for go_id in go_parent:
                if go_id in go_set:
                    vals[goid_idx[go_id]] = max(vals[goid_idx[go_id]], score)
        pred_scores.append(vals)
    pred_scores = np.array(pred_scores)
    true_scores = np.array(true_scores)

    result_fmax, result_t, precisions, recalls = fmax(go, true_scores, pred_scores, idx_goid)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    result_aupr = np.trapz(precisions, recalls)
    
    return result_fmax, result_aupr, result_t


def compute_performance(proteins, predscore, true_label, class_tag):
    go_file = '../dataset/go.obo'
    go = Ontology(go_file, with_rels=True)
    file_path = f'../dataset/select_min_count_1_{class_tag}_labels.csv'
    all_go = list(pd.read_csv(file_path)['functions'])

    pred_gos = []
    for pred in predscore:
        pred_go = {}
        for i ,score in enumerate(pred):
            pred_go[all_go[i]] = float(score)
        pred_gos.append(pred_go)
    true_gos = []
    for label in true_label:
        true_go = []
        for i, l in enumerate(label):
            if l == 1:
                true_go.append(all_go[i])
        true_gos.append(true_go)

    save_dict = {}
    save_dict['protein_id'] = proteins
    save_dict['gos'] = true_gos
    save_dict['predictions'] = pred_gos

    df = pd.DataFrame(save_dict)
    F_max, Aupr, threadhold = new_compute_performance(df, go, class_tag)

    print('Have done', class_tag, 'F_max:', F_max, 'Aupr:', Aupr, 'threadhold:', threadhold)

    return F_max, Aupr, threadhold
