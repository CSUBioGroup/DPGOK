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

__all__ = ['fmax', 'aupr', 'ROOT_GO_TERMS', 'compute_performance', 'compute_performance_deepgoplus', 'read_pkl', 'save_pkl']
ROOT_GO_TERMS = {'GO:0003674', 'GO:0008150', 'GO:0005575'}

def fmax(go, targets, scores, idx_goid):
    targets = ssp.csr_matrix(targets)
    
    fmax_ = 0.0, 0.0, 0.0
    precisions = []
    recalls = []
    icprecisions = []
    icrecalls = []
    dpprecisions = []
    dprecalls = []
    goic_list=[]
    godp_list=[]
    for i in range(len(idx_goid)):
        goic_list.append(go.get_ic(idx_goid[i]))
    for i in range(len(idx_goid)):
        godp_list.append(go.get_icdepth(idx_goid[i]))
    goic_vector=np.array(goic_list).reshape(-1,1)
    godp_vector=np.array(godp_list).reshape(-1,1)
    
    for cut in (c / 100 for c in range(101)):
        cut_sc = ssp.csr_matrix((scores >= cut).astype(np.int32))
        correct = cut_sc.multiply(targets).sum(axis=1)
        correct_sc = cut_sc.multiply(targets)
        fp_sc = cut_sc-correct_sc
        fn_sc = targets-correct_sc
        
        correct_ic=ssp.csr_matrix(correct_sc.dot(goic_vector))
        cut_ic=ssp.csr_matrix(cut_sc.dot(goic_vector))
        targets_ic=ssp.csr_matrix(targets.dot(goic_vector))
        
        correct_dp=ssp.csr_matrix(correct_sc.dot(godp_vector))
        cut_dp=ssp.csr_matrix(cut_sc.dot(godp_vector))
        targets_dp=ssp.csr_matrix(targets.dot(godp_vector))
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p, r = correct / cut_sc.sum(axis=1), correct / targets.sum(axis=1)
            p, r = np.average(p[np.invert(np.isnan(p))]), np.average(r)
            
            mi=fp_sc.dot(goic_vector).sum(axis=0)
            ru=fn_sc.dot(goic_vector).sum(axis=0)
            mi/=len(targets.sum(axis=1))
            ru/=len(targets.sum(axis=1))
            
            icp, icr = correct_ic/cut_ic, correct_ic/targets_ic
            icp, icr = np.average(icp[np.invert(np.isnan(icp))]), np.average(icr)
            
            dpp, dpr = correct_dp/cut_dp, correct_dp/targets_dp
            dpp, dpr = np.average(dpp[np.invert(np.isnan(dpp))]), np.average(dpr)
            
        if np.isnan(p):
            precisions.append(0.0)
            recalls.append(r)
        else:
            precisions.append(p)
            recalls.append(r)
            
        if np.isnan(icp):
            icprecisions.append(0.0)
            icrecalls.append(icr)
        else:
            icprecisions.append(icp)
            icrecalls.append(icr)
            
        if np.isnan(dpp):
            dpprecisions.append(0.0)
            dprecalls.append(dpr)
        else:
            dpprecisions.append(dpp)
            dprecalls.append(dpr)
        
        try:
            fmax_ = max(fmax_, (2 * p * r / (p + r) if p + r > 0.0 else 0.0, math.sqrt(ru*ru + mi*mi) , cut))
        except ZeroDivisionError:
            pass
    return fmax_[0], fmax_[1], fmax_[2], precisions, recalls, icprecisions, icrecalls, dpprecisions, dprecalls

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
    # print(len(go_set))
    
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
        if sum(vals) == 0:
            print(i)
        true_scores.append(vals)

        # pred
        vals = [-1]*len(labels)
        for items,score in row.predictions.items():
            if items in go_set:
                vals[goid_idx[items]] = max(score, vals[goid_idx[items]])
            go_parent = go.get_anchestors(items)
            for go_id in go_parent:
                if go_id in go_set:
                    vals[goid_idx[go_id]] = max(vals[goid_idx[go_id]], score)
        pred_scores.append(vals)
    pred_scores = np.array(pred_scores)
    true_scores = np.array(true_scores)
    # print(pred_scores.shape, true_scores.shape, sum(pred_scores<0), sum(pred_scores>0))
    
    result_fmax, result_smin, result_t, precisions, recalls, icprecisions, icrecalls, dpprecisions, dprecalls = fmax(go, true_scores, pred_scores, idx_goid)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    result_aupr = np.trapz(precisions, recalls)
    
    icprecisions = np.array(icprecisions)
    icrecalls = np.array(icrecalls)
    sorted_index = np.argsort(icrecalls)
    icrecalls = icrecalls[sorted_index]
    icprecisions = icprecisions[sorted_index]
    result_icaupr = np.trapz(icprecisions, icrecalls)
    
    dpprecisions = np.array(dpprecisions)
    dprecalls = np.array(dprecalls)
    sorted_index = np.argsort(dprecalls)
    dprecalls = dprecalls[sorted_index]
    dpprecisions = dpprecisions[sorted_index]
    result_dpaupr = np.trapz(dpprecisions, dprecalls)

    return result_fmax, result_smin , result_aupr, result_icaupr, result_dpaupr, result_t

def compute_performance_test(ont, go, all_terms, predscore, true_gos):
    pred_gos = []
    for pred in predscore:
        pred_go = {}
        for i ,score in enumerate(pred):
            pred_go[all_terms[i]] = float(score)
        pred_gos.append(pred_go)
    
    all_gos = []
    # for p in true_gos:
    #     gos = [g for g in true_gos[p]]
    #     all_gos.append(gos)
    for p in true_gos:
        gos = []
        for i, l in enumerate(true_gos[p]):
            if l==1:
                gos.append(all_terms[i])
        all_gos.append(gos)
    save_dict = {}
    # save_dict['protein_id'] = proteins
    save_dict['gos'] = all_gos
    save_dict['predictions'] = pred_gos

    df = pd.DataFrame(save_dict)
    # F_max, Aupr, threadhold = new_compute_performance(df, go, class_tag)
    result_fmax, result_smin , result_aupr, result_icaupr, result_dpaupr, result_t = new_compute_performance(df, go, ont)

    print('Have done', ont, 
          'F_max:', result_fmax, 
          'Aupr:', result_aupr, 
          'threadhold:', result_t,
          'Smin:', result_smin,
          'ICAupr:', result_icaupr,
          'DPAupr:', result_dpaupr,
         )

    return result_fmax, result_smin , result_aupr, result_icaupr, result_dpaupr, result_t


if __name__ == "__main__":
    for ont in ['mf','cc','bp']: 
        print(ont)
        go_file = '../dataset/go.obo'
        go = Ontology(go_file, with_rels=True)
        data = read_pkl('../dataset/train_data_separate.pkl')
        proteins = list(pd.read_csv(f'../dataset/{ont}/test_data_separate_{ont}_proteins.csv')['proteins'])
        all_annotations=[]
        for p in data:
            item_set=set()
            for item in data[p]['annotations']:
                item=item.split('|')[0]
                if go.has_term(item):
                    item_set |= go.get_anchestors(item)
            all_annotations.append(list(item_set))
        go.calculate_ic(all_annotations)
        
        p_label = read_pkl(f'../dataset/{ont}/test_data_separate_{ont}_labels.pkl')
        s_plabels = {}
        for i,p in enumerate(proteins):
            s_plabels[p] = p_label[p]
        term_file = f'../dataset/select_min_count_1_{ont}_labels.csv'
        all_terms = list(pd.read_csv(term_file)['functions'])
        res = read_pkl(f'../results/DuGPro_{ont}_finalres.pkl')
        compute_performance_test(ont, go, all_terms, res, s_plabels)
        