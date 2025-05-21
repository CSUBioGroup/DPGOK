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

__all__ = ['fmax', 'aupr', 'ROOT_GO_TERMS', 'compute_performance', 'compute_performance_deepgoplus', 'read_pkl', 'save_pkl']
ROOT_GO_TERMS = {'GO:0003674', 'GO:0008150', 'GO:0005575'}

def fmax(go, targets, scores, idx_goid):
    targets = ssp.csr_matrix(targets)
    
    # fmax_ = 0.0, 0.0, 0.0
    fmax_ = 0.0, 0.0
    precisions = []
    recalls = []
    # icprecisions = []
    # icrecalls = []
    # dpprecisions = []
    # dprecalls = []
    # goic_list=[]
    # godp_list=[]
    # for i in range(len(idx_goid)):
    #     goic_list.append(go.get_ic(idx_goid[i]))
    # for i in range(len(idx_goid)):
    #     godp_list.append(go.get_icdepth(idx_goid[i]))
    # goic_vector=np.array(goic_list).reshape(-1,1)
    # godp_vector=np.array(godp_list).reshape(-1,1)
    
    for cut in (c / 100 for c in range(101)):
        cut_sc = ssp.csr_matrix((scores >= cut).astype(np.int32))
        correct = cut_sc.multiply(targets).sum(axis=1)
        correct_sc = cut_sc.multiply(targets)
        fp_sc = cut_sc-correct_sc
        fn_sc = targets-correct_sc
        
        # correct_ic=ssp.csr_matrix(correct_sc.dot(goic_vector))
        # cut_ic=ssp.csr_matrix(cut_sc.dot(goic_vector))
        # targets_ic=ssp.csr_matrix(targets.dot(goic_vector))
        #
        # correct_dp=ssp.csr_matrix(correct_sc.dot(godp_vector))
        # cut_dp=ssp.csr_matrix(cut_sc.dot(godp_vector))
        # targets_dp=ssp.csr_matrix(targets.dot(godp_vector))
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p, r = correct / cut_sc.sum(axis=1), correct / targets.sum(axis=1)
            p, r = np.average(p[np.invert(np.isnan(p))]), np.average(r)
            
            # mi=fp_sc.dot(goic_vector).sum(axis=0)
            # ru=fn_sc.dot(goic_vector).sum(axis=0)
            # mi/=len(targets.sum(axis=1))
            # ru/=len(targets.sum(axis=1))
            #
            # icp, icr = correct_ic/cut_ic, correct_ic/targets_ic
            # icp, icr = np.average(icp[np.invert(np.isnan(icp))]), np.average(icr)
            #
            # dpp, dpr = correct_dp/cut_dp, correct_dp/targets_dp
            # dpp, dpr = np.average(dpp[np.invert(np.isnan(dpp))]), np.average(dpr)
            
        if np.isnan(p):
            precisions.append(0.0)
            recalls.append(r)
        else:
            precisions.append(p)
            recalls.append(r)
            
        # if np.isnan(icp):
        #     icprecisions.append(0.0)
        #     icrecalls.append(icr)
        # else:
        #     icprecisions.append(icp)
        #     icrecalls.append(icr)
        #
        # if np.isnan(dpp):
        #     dpprecisions.append(0.0)
        #     dprecalls.append(dpr)
        # else:
        #     dpprecisions.append(dpp)
        #     dprecalls.append(dpr)
        
        try:
            # fmax_ = max(fmax_, (2 * p * r / (p + r) if p + r > 0.0 else 0.0, math.sqrt(ru*ru + mi*mi) , cut))
            fmax_ = max(fmax_, (2 * p * r / (p + r) if p + r > 0.0 else 0.0, cut))
        except ZeroDivisionError:
            pass
    # return fmax_[0], fmax_[1], fmax_[2], precisions, recalls, icprecisions, icrecalls, dpprecisions, dprecalls
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
    # #------------------------------removed functions---------------------------
    go_set.remove(FUNC_DICT[ont])
    # # print(len(go_set))
    
    labels = list(go_set)
    # go_set = set(labels)

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
    # print(pred_scores.shape, true_scores.shape, sum(pred_scores<0), sum(pred_scores>0))
    
#     result_fmax, result_smin, result_t, precisions, recalls, icprecisions, icrecalls, dpprecisions, dprecalls = fmax(go, true_scores, pred_scores, idx_goid)
    result_fmax, result_t, precisions, recalls = fmax(go, true_scores, pred_scores, idx_goid)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    result_aupr = np.trapz(precisions, recalls)
    
#     icprecisions = np.array(icprecisions)
#     icrecalls = np.array(icrecalls)
#     sorted_index = np.argsort(icrecalls)
#     icrecalls = icrecalls[sorted_index]
#     icprecisions = icprecisions[sorted_index]
#     result_icaupr = np.trapz(icprecisions, icrecalls)
    
#     dpprecisions = np.array(dpprecisions)
#     dprecalls = np.array(dprecalls)
#     sorted_index = np.argsort(dprecalls)
#     dprecalls = dprecalls[sorted_index]
#     dpprecisions = dpprecisions[sorted_index]
#     result_dpaupr = np.trapz(dpprecisions, dprecalls)

#     return result_fmax, result_smin , result_aupr, result_icaupr, result_dpaupr, result_t
    return result_fmax, result_aupr, result_t


# def generate_result(input_file, output_path, go_file, real_test_protein_mess, all_protein_information, metrics):
#     all_files={}
#     all_files['Your_method']=input_file
    
#     go = Ontology(go_file, with_rels=True)
    
#     all_annotations=[]
#     for key,val in all_protein_information.items():
#         item_set=set()
#         for item in val['annotations']:
#             item=item.split('|')[0]
#             if go.has_term(item):
#                 item_set |= go.get_anchestors(item)
#         all_annotations.append(list(item_set))
#     go.calculate_ic(all_annotations)

#     all_tags = [ 'mf','cc','bp']
#     all_results = OrderedDict()
#     all_results['methods'] = []
#     all_results['methods'].append(math.nan)
#     for m in all_files.keys():
#         all_results['methods'].append(m)
#     all_metrics=['F_max', 'Smin', 'Aupr', 'ICAupr', 'DPAupr', 'threadhold']
#     metric_list=[]
#     for metric in metrics:
#         metric_list.append(all_metrics[int(metric)])

#     for evas in metric_list:
#         for num, tag in enumerate(all_tags):
#             all_results["{0}_{1}".format(evas, num)] = []
#             all_results["{0}_{1}".format(evas, num)].append(tag)


#     for num, tag in enumerate(all_tags):
#         for method,mfile in all_files.items():
#             save_dict = {}
#             save_dict['protein_id'] = []
#             save_dict['gos'] = []
#             save_dict['predictions'] = []

#             with open(mfile, 'rb') as fr:
#                 method_predict_result = pkl.load(fr)

#             for protein,val in method_predict_result.items():
#                 if real_test_protein_mess[protein]['all_{0}'.format(tag)]==set():
#                     continue
#                 if tag not in method_predict_result[protein]:
#                     method_predict_result[protein][tag]={}

#                 save_dict['protein_id'].append(protein)
#                 save_dict['gos'].append(real_test_protein_mess[protein]['all_{0}'.format(tag)])
#                 save_dict['predictions'].append(method_predict_result[protein][tag])  #{go:score}

#             df = pd.DataFrame(save_dict)
#             F_max,Smin,Aupr,ICAupr, DPAupr,threadhold = new_compute_performance(df,go,tag)
            
#             if 'F_max' in metric_list:
#                 all_results["{0}_{1}".format('F_max', num)].append(round(F_max,5))
#             if 'Smin' in metric_list:
#                 all_results["{0}_{1}".format('Smin', num)].append(round(Smin,5))
#             if 'Aupr' in metric_list:
#                 all_results["{0}_{1}".format('Aupr', num)].append(round(Aupr,5))
#             if 'ICAupr' in metric_list:
#                 all_results["{0}_{1}".format('ICAupr', num)].append(round(ICAupr,5))
#             if 'DPAupr' in metric_list:
#                 all_results["{0}_{1}".format('DPAupr', num)].append(round(DPAupr,5))

#             print('Have done', method, tag, 'F_max:', F_max, 'Smin:' ,Smin, 'Aupr:', Aupr,'ICAupr:', ICAupr,'DPAupr:', DPAupr,'threadhold:',threadhold)

#     df = pd.DataFrame(all_results)
#     df.to_csv('{0}/test_evaluation_results.csv'.format(output_path))
    
# def main(input_file, output_path, test_data_file, all_protein_information_file, go_file, metrics):
#     with open(test_data_file,'rb') as f:
#         test_data=pkl.load(f)
    
#     with open(all_protein_information_file,'rb') as f:
#         all_protein_information=pkl.load(f)

#     generate_result(input_file, output_path, go_file, test_data, all_protein_information, metrics)
    
# if __name__=='__main__':
#     args=parse_args()
#     input_file=args.predict
#     output_path=args.output_path
#     if not os.path.exists(output_path):
#         os.mkdir(output_path)
#     test_data_file=args.true
#     all_protein_information_file=args.background
#     go_file=args.go
#     metrics=args.metrics
#     metrics=metrics.strip().split(',')
#     main(input_file, output_path, test_data_file, all_protein_information_file, go_file, metrics)


def compute_performance(proteins, predscore, true_label, class_tag):
    go_file = '/public/home/hpc244706074/myProject/data/go.obo'
    go = Ontology(go_file, with_rels=True)

    file_path = f'/public/home/hpc244706074/myProject/data/select_min_count_1_{class_tag}_labels.csv'
    # file_path = f'/public/home/hpc244706074/myProject/dataset/all_go_{class_tag}_terms.csv'
    all_go = list(pd.read_csv(file_path)['functions'])
    # with open(file_path,'rb') as f:
    #     all_go = pickle.load(f)

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
    # F_max, Aupr, threadhold = new_compute_performance(df, go, class_tag)
    F_max, Aupr, threadhold = new_compute_performance(df, go, all_go, class_tag)

    print('Have done', class_tag, 'F_max:', F_max, 'Aupr:', Aupr, 'threadhold:', threadhold)

    return F_max, Aupr, threadhold

def compute_performance_test(proteins, predscore, true_gos, class_tag):
    go_file = '/public/home/hpc244706074/myProject/data/go.obo'
    go = Ontology(go_file, with_rels=True)

    file_path = f'/public/home/hpc244706074/myProject/data/select_min_count_1_{class_tag}_labels.csv'
    all_go = list(pd.read_csv(file_path)['functions'])

    pred_gos = []
    for pred in predscore:
        pred_go = {}
        for i ,score in enumerate(pred):
            pred_go[all_go[i]] = float(score)
        pred_gos.append(pred_go)
    
    all_gos = []
    for p in proteins:
        gos = [g for g in true_gos[p]]
        all_gos.append(gos)
    save_dict = {}
    save_dict['protein_id'] = proteins
    save_dict['gos'] = all_gos
    save_dict['predictions'] = pred_gos

    df = pd.DataFrame(save_dict)
    # F_max, Aupr, threadhold = new_compute_performance(df, go, class_tag)
    F_max, Aupr, threadhold = new_compute_performance(df, go, class_tag)

    print('Have done', class_tag, 'F_max:', F_max, 'Aupr:', Aupr, 'threadhold:', threadhold)

    return F_max, Aupr, threadhold

def compute_performance_test0(proteins, predscore, true_gos, class_tag):
    go_file = '/public/home/hpc244706074/myProject/dataset/go.obo'
    go = Ontology(go_file, with_rels=True)

    file_path = f'/public/home/hpc244706074/myProject/dataset/select_min_count_1_{class_tag}_labels.csv'
    all_go = list(pd.read_csv(file_path)['functions'])

    pred_gos = []
    for pred in predscore:
        pred_go = {}
        for i ,score in enumerate(pred):
            pred_go[all_go[i]] = float(score)
        pred_gos.append(pred_go)
    
    all_gos = []
    for p in proteins:
        gos = [g for g in true_gos[p]]
        all_gos.append(gos)
    save_dict = {}
    save_dict['protein_id'] = proteins
    save_dict['gos'] = all_gos
    save_dict['predictions'] = pred_gos

    df = pd.DataFrame(save_dict)
    # F_max, Aupr, threadhold = new_compute_performance(df, go, class_tag)
    F_max, Aupr, threadhold = new_compute_performance(df, go, class_tag)

    print('Have done', class_tag, 'F_max:', F_max, 'Aupr:', Aupr, 'threadhold:', threadhold)

    return F_max, Aupr, threadhold

def compute_performance_zeroshot(proteins, predscore, true_gos, class_tag):
    go_file = '/public/home/hpc244706074/myProject/dataset/go.obo'
    go = Ontology(go_file, with_rels=True)

    file_path = f'/public/home/hpc244706074/myProject/dataset/train&test_{class_tag}_labels.csv'
    all_go = list(pd.read_csv(file_path)['functions'])

    pred_gos = []
    for pred in predscore:
        pred_go = {}
        for i ,score in enumerate(pred):
            pred_go[all_go[i]] = float(score)
        pred_gos.append(pred_go)
    
    all_gos = []
    for p in proteins:
        gos = [g for g in true_gos[p]]
        all_gos.append(gos)
    save_dict = {}
    save_dict['protein_id'] = proteins
    save_dict['gos'] = all_gos
    save_dict['predictions'] = pred_gos

    df = pd.DataFrame(save_dict)
    # F_max, Aupr, threadhold = new_compute_performance(df, go, class_tag)
    F_max, Aupr, threadhold = new_compute_performance(df, go, class_tag)

    print('Have done', class_tag, 'F_max:', F_max, 'Aupr:', Aupr, 'threadhold:', threadhold)

    return F_max, Aupr, threadhold

def ev_SOTA():
    # for m in ['ATGO+','deepgose','deepgozero']:
    for m in ['deepgozero']:
        print(m)
        for ont in ['mf', 'cc']:
            proteins = list(pd.read_csv(f'/public/home/hpc244706074/myProject/dataset/{ont}/test_data_separate_{ont}_proteins.csv')['proteins'])
            # res = read_pkl(f'/public/home/hpc244706074/myProject/dataset/other_methods_pre/test_{ont}_{m}.pkl')
            res = read_pkl(f'/public/home/hpc244706074/compared_method_results/test_deepgozero_{ont}_predict.pkl')
            p_labels = read_pkl(f'/public/home/hpc244706074/myProject/dataset/{ont}/test_data_separate_{ont}_labels_goname.pkl')
            
            # pred_score = []
            # for p in proteins:
            #     pred_score.append(res[p])
            pred_score = res
            F_max, Aupr, threadhold = compute_performance_test(proteins, pred_score, p_labels , ont)
            print('mf: aupr:%0.6f,F_max:%.6f,threadhold:%.6f\n' % (
                            (Aupr, F_max, threadhold)))


# 将 g:score 传播映射到训练集的标签中
def blastknn_2_label(ont):
    go_file = '/public/home/hpc244706074/myProject/dataset/go.obo'
    go = Ontology(go_file, with_rels=True)
    proteins = list(pd.read_csv(f'/public/home/hpc244706074/myProject/dataset/{ont}/valid_data_separate_{ont}_proteins.csv')['proteins'])
    gos = list(pd.read_csv(f'/public/home/hpc244706074/myProject/dataset/select_min_count_1_{ont}_labels.csv')['functions'])
    goid_idx = {g:i for i,g in enumerate(gos)}
    blast_res = read_pkl(f'/public/home/hpc244706074/compared_method_results/blast_valid_{ont}_predict_5000.pkl')
    pred_scores = {}
    # Annotations
    for p in proteins:
        # pred
        vals = np.zeros(len(gos),dtype = np.float32)
        for items,score in blast_res[p][ont].items():
    #       print(f'item:{items},score:{score}, vals:{vals[goid_idx[items]]}')
            score = float(score)
            if items in gos:
                vals[goid_idx[items]] = max(score, vals[goid_idx[items]])
            # go_parent = go.get_anchestors(items)
            # for go_id in go_parent:
            #     if go_id in gos:
            #         vals[goid_idx[go_id]] = max(vals[goid_idx[go_id]], score)
        pred_scores[p] = vals
    outfile = f'/public/home/hpc244706074/myProject/dataset/other_methods_pre/valid_{ont}_blastknn.pkl'
    save_pkl(outfile,pred_scores)


def add_blast(ont,res_file):
    proteins = list(pd.read_csv(f'/public/home/hpc244706074/myProject/dataset/{ont}/test_data_separate_{ont}_proteins.csv')['proteins'])
    gos = list(pd.read_csv(f'/public/home/hpc244706074/myProject/dataset/select_min_count_1_{ont}_labels.csv')['functions'])
    go_idx = {g:i for i,g in enumerate(gos)}
    with open(f'/public/home/hpc244706074/myProject/dataset/{ont}/test_data_separate_{ont}_labels_goname.pkl', 'rb') as fr:
        p_labels = pkl.load(fr)
        # labels = []
        # for p in proteins:
        #     list_l = [i for i in p_labels[p]]
        #     labels.append(list_l)
    
    with open(f'/public/home/hpc244706074/compared_method_results/blast_{ont}_predict_5000.pkl', 'rb') as fr:
        blast_res = pkl.load(fr)
    blast_r = []
    for p in proteins:
        r = [0]*len(gos)
        go_score = blast_res[p][ont]
        for g,s in go_score.items():
            r[go_idx[g]] = float(s)
        blast_r.append(r)

    # with open(f'/public/home/hpc244706074/myProject/results/model_v32.1_{ont}_loss.pkl', 'rb') as fr:
    with open(res_file, 'rb') as fr:
        res = pkl.load(fr)

    for i in range(0,11):
        final_res = []
        for j in range(0,len(proteins)):
            r2 = np.array(res['preds'][j]) * (1-i/10) + np.array(blast_r[j]) * (i/10)
            final_res.append(r2)
        print(f'my:{(1-i/10)}; blast:{i/10}')
        F_max, Aupr, threadhold = compute_performance_test(proteins, final_res, p_labels , ont)
        
        print('mf: aupr:%0.6f,F_max:%.6f,threadhold:%.6f\n' % ((Aupr, F_max, threadhold)))


if __name__ == "__main__":
    ev_SOTA()
    # for ont in ['cc']:
    #     print(ont)
    #     resfile = {'mf':f'/public/home/hpc244706074/myProject/results/model_v33.8_{ont}_fmax.pkl', 
    #                 'bp':f'/public/home/hpc244706074/myProject/results/model_v33.8_{ont}_fmax.pkl', 
    #                 'cc':f'/public/home/hpc244706074/myProject/results/model_v33.81_{ont}_fmax.pkl'}
    #     add_blast(ont,resfile[ont])

