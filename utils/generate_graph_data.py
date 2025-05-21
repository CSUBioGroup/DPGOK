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
    def __init__(self, filename='../dataset/go.obo', with_rels=False):
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
                    obj['positively_regulates'] = list()
                    obj['negatively_regulates'] = list()
                    obj['alt_ids'] = list()
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    if obj is not None:
                        ont[obj['id']] = obj
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
                            obj['part_of'].append(it[1])
                        elif it[0] == 'regulates':
                            obj['regulates'].append(it[1])
                        elif it[0] == 'positively_regulates':
                            obj['positively_regulates'].append(it[1])
                        elif it[0] == 'negatively_regulates':
                            obj['negatively_regulates'].append(it[1])
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
        # term_set.remove(term_id)
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
#                 part of 原始的传递性
                for parent_id in self.ont[t_id]['part_of']:
                    if parent_id in self.ont:
                        q.append(parent_id)
#                 part of - is a -> part of
                for parent_id in self.ont[t_id]['part_of']:
                    for pp_id in self.ont[parent_id]['is_a']:
                        if pp_id in self.ont:
                            q.append(pp_id)
#                  is a - part_of -> part_of
                for parent_id in self.ont[t_id]['is_a']:
                    for pp_id in self.ont[parent_id]['part_of']:
                        if pp_id in self.ont:
                            q.append(pp_id)
        # term_set.remove(term_id)   #自相关关系成立
        return term_set
    
    def get_positively_regulates_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
#                原始传递
                for parent_id in self.ont[t_id]['positively_regulates']:
                    if parent_id in self.ont:
                        q.append(parent_id)
#                负负得正
                for parent_id in self.ont[t_id]['negatively_regulates']:
                    for pp_id in self.ont[parent_id]['negatively_regulates']:
                        if pp_id in self.ont:
                            q.append(pp_id)
                # is a - positively_regulates -> positively_regulates
                for parent_id in self.ont[t_id]['is_a']:
                    for pp_id in self.ont[parent_id]['positively_regulates']:
                        if pp_id in self.ont:
                            q.append(pp_id)
#                  positively_regulates - is a  -> positively_regulates           
                for parent_id in self.ont[t_id]['positively_regulates']:
                    for pp_id in self.ont[parent_id]['is_a']:
                        if pp_id in self.ont:
                            q.append(pp_id)
#                  positively_regulates - part of  -> positively_regulates           
                for parent_id in self.ont[t_id]['positively_regulates']:
                    for pp_id in self.ont[parent_id]['part_of']:
                        if pp_id in self.ont:
                            q.append(pp_id)
        term_set.remove(term_id)
        return term_set
    
    def get_negatively_regulates_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
#                负-正->负
                for parent_id in self.ont[t_id]['negatively_regulates']:
                    for pp_id in self.ont[parent_id]['positively_regulates']:
                        if pp_id in self.ont:
                            q.append(pp_id)
#                 is a - negatively_regulates -> negatively_regulates
                for parent_id in self.ont[t_id]['is_a']:
                    for pp_id in self.ont[parent_id]['negatively_regulates']:
                        if pp_id in self.ont:
                            q.append(pp_id)
#                  negatively_regulates - is a  -> negatively_regulates           
                for parent_id in self.ont[t_id]['negatively_regulates']:
                    for pp_id in self.ont[parent_id]['is_a']:
                        if pp_id in self.ont:
                            q.append(pp_id)
#                  negatively_regulates - part of  -> negatively_regulates           
                for parent_id in self.ont[t_id]['negatively_regulates']:
                    for pp_id in self.ont[parent_id]['part_of']:
                        if pp_id in self.ont:
                            q.append(pp_id)
        # 另外加上直接的负调节关系
        for parent_id in self.get_negatively_regulates_parents(term_id):
            term_set.add(parent_id)
        term_set.remove(term_id)
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
        for parent_id in self.ont[term_id]['is_a']:
            for pp_id in self.ont[parent_id]['part_of']:
                if pp_id in self.ont:
                    term_set.add(pp_id)
        for parent_id in self.ont[term_id]['part_of']:
            for pp_id in self.ont[parent_id]['is_a']:
                if pp_id in self.ont:
                    term_set.add(pp_id)
        
        return term_set
    
    def get_positively_regulates_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
#         原始正向调节关系
        for parent_id in self.ont[term_id]['positively_regulates']:
            if parent_id in self.ont:
                term_set.add(parent_id)
#         负负得正获得的正向调节关系
        for parent_id in self.ont[term_id]['negatively_regulates']:
            for pp_id in self.ont[parent_id]['negatively_regulates']:
                if pp_id in self.ont:
                    term_set.add(pp_id)
        for parent_id in self.ont[term_id]['is_a']:
            for pp_id in self.ont[parent_id]['positively_regulates']:
                if pp_id in self.ont:
                    term_set.add(pp_id)
        for parent_id in self.ont[term_id]['positively_regulates']:
            for pp_id in self.ont[parent_id]['is_a']:
                if pp_id in self.ont:
                    term_set.add(pp_id)
        for parent_id in self.ont[term_id]['positively_regulates']:
            for pp_id in self.ont[parent_id]['part_of']:
                if pp_id in self.ont:
                    term_set.add(pp_id)
        return term_set
    
    def get_negatively_regulates_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['negatively_regulates']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        for parent_id in self.ont[term_id]['is_a']:
            for pp_id in self.ont[parent_id]['negatively_regulates']:
                if pp_id in self.ont:
                    term_set.add(pp_id)
        for parent_id in self.ont[term_id]['negatively_regulates']:
            for pp_id in self.ont[parent_id]['is_a']:
                if pp_id in self.ont:
                    term_set.add(pp_id)
        for parent_id in self.ont[term_id]['negatively_regulates']:
            for pp_id in self.ont[parent_id]['part_of']:
                if pp_id in self.ont:
                    term_set.add(pp_id)
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
    
if __name__ == "__main__":
    go_file = '../dataset/go.obo'
    go = Ontology(go_file, with_rels=True)
    for ont in ['mf','bp','cc']:
        print(ont)
        all_terms = list(pd.read_csv(f'../dataset/select_min_count_1_{ont}_labels.csv')['functions'])
        go_dict = {g:i for i, g in enumerate(all_terms)}
        sparse_dp = {}
        sparse_ac = {}
    #   is a relationship
        sparse_ac['is_a'] = {}
        head = []
        tail = []
        for g in all_terms:
            gps = go.get_anchestors(g)
            for gp in gps:
                if gp in all_terms:
                    head.append(go_dict[g])
                    tail.append(go_dict[gp])
        sparse_ac['is_a']['head'] = head
        sparse_ac['is_a']['tail'] = tail
        print(f'is_a_all_anchestors: {len(head)}')

        sparse_dp['is_a'] = {}
        head = []
        tail = []
        for g in all_terms:
            gps = go.get_parents(g)
            for gp in gps:
                if gp in all_terms:
                    head.append(go_dict[g])
                    tail.append(go_dict[gp])
        sparse_dp['is_a']['head'] = head
        sparse_dp['is_a']['tail'] = tail
        print(f'is_a_dp: {len(head)}')
        
    #   part of relationship
        sparse_ac['part_of'] = {}
        head = []
        tail = []
        for g in all_terms:
            gps = go.get_part_of_anchestors(g)
            for gp in gps:
                if gp in all_terms:
                    head.append(go_dict[g])
                    tail.append(go_dict[gp])
        sparse_ac['part_of']['head'] = head
        sparse_ac['part_of']['tail'] = tail
        print(f'part_of_all_anchestors: {len(head)}')
        
        sparse_dp['part_of'] = {}
        head = []
        tail = []
        for g in all_terms:
            gps = go.get_part_of_parents(g)
            for gp in gps:
                if gp in all_terms:
                    head.append(go_dict[g])
                    tail.append(go_dict[gp])
        sparse_dp['part_of']['head'] = head
        sparse_dp['part_of']['tail'] = tail
        print(f'part_of_dp: {len(head)}')
        
    #   positively_regulates relationship
        sparse_ac['positively_regulates'] = {}
        head = []
        tail = []
        for g in all_terms:
            gps = go.get_positively_regulates_anchestors(g)
            for gp in gps:
                if gp in all_terms:
                    head.append(go_dict[g])
                    tail.append(go_dict[gp])
        sparse_ac['positively_regulates']['head'] = head
        sparse_ac['positively_regulates']['tail'] = tail
        print(f'positively_regulates_all_anchestors: {len(head)}')
        
        sparse_dp['positively_regulates'] = {}
        head = []
        tail = []
        for g in all_terms:
            gps = go.get_positively_regulates_parents(g)
            for gp in gps:
                if gp in all_terms:
                    head.append(go_dict[g])
                    tail.append(go_dict[gp])
        sparse_dp['positively_regulates']['head'] = head
        sparse_dp['positively_regulates']['tail'] = tail
        print(f'positively_regulates_dp: {len(head)}')

    #   negatively_regulates relationship
        sparse_ac['negatively_regulates'] = {}
        head = []
        tail = []
        for g in all_terms:
            gps = go.get_negatively_regulates_anchestors(g)
            for gp in gps:
                if gp in all_terms:
                    head.append(go_dict[g])
                    tail.append(go_dict[gp])
        sparse_ac['negatively_regulates']['head'] = head
        sparse_ac['negatively_regulates']['tail'] = tail
        print(f'negatively_regulates_all_anchestors: {len(head)}')
        
        sparse_dp['negatively_regulates'] = {}
        head = []
        tail = []
        for g in all_terms:
            gps = go.get_negatively_regulates_parents(g)
            for gp in gps:
                if gp in all_terms:
                    head.append(go_dict[g])
                    tail.append(go_dict[gp])
        sparse_dp['negatively_regulates']['head'] = head
        sparse_dp['negatively_regulates']['tail'] = tail
        print(f'negatively_regulates_dp: {len(head)}')
        
        save_pkl(f'../dataset/{ont}_ancestors_pairs.pkl', sparse_ac)
        save_pkl(f'../dataset/{ont}_direct_parents_pairs.pkl', sparse_dp)
   