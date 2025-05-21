import pickle as pkl
import pandas as pd
import numpy as np
import sys
import os
import gzip
import json
from collections import deque, Counter,defaultdict

# import argparse
# def parse_args():
#     parser = argparse.ArgumentParser(description='uniprot_API')
#     parser.add_argument('--uniprot_sport_file1', default='./data/uniprot_sprot.dat')
#     parser.add_argument('--uniprot_sport_file2', default='./data/uniprot_sprot.dat')
#     parser.add_argument('--uniprot_sport_file3', default='./data/uniprot_sprot.dat')
#     parser.add_argument('--output_file', default='./uniprot.pkl')
#     parser.add_argument('--go_file', default='./data/go.obo')
#     parser.add_argument('--bp_freq', default=50)
#     parser.add_argument('--cc_freq', default=50)
#     parser.add_argument('--mf_freq', default=50)
    
#     args = parser.parse_args()
#     return args

def read_pkl(input_file):
    with open(input_file,'rb') as fr:
        temp_result = pkl.load(fr)
    
    return temp_result

def save_pkl(output_file,data):
    with open(output_file,'wb') as fw:
        pkl.dump(data,fw)
        
def get_label(anations,func_list):
    temp_result = []
    for label in func_list:
        if label in anations:
            temp_result.append(1)
        else:
            temp_result.append(0)
    return np.array(temp_result)

NAMESPACES = {
    'cc': 'cellular_component',
    'mf': 'molecular_function',
    'bp': 'biological_process'
}

NAMESPACES_reverse = {
     'cellular_component':'cc',
     'molecular_function': 'mf',
     'biological_process':'bp',
}
ROOT_GO_TERMS = {'GO:0003674', 'GO:0008150', 'GO:0005575'}
#实验类型数据
EXP_CODES = set([
    'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC',])
#    'HTP', 'HDA', 'HMP', 'HGI', 'HEP'])

#CAFA中的物种
CAFA_TARGETS = set([
    '10090', '223283', '273057', '559292', '85962',
    '10116',  '224308', '284812', '7227', '9606',
    '160488', '237561', '321314', '7955', '99287',
    '170187', '243232', '3702', '83333', '208963',
    '243273', '44689', '8355'])

taxonTable = {'10116':'RAT','9606':'HUMAN','3702':'ARATH','7955':'DANRE','44689':'DICDI',
    '7227':'DROME','83333':'ECOLI','10090':'MOUSE','208963':'PSEAE',
    '237561':'CANAX','559292':'YEAST','284812':'SCHPO','8355':'XENLA','224308':'BACSU',
    '99287':'SALTY','243232':'METJA','321314':'SALCH','160488':'PSEPK','223283':'PSESM',
    '85962':'HELPY','243273':'MYCGE','170187':'STRPN','273057':'SULSO','all':'all','prokarya':'prokarya','eukarya':'eukarya'}


def is_cafa_target(org):
    return org in CAFA_TARGETS

def is_exp_code(code):
    return code in EXP_CODES

def load_uniport_data(uniprot_file, max_seqlen):
    proteins = list()
    accessions = list()
    sequences = list()
    annotations = list()
    interpros = list()
    orgs = list()
    
    with open(uniprot_file, 'r') as f:
        prot_id = ''
        prot_ac = ''
        seq = ''
        org = ''
        annots = list()
        ipros = list()
        
        for idx, line in enumerate(f):
            items = line.strip().split('   ')
#             items = line.strip().split('\t')
            if items[0] == 'ID' and len(items) > 1:
                if prot_id != '':
                    proteins.append(prot_id)
                    accessions.append(prot_ac)
                    sequences.append(seq)
                    annotations.append(annots)
                    interpros.append(ipros)
                    orgs.append(org)
                prot_id = items[1]
                annots = list()
                ipros = list()
                seq = ''
            elif items[0] == 'AC' and len(items) > 1:
                prot_ac = items[1]
            elif items[0] == 'OX' and len(items) > 1:
                if items[1].startswith('NCBI_TaxID='):
                    org = items[1][11:]
                    end = org.find(' ')
                    org = org[:end]
                else:
                    org = ''
            elif items[0] == 'DR' and len(items) > 1:
                items = items[1].split('; ')
                if items[0] == 'GO':
                    go_id = items[1]
                    code = items[3].split(':')[0]
                    annots.append(go_id + '|' + code)
                if items[0] == 'InterPro':
                    ipro_id = items[1]
                    ipros.append(ipro_id)
            elif items[0] == 'SQ':
                seq = next(f).strip().replace(' ', '')
                while True:
                    sq = next(f).strip().replace(' ', '')
                    if sq == '//':
                        break
                    else:
                        seq += sq
        if len(seq) <= max_seqlen:
            proteins.append(prot_id)
            accessions.append(prot_ac)
            sequences.append(seq)
            annotations.append(annots)
            interpros.append(ipros)
            orgs.append(org)
            
    return proteins, accessions, sequences, annotations, interpros, orgs

class Ontology(object):
    def __init__(self, filename='./uniprot_sprot_train_test_data_oral/go.obo', with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ic = None

    def has_term(self, term_id):
        return term_id in self.ont

    def calculate_ic(self, annots):
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
    
def get_uniport_mess(go_file, input_file, filter_exp, cafa_targets, anchestor_annots, max_seqlen):
    go = Ontology(go_file, with_rels=True)
    proteins, accessions, sequences, annotations, interpros, orgs = load_uniport_data(input_file, max_seqlen)
    
    df = pd.DataFrame({
        'proteins': proteins,
        'accessions': accessions,
        'sequences': sequences,
        'annotations': annotations,
        'interpros': interpros,
        'orgs': orgs
    })
    
    if filter_exp:
        index = []
        annotations = []
        for i, row in enumerate(df.itertuples()):
            annots = []
            for annot in row.annotations:
                go_id, code = annot.split('|')
                if is_exp_code(code):
                    annots.append(annot)

            # Ignore proteins without experimental annotations
            if len(annots) == 0:
                continue
            index.append(i)
            annotations.append(annots)
        df = df.iloc[index]
        df = df.reset_index()
        df['annotations'] = annotations
    
    if cafa_targets:
        index = []
        for i, row in enumerate(df.itertuples()):
            if is_cafa_target(row.orgs):
                index.append(i)
        df = df.iloc[index]
        df = df.reset_index()
    
    if anchestor_annots:
        prop_annotations = []
        for i, row in df.iterrows():
            annot_set = set()
            annots = row['annotations']
            for go_id in annots:
                go_id = go_id.split('|')[0] # In case if it has code
                annot_set |= go.get_anchestors(go_id)
            annots = list(annot_set)
            prop_annotations.append(annots)
        df['annotations'] = prop_annotations
        
    return df

# 5.1 生成预测的label信息  根据train蛋白质中出现功能频率划分label
def get_train_labels(uniprot_df,min_count_fre,go_file,output_path):
    label_df={}
    go = Ontology(go_file, with_rels=True)
    cnt = Counter()
    
    for ii,rows in enumerate(uniprot_df.itertuples()):
        annots = rows.annotations
        
        annots_set = set()
        for go_id in annots:
            go_id,ex_code = go_id.split('|')
            annots_set |= go.get_anchestors(go_id)
        
        for go_id in annots_set:
            cnt[go_id] += 1
    
    for tag in ['bp','cc','mf']:
        res = []
        min_count = min_count_fre[tag]
        for key, val in cnt.items():
            if val >= min_count and go.get_namespace(key) == NAMESPACES[tag]:
                res.append(key)
 
        df = pd.DataFrame({'functions':res})
        df.to_csv('{0}/select_min_count_{1}_{2}_labels.csv'.format(output_path,min_count,tag))
        label_df[tag]=df
    return label_df

def generate_valid_data(uniprot_df1,uniprot_df2):
    data_one = uniprot_df1.set_index('proteins')
    data_two = uniprot_df2
    
    selected_protein = {}
    for protein in data_one.index:
        selected_protein[protein] = []

    count = 0
    temp_count = 0
    index = []
    for i, row in enumerate(data_two.itertuples()):

        protein = row.proteins
    
        if protein in selected_protein:
            continue

        temp_count += 1
        count += 1
        index.append(i)

    data_two = data_two.iloc[index]
    data_two = data_two.reset_index(drop=True)
    return data_two
    
def generate_test_data(uniprot_df1,uniprot_df2,uniprot_df3):
    data_one = uniprot_df1.set_index('proteins')
    data_two = uniprot_df2.set_index('proteins')
    data_three=uniprot_df3
    
    selected_protein_one = {}
    for protein in data_one.index:
        selected_protein_one[protein] = []
        
    selected_protein_two = {}
    for protein in data_two.index:
        selected_protein_two[protein] = []

    count = 0
    temp_count = 0
    index = []
    for i, row in enumerate(data_three.itertuples()):

        protein = row.proteins
    
        if protein in selected_protein_one:
            continue
        if protein in selected_protein_two:
            continue

        temp_count += 1
        count += 1
        index.append(i)

    data_three = data_three.iloc[index]
    data_three = data_three.reset_index(drop=True)
    return data_three

# 5.2 根据之前生成的label信息，对train和test数据集里的蛋白质功能进行性划分,,所有功能寻找到所有的父节点
def separate_protein_data(uniprot_df,go_file,output_file,min_count_frequency,label_df):
    go = Ontology(go_file, with_rels=True)
    
    result = {}
    all_labels = {}
    for tag in ['bp','cc','mf']:
        all_labels[tag] = list(label_df[tag]['functions'])
        
    for row in uniprot_df.itertuples():

        protein = row.proteins
        accessions = row.accessions
        sequences = row.sequences
        annotations = row.annotations
        interpros = row.interpros
        orgs = row.orgs

        result[protein] = {}

        result[protein]['accessions'] = accessions.split(';')
        result[protein]['sequences'] = sequences
        result[protein]['annotations'] = annotations
        result[protein]['interpros'] = interpros
        result[protein]['orgs'] = orgs

        for tag in ['bp','cc','mf']:
            result[protein]['selected_{0}'.format(tag)] = set()
            result[protein]['all_{0}'.format(tag)] = set()

        
        for anno in annotations:
            anno, ex_code = anno.split('|')
            annots_set = go.get_anchestors(anno)
            for anno2 in annots_set:
                inner_tag = NAMESPACES_reverse[go.get_namespace(anno2)]
                result[protein]['all_{0}'.format(inner_tag)].add(anno2)
                if anno2 in all_labels[inner_tag]:
                    result[protein]['selected_{0}'.format(inner_tag)].add(anno2)
    print(len(result))
    save_pkl(output_file, result)

def seperate_ont_data(min_count_frequency):
    save_path = f'../dataset'
    predict_func = {}    
    for tag in ['bp','cc','mf']:
        predict_func[tag] = list(pd.read_csv("../dataset/select_min_count_{1}_{0}_labels.csv".format(tag,min_count_frequency[tag]))['functions'])
        ont_path = f'../dataset/{tag}'
        if not os.path.exists(ont_path):
            os.mkdir(ont_path)

    for file in ['test_data_separate', 'valid_data_separate', 'train_data_separate']:
        input_file = "../dataset/{0}.pkl".format(file)
        datas = read_pkl(input_file)
        
        for tag in ['bp','cc','mf']:
            gos = {}
            labels = {}
            protein_sequence = {}
            select_proteins = []
            for key,value in datas.items(): #key = protein
                anations = value['selected_{0}'.format(tag)]
                if len(anations) == 0:
                    continue
                # 只有根节点注释的蛋白也过滤
                if len(anations) == 1 and next(iter(anations)) in ROOT_GO_TERMS:
                    continue
                # 获取蛋白质id列表和gos和标签list
                select_proteins.append(key)
                protein_sequence[key] = value['sequences']
                gos[key] = len(value['all_{0}'.format(tag)])
                labels[key] = get_label(anations,predict_func[tag])
                
            df = pd.DataFrame({'proteins':select_proteins})
            df.to_csv(save_path+"/{1}/{0}_{1}_proteins.csv".format(file, tag))
            save_pkl(save_path+"/{1}/{0}_{1}_gos.pkl".format(file, tag),gos)
            save_pkl(save_path+"/{1}/{0}_{1}_labels.pkl".format(file, tag),labels)
            save_pkl(save_path+"/{1}/{0}_{1}_sequences.pkl".format(file, tag),protein_sequence)
            print(file, tag, len(select_proteins))

def seq_fasta(input_file, output_file):
    data = read_pkl(input_file)
    with open(output_file,'w') as fw:
        for p in data:
            fw.write('>'+p+'\n')
            seq = data[p]['sequences']
            fw.write(seq+'\n')

def main(uniprot_sport_file1,uniprot_sport_file2,uniprot_sport_file3,output_path,go_file,min_count_frequency, max_seqlen):
    filter_exp = True
    cafa_targets = True
    anchestor_annots = False
    uniprot_df1=get_uniport_mess(go_file, uniprot_sport_file1, filter_exp, cafa_targets, anchestor_annots, max_seqlen)
    uniprot_df2=get_uniport_mess(go_file, uniprot_sport_file2, filter_exp, cafa_targets, anchestor_annots, max_seqlen)
    uniprot_df3=get_uniport_mess(go_file, uniprot_sport_file3, filter_exp, cafa_targets, anchestor_annots, max_seqlen)
    valid_df=generate_valid_data(uniprot_df1,uniprot_df2)
    test_df=generate_test_data(uniprot_df1,uniprot_df2,uniprot_df3)
    label_df=get_train_labels(uniprot_df1, min_count_frequency, go_file,output_path)
    separate_protein_data(uniprot_df1,go_file,output_path+'/train_data_separate.pkl',min_count_frequency,label_df)
    separate_protein_data(valid_df,go_file,output_path+'/valid_data_separate.pkl',min_count_frequency,label_df)
    separate_protein_data(test_df,go_file,output_path+'/test_data_separate.pkl',min_count_frequency,label_df)
    seperate_ont_data(min_count_frequency)
    for t in ['test', 'valid', 'train']:
        print(t)
        seq_fasta(f'../dataset/{t}_data_separate.pkl',f'../dataset/{t}_seq.fasta')
    
    
if __name__=='__main__':
    uniprot_sport_file1='../data_cache/uniprot_202201/uniprot_sprot.dat'
    uniprot_sport_file2='../data_cache/uniprot_202301/uniprot_sprot.dat'
    uniprot_sport_file3='../data_cache/uniprot_202405/uniprot_sprot.dat'
    output_path='../dataset'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    go_file='../data_cache/go.obo'

    min_count_frequency={'bp':1,'cc':1,'mf':1}
    max_seqlen = 100000
    main(uniprot_sport_file1,uniprot_sport_file2,uniprot_sport_file3,output_path,go_file,min_count_frequency, max_seqlen)
    