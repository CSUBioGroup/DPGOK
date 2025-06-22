import torch
from data_load import dataSet_test, read_pkl
from model import DPGOK
import dgl
import numpy as np
import time
import pickle as pkl
import argparse



def predict(ont, class_nums, test_dataset, model_file, dp_file, ac_file, output_path, batch_size):
    torch.multiprocessing.set_sharing_strategy('file_system')

    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                               pin_memory=(torch.cuda.is_available()),
                                               num_workers=1, drop_last=False, prefetch_factor=4)
    
    # load data
    dp_pairs = read_pkl(dp_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    head2ids1 = torch.tensor(dp_pairs['is_a']['head'], dtype=torch.long, device=device)
    tail2ids1 = torch.tensor(dp_pairs['is_a']['tail'], dtype=torch.long, device=device)
    head2ids2 = torch.tensor(dp_pairs['part_of']['head'], dtype=torch.long, device=device)
    tail2ids2 = torch.tensor(dp_pairs['part_of']['tail'], dtype=torch.long, device=device)
    head2ids3 = torch.tensor(dp_pairs['positively_regulates']['head'], dtype=torch.long, device=device)
    tail2ids3 = torch.tensor(dp_pairs['positively_regulates']['tail'], dtype=torch.long, device=device)
    head2ids4 = torch.tensor(dp_pairs['negatively_regulates']['head'], dtype=torch.long, device=device)
    tail2ids4 = torch.tensor(dp_pairs['negatively_regulates']['tail'], dtype=torch.long, device=device)
    
    rel_num = 2
    ac_pairs = read_pkl(ac_file)
    go_adj1 = dgl.graph((ac_pairs['is_a']['head'], ac_pairs['is_a']['tail']), num_nodes=class_nums[ont])
    go_adj2 = dgl.graph((ac_pairs['part_of']['head'], ac_pairs['part_of']['tail']), num_nodes=class_nums[ont])
    go_adj1 = go_adj1.to(device)
    go_adj2 = go_adj2.to(device)
    go_adj3 = None
    go_adj4 = None
    if len(ac_pairs['positively_regulates']['head']) !=0:
        rel_num = rel_num+1
        go_adj3 = dgl.graph((ac_pairs['positively_regulates']['head'], ac_pairs['positively_regulates']['tail']), num_nodes=class_nums[ont])
        go_adj3 = go_adj3.to(device)
        
    if len(ac_pairs['negatively_regulates']['head']) != 0:
        rel_num = rel_num+1
        go_adj4 = dgl.graph((ac_pairs['negatively_regulates']['head'], ac_pairs['negatively_regulates']['tail']), num_nodes=class_nums[ont])
        go_adj4 = go_adj4.to(device)
    
    # init model
    model = DPGOK(class_nums[ont], rel_num, go_adj1, go_adj2, go_adj3, go_adj4)
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))

    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Model on eval mode
    model.eval()

    all_proteins = []
    all_preds = []
    for batch_idx, (protein, esm_feats) in enumerate(loader):
        # Create vaiables
        with torch.no_grad():
            if torch.cuda.is_available():
                esm_var = torch.autograd.Variable(esm_feats.cuda(non_blocking=True))
            else:
                esm_var = torch.autograd.Variable(esm_feats)
        output, _ = model(esm_var,head2ids1, tail2ids1,
            head2ids2,
            tail2ids2,
            head2ids3,
            tail2ids3,
            head2ids4,
            tail2ids4,)

        all_preds.append(output.data.cpu().numpy())
        all_proteins.append(protein)

    all_preds = np.concatenate(all_preds, axis=0)
    all_proteins = np.concatenate(all_proteins, axis=0)
    
    res = {}
    res['proteins'] = all_proteins
    res['preds'] = all_preds
    for i, p in enumerate(all_proteins):
        res[p] = all_preds[i]
    with open(output_path,'wb') as fw:
        pkl.dump(res, fw)

def merge_results(matrix_list, method='mean'):
    if method == 'mean':
        res = np.mean(matrix_list, axis=0)
    elif method == 'max':
        res = np.max(matrix_list, axis=0)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ont", type=str, required=True)
    args = parser.parse_args()
    ont = args.ont
    BASE_PATH = '../dataset'
    class_nums = {'mf':6091, 'bp':18832, 'cc':2604}
    dp_file=f'../dataset/{ont}_direct_parents_pairs.pkl'
    ac_file=f'../dataset/{ont}_ancestors_pairs.pkl'
    test_dataset = dataSet_test(esm2_file=f'{BASE_PATH}/test_esm2.pkl', 
                            protein_file=f'{BASE_PATH}/{ont}/test_data_separate_{ont}_proteins.csv')
    # predict
    for num in range(5):
        output = f'../results/DPGOK_{ont}_res_{num}.pkl'
        model_file=f'../saved_model/DPGOK_{ont}_count_{num}.dat'
        batch_size = 32
        predict(ont, class_nums, test_dataset, model_file, dp_file, ac_file, output, 32)

    # merge results
    res_list = []
    for num in range(5):
        res = read_pkl(f'../results/DPGOK_{ont}_res_{num}.pkl')
        res_list.append(res['preds'])
    final_res = merge_results(res_list)
    with open(f'../results/DPGOK_{ont}_finalres.pkl','wb') as fw:
        pkl.dump(final_res, fw)

    