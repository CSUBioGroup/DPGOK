import torch
import dgl
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.nn.init import xavier_normal_
import numpy as np
import pickle as pkl
import time
import random
import os
import argparse
from data_load import dataSet
from model import DPGOK
from evaluation import compute_performance

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def weight_init(m):
    if isinstance(m,nn.Linear):
        xavier_normal_(m.weight.data)

def read_pkl(input_file):
    with open(input_file,'rb') as fr:
        temp_result = pkl.load(fr)
    return temp_result

def save_pkl(output_file, data):
    with open(output_file,'wb') as fw:
        pkl.dump(data, fw)


def train_epoch(model, loader, optimizer, epoch, all_epochs, head2ids1,
            tail2ids1,
            head2ids2,
            tail2ids2,
            head2ids3,
            tail2ids3,
            head2ids4,
            tail2ids4,
            print_freq=200):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, (protein, esm_feats, label) in enumerate(loader):
        # Create vaiables
        with torch.no_grad():
            if torch.cuda.is_available():
                esm_var = torch.autograd.Variable(esm_feats.cuda(non_blocking=True))
                label_var = torch.autograd.Variable(label.cuda(non_blocking=True).float())
            else:
                esm_var = torch.autograd.Variable(esm_feats)
                label_var = torch.autograd.Variable(label.float())

        output, eloss = model(esm_var, head2ids1, tail2ids1,
            head2ids2,
            tail2ids2,
            head2ids3,
            tail2ids3,
            head2ids4,
            tail2ids4,)
 
        loss = torch.nn.functional.binary_cross_entropy(output, label_var).cuda() + eloss

        # measure accuracy and record loss
        batch_size = label.size(0)
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, all_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg)
            ])
            print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg

def eval_epoch(model, loader, ont,
            head2ids1,
            tail2ids1,
            head2ids2,
            tail2ids2,
            head2ids3,
            tail2ids3,
            head2ids4,
            tail2ids4,
            print_freq=10, 
            is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    global THREADHOLD

    # Model on eval mode
    model.eval()

    all_proteins = []
    all_preds = []
    all_labels = []
    end = time.time()
    for batch_idx, (protein, esm_feats, label) in enumerate(loader):
        # Create vaiables
        with torch.no_grad():
            if torch.cuda.is_available():
                esm_var = torch.autograd.Variable(esm_feats.cuda(non_blocking=True))
                label_var = torch.autograd.Variable(label.cuda(non_blocking=True).float())
            else:
                esm_var = torch.autograd.Variable(esm_feats)
                label_var = torch.autograd.Variable(label.float())
        output, eloss = model(esm_var,head2ids1, tail2ids1,
            head2ids2,
            tail2ids2,
            head2ids3,
            tail2ids3,
            head2ids4,
            tail2ids4,)
        
        loss = torch.nn.functional.binary_cross_entropy(output, label_var).cuda() + eloss

        # measure accuracy and record loss
        batch_size = label.size(0)
        losses.update(loss.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print status
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Test' if is_test else 'Valid',
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
            ])
            print(res)

        all_preds.append(output.data.cpu().numpy())
        all_proteins.append(protein)
        all_labels.append(label)


    all_preds = np.concatenate(all_preds, axis=0)
    all_proteins = np.concatenate(all_proteins, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    res = {}
    res['proteins'] = all_proteins
    res['preds'] = all_preds
    

    f_max, aupr, t_max = compute_performance(all_proteins, all_preds, all_labels, ont)
    THREADHOLD = t_max
    
    # Return summary statistics
    return batch_time.avg, losses.avg, f_max, t_max, aupr, res


def train(ont, lr, n_epochs, class_nums, batch_size, dp_file, ac_file):
    # load data
    BASE_PATH = '../dataset'
    train_dataset = dataSet(esm2_file=f'{BASE_PATH}/train_esm2.pkl', 
                            label_file=f'{BASE_PATH}/{ont}/train_data_separate_{ont}_labels.pkl', 
                            protein_file=f'{BASE_PATH}/{ont}/train_data_separate_{ont}_proteins.csv')

    valid_dataset = dataSet(esm2_file=f'{BASE_PATH}/valid_esm2.pkl', 
                            label_file=f'{BASE_PATH}/{ont}/valid_data_separate_{ont}_labels.pkl', 
                            protein_file=f'{BASE_PATH}/{ont}/valid_data_separate_{ont}_proteins.csv')

    torch.multiprocessing.set_sharing_strategy('file_system')
    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()),
                                               num_workers=1, drop_last=False, prefetch_factor=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                               pin_memory=(torch.cuda.is_available()),
                                               num_workers=1, drop_last=False, prefetch_factor=4)
    # load graph data
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
    go_adj1 = dgl.graph((ac_pairs['is_a']['head'], ac_pairs['is_a']['tail']), num_nodes=class_nums)
    go_adj2 = dgl.graph((ac_pairs['part_of']['head'], ac_pairs['part_of']['tail']), num_nodes=class_nums)
    go_adj1 = go_adj1.to(device)
    go_adj2 = go_adj2.to(device)
    go_adj3 = None
    go_adj4 = None
    if len(ac_pairs['positively_regulates']['head']) !=0:
        rel_num = rel_num+1
        go_adj3 = dgl.graph((ac_pairs['positively_regulates']['head'], ac_pairs['positively_regulates']['tail']), num_nodes=class_nums)
        go_adj3 = go_adj3.to(device)
        
    if len(ac_pairs['negatively_regulates']['head']) != 0:
        rel_num = rel_num+1
        go_adj4 = dgl.graph((ac_pairs['negatively_regulates']['head'], ac_pairs['negatively_regulates']['tail']), num_nodes=class_nums)
        go_adj4 = go_adj4.to(device)

    # init model
    model = DPGOK(class_nums, rel_num, go_adj1, go_adj2, go_adj3, go_adj4)
    model.apply(weight_init)
    
    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # Train model
    best_loss = 10000
    best_fmax = -1
    for epoch in range(n_epochs):
        _, train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            all_epochs=n_epochs,
            head2ids1 = head2ids1,
            tail2ids1 = tail2ids1,
            head2ids2 = head2ids2,
            tail2ids2 = tail2ids2,
            head2ids3 = head2ids3,
            tail2ids3 = tail2ids3,
            head2ids4 = head2ids4,
            tail2ids4 = tail2ids4)
        
        print('epoch:%03d,train_loss:%0.5f\n' % ((epoch + 1), train_loss))

        _, valid_loss, f_max, t_max, aupr, valid_res = eval_epoch(
            model=model,
            loader=valid_loader,
            ont = ont,
            is_test=False,
            head2ids1 = head2ids1,
            tail2ids1 = tail2ids1,
            head2ids2 = head2ids2,
            tail2ids2 = tail2ids2,
            head2ids3 = head2ids3,
            tail2ids3 = tail2ids3,
            head2ids4 = head2ids4,
            tail2ids4 = tail2ids4,
        )
        
        print(
        'epoch:%03d,valid_loss:%0.5f\naupr:%0.6f,F_max:%.6f,threadhold:%.6f\n' % (
            (epoch + 1), valid_loss, aupr, f_max, t_max))
        
        #直接保存最后5轮的结果
        if epoch >= n_epochs-5:
            torch.save(model.state_dict(),f'../saved_model/DPGOK_{ont}_count_{n_epochs-epoch-1}.dat' )
        if f_max > best_fmax:
            best_fmax = f_max
            print(f"new best f_max:{f_max}(threadhold:{t_max})")
         
        if best_loss > valid_loss:
            best_loss = valid_loss
            print("new best loss:{0}".format(best_loss))
            
            

def seed_torch(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dgl.seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ont", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)
    # lr = {'mf':7e-5, 'cc':5e-5,'bp':5e-5}
    args = parser.parse_args()
    ont = args.ont
    lr = args.lr
    seed_torch()
    batch_size = {'mf':64, 'cc':64, 'bp':16}
    n_epochs = {'mf':25, 'cc':30, 'bp':20}
    class_nums = {'mf':6091, 'bp':18832, 'cc':2604}
    dp_file=f'../dataset/{ont}_direct_parents_pairs.pkl'
    ac_file=f'../dataset/{ont}_ancestors_pairs.pkl'
    print(f'train   {ont}  lr:{lr}  total_epoch:{n_epochs[ont]}')
    train(ont, lr, n_epochs[ont], class_nums[ont], batch_size[ont], dp_file, ac_file)