
import pickle as pkl
import dgl.nn as dglnn
import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn import RelGraphConv

class Cross_Att_GO(nn.Module):
    def __init__(self,in_dim,hidden_dim, head=1):
        super(Cross_Att_GO,self).__init__()
        self.head = head
        
        self.trans_q_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.trans_k_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.trans_v_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])

        self.concat_trans = nn.Linear((hidden_dim)*head, hidden_dim, bias=False)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, mut_emd, nb_emd):
        multi_output = []
        for i in range(self.head):
            q = self.trans_q_list[i](nb_emd)
            k = self.trans_k_list[i](mut_emd)
            v = self.trans_v_list[i](nb_emd)
            att = torch.sum(torch.mul(q, k)/torch.sqrt(torch.tensor(256.0)), dim=-1, keepdim=True)
            alpha = F.softmax(att, dim=1)
            tp = v*alpha
            multi_output.append(tp)

        multi_output = torch.cat(multi_output, dim=-1)
        multi_output = self.concat_trans(multi_output)
        multi_output = self.layernorm(multi_output + nb_emd)
        multi_output = self.layernorm(self.ff(multi_output)+multi_output)
        return multi_output
    
class Cross_Att_PRO(nn.Module):
    def __init__(self,in_dim,hidden_dim, head=1):
        super(Cross_Att_PRO,self).__init__()
        self.head = head
        
        self.trans_q_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.trans_k_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.trans_v_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])

        self.concat_trans = nn.Linear((hidden_dim)*head, hidden_dim, bias=False)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, mut_emd, nb_emd):
        multi_output = []
        for i in range(self.head):
            q = self.trans_q_list[i](nb_emd)
            k = self.trans_k_list[i](mut_emd)
            v = self.trans_v_list[i](nb_emd)
            att = torch.sum(torch.mul(q, k)/torch.sqrt(torch.tensor(256.0)), dim=-2, keepdim=True)
            alpha = F.softmax(att, dim=-2)
            tp = v*alpha
            multi_output.append(tp)

        multi_output = torch.cat(multi_output, dim=-1)
        multi_output = self.concat_trans(multi_output)
        multi_output = self.layernorm(multi_output + nb_emd)
        multi_output = self.layernorm(self.ff(multi_output)+multi_output)
        return multi_output
    
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_feats, activation=nn.LeakyReLU(),allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hidden_feats, out_feats, activation=nn.Sigmoid(),allow_zero_in_degree=True)
    
    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = self.conv2(g, h)
        return h

class AttentionFusion(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Linear(embed_dim, 1, bias=False)  
        self.softmax = nn.Softmax(dim=-3)

    def forward(self, embeddings):
        scores = self.attention(embeddings)  # (n, go_num, 1)
        # print(f'scores:{scores.shape}')
        scores = self.softmax(scores)
        go_embedding_final = torch.sum(scores * embeddings, dim=-3)
        # print(f'go_embedding_final:{go_embedding_final.shape}')
        return go_embedding_final
    


class ESM_MLP(nn.Module):
    def __init__(self, class_nums):
        super(ESM_MLP, self).__init__()
        self.dropout = 0.3
        self.go_num = class_nums
        self.rel_num = 1
        self.output_dim = class_nums
        
        self.output_layer = nn.Sequential(nn.Linear(2560, 2560),
                            nn.ReLU(),
                            nn.Linear(2560, 5120),
                            nn.Dropout(self.dropout),
                            nn.ReLU(),
                            nn.Linear(5120,self.output_dim),
                            nn.Sigmoid())

    def forward(self, pro_esm2):
        output = self.output_layer(pro_esm2)
        return output

class DuGProModel(nn.Module):
    def __init__(self, class_nums, rel_num, g1, g2, g3, g4):
        super(DuGProModel, self).__init__()
        self.dropout = 0.2
        self.go_num = class_nums
        self.embed_dim = 128
        self.rel_num = rel_num
        self.output_dim = class_nums
        self.g = [g for g in [g1, g2, g3, g4] if g is not None]

        self.node_embed = nn.Embedding(self.go_num, self.embed_dim)
        torch.nn.init.xavier_uniform_(self.node_embed.weight)

        self.gcn_list = nn.ModuleList([GCN(self.embed_dim, 2*self.embed_dim ,self.embed_dim) for _ in range(self.rel_num)])

         # initialize rel_embedding
        self.rel_embed = nn.Embedding(self.rel_num, 128)
        torch.nn.init.xavier_uniform_(self.rel_embed.weight)
        
        self.pro_layer = nn.Sequential(nn.Linear(2560, 2560),
                                        nn.ReLU(),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(2560,512),
                                        nn.ReLU(),
                                        nn.Linear(512,128))

        self.fusion_layer = AttentionFusion(embed_dim=128)
        
        self.weight_layer = nn.Sequential(nn.Linear(2560, 5120),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout),
                                     nn.Linear(5120, 5120),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout),
                                     nn.Linear(5120,self.output_dim),
                                     nn.Sigmoid())

        self.go_layer = nn.Sequential(
                                        nn.Linear(128, 128),
                                        nn.LeakyReLU(),
                                        nn.Linear(128, 128),
                                        nn.Sigmoid())
    
        self.output_layer = nn.Sequential(nn.Linear(128, 128),
                                        nn.LeakyReLU(),
                                        nn.Linear(128, 1),
                                        nn.Sigmoid())

    def forward(self, pro_esm2, head2ids1, tail2ids1, head2ids2=None, tail2ids2=None, head2ids3=None, tail2ids3=None, head2ids4=None, tail2ids4=None):

        #generate protein feature
        pro_feats = self.pro_layer(pro_esm2)
        pro_feats = torch.unsqueeze(pro_feats, dim=1)

        # GO weights
        go_weights =  self.weight_layer(pro_esm2)
        go_weights = torch.unsqueeze(go_weights, dim=1).transpose(-2,-1)

        # GO graph
        node_embed = self.node_embed.weight
        go_embeds = []
        for i in range(self.rel_num):
            go_embed = self.gcn_list[i](self.g[i], node_embed)
            go_embeds.append(go_embed)
        go_embeds = torch.stack(go_embeds,dim=0)
        go_embeds = self.fusion_layer(go_embeds)

        # protein-aware GO
        go_feats = go_weights * go_embeds
        go_feats = self.go_layer(go_feats)  
        
        # fusion and classification
        features = pro_feats * go_feats + pro_feats
        output = self.output_layer(features)
        output = output.transpose(-2, -1)
        output = torch.squeeze(output, dim=-2)

        # knowledge graph constraint
        e_loss = self.calculate_kg_loss(head2ids1, tail2ids1, head2ids2, tail2ids2, head2ids3, tail2ids3, head2ids4, tail2ids4, go_feats)
        
        return output, e_loss
    

    def calculate_kg_loss(self, head2ids1, tail2ids1, head2ids2, tail2ids2, head2ids3, tail2ids3, head2ids4, tail2ids4, embeddings):
        cosine_similarities1 = F.cosine_similarity(embeddings[:,head2ids1,:]+ self.rel_embed.weight[0], embeddings[:,tail2ids1,:], dim=-1)
        loss1 = (1 - cosine_similarities1).mean()
        
        if head2ids2 !=None and len(head2ids2) != 0:
            cosine_similarities2 = F.cosine_similarity(embeddings[:,head2ids2,:] + self.rel_embed.weight[1], embeddings[:,tail2ids2,:], dim=-1) 
            loss2 = (1 - cosine_similarities2).mean()
        else:
            loss2 = torch.tensor(0)

        if head2ids3 !=None and len(head2ids3) != 0:
            cosine_similarities3 = F.cosine_similarity(embeddings[:,head2ids3,:] + self.rel_embed.weight[2], embeddings[:,tail2ids3,:], dim=-1) 
            loss3 = (1 - cosine_similarities3).mean()
        else:
            loss3 = torch.tensor(0)

        if head2ids4 !=None and len(head2ids4) != 0:
            cosine_similarities4 = F.cosine_similarity(embeddings[:,head2ids4,:] + self.rel_embed.weight[3], embeddings[:,tail2ids4,:], dim=-1)
            loss4 = (1 - cosine_similarities4).mean()
        else:
            loss4 = torch.tensor(0)
        return loss1+loss2+loss3+loss4
