import torch
import torch.nn as nn
import numpy as np
import csv
from torch.nn import init
import torch.nn.functional as F
from transformers import BertTokenizer, BertForPreTraining,BertModel
tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

class encoder(nn.Module):
    def __init__(self, args, cuda="cpu", drug_dict={}, drug_smile=[], drug_description=[], is_drug_des=True):
        super(encoder, self).__init__()
        # self.is_drug_des = is_drug_des
        self.is_drug_des = is_drug_des
        self.drug_smile = drug_smile
        self.drug_dict = drug_dict
        self.drug_description = drug_description
        self.embed_dim = args.embed_dim
        self.batch_size = args.batch_size
        self.device = cuda
        bert_model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')

        if self.device.type =="cuda":
            bert_model.load_state_dict(torch.load('./save_model/ckpt_ret01.pt'),strict=False)
            self.bert_model = bert_model.cuda()
        else:
            bert_model.load_state_dict(torch.load('./save_model/ckpt_ret01.pt', map_location=torch.device('cpu')),strict=False)
            self.bert_model = bert_model

        self.outdes = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, self.embed_dim),
        )
        self.outsmi = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, self.embed_dim),
        )

        self.dropout = nn.Dropout(0.2)
        self.deslayer = nn.Linear(768, self.embed_dim)
        self.deslayer1 = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout1 = nn.Dropout(0.2)
        self.smileslayer = nn.Linear(768, self.embed_dim)
        self.smileslayer1 = nn.Linear(self.embed_dim, self.embed_dim)


    def forward(self, nodes):
        #的鬼地方个
        if self.is_drug_des:
            sum_node_des = nodes.cpu().numpy()
            sum_des = []
            sum_logits_des = []
            for i in range(len(sum_node_des)):
                sum_des.append(self.drug_smile[i])
            for i in range(len(sum_node_des)):
                inp_SM = tokenizer.encode(sum_des[i])
                inp_SM = inp_SM[:min(512, len(inp_SM))]
                inp_SM = torch.from_numpy(np.array(inp_SM)).long().unsqueeze(0)
                att_SM = torch.ones(inp_SM.shape).long()
                token_type_ids = torch.zeros(inp_SM.shape).long()
                if self.device.type == "cuda":
                    inp_SM = inp_SM.cuda()
                    att_SM = att_SM.cuda()
                    token_type_ids = token_type_ids.cuda()
                pooled_output_des =self.bert_model(inp_SM, token_type_ids=token_type_ids, attention_mask=att_SM)['pooler_output']
                # pooled_output_des = F.relu(self.deslayer(pooled_output_des.to(self.device)), inplace=True)
                # pooled_output_des = F.dropout(pooled_output_des, training=self.training, p=self.dropout)
                pooled_output_des = self.outdes(pooled_output_des)
                sum_logits_des.append(pooled_output_des)
            sum_tensor_logits_des = torch.cat(sum_logits_des, 0)
            return sum_tensor_logits_des
        else:
            sum_node_smiles = nodes.cpu().numpy()
            # for i in range(len(nodes)):
            #     sum_smiles.append(self.drug_smile[nodes])
            # txt = self.drug_smile[sum_smiles]
            sum_smiles = []
            sum_logits_smi = []
            for i in range(len(sum_node_smiles)):
                sum_smiles.append(self.drug_smile[i])
            for i in range(len(sum_node_smiles)):
                inp_txt = tokenizer.encode(sum_smiles[i])
                inp_txt = inp_txt[:min(512, len(inp_txt))]
                inp_txt = torch.from_numpy(np.array(inp_txt)).long().unsqueeze(0)
                att_txt = torch.ones(inp_txt.shape).long()
                token_type_ids = torch.zeros(inp_txt.shape).long()
                if self.device.type == "cuda":
                    inp_txt = inp_txt.cuda()
                    att_txt = att_txt.cuda()
                    token_type_ids = token_type_ids.cuda()
                pooled_output_smiles = self.bert_model(inp_txt, token_type_ids=token_type_ids, attention_mask=att_txt)['pooler_output']
                pooled_output_smiles = self.outsmi(pooled_output_smiles)

                sum_logits_smi.append(pooled_output_smiles)
            sum_tensor_logits_smi = torch.cat(sum_logits_smi,0)
            return sum_tensor_logits_smi