import csv

import numpy as np
import torch
from torch import nn
from transformers import BertTokenizer, BertForPreTraining,BertModel
class BigModel(nn.Module):
    def __init__(self, main_model):
        super(BigModel, self).__init__()
        self.main_model = main_model
    def forward(self, tok, att,device):
        typ = torch.zeros(tok.shape).long()
        if device.type == "cuda":
            typ = typ.cuda()
            tok = tok.cuda()
            att = att.cuda()
        pooled_output = self.main_model(tok, token_type_ids=typ, attention_mask=att)['pooler_output']
        return pooled_output

def get_Drug_smiles_embed(drug_smile,device):
    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    bert_model0 = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
    model = BigModel(bert_model0)
    if device.type == "cuda":
        model.load_state_dict(torch.load('save_model/ckpt_ret01.pt'))
        model = model.cuda()
    else:
        model.load_state_dict(torch.load('save_model/ckpt_ret01.pt', map_location=torch.device('cpu')))

    model.eval()
    sum_logits_smi=[]
    with torch.no_grad():
        for i in range(len(drug_smile)):
            inp_txt = tokenizer.encode(drug_smile[i])
            inp_txt = inp_txt[:min(512, len(inp_txt))]
            inp_txt = torch.from_numpy(np.array(inp_txt)).long().unsqueeze(0)
            att_txt = torch.ones(inp_txt.shape).long()
            logits_des = model(inp_txt, att_txt,device)
            sum_logits_smi.append(logits_des)
    return sum_logits_smi


def get_Drug_smiles_embed_micro(drug_smile,device):
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
    model = BertModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
    model.to(device)
    model.eval()
    sum_logits_smi=[]
    with torch.no_grad():
        for i in range(len(drug_smile)):
            inp_txt = tokenizer.encode(drug_smile[i])
            inp_txt = inp_txt[:min(512, len(inp_txt))]
            inp_txt = torch.from_numpy(np.array(inp_txt)).long().unsqueeze(0).to(device)
            att_txt = torch.ones(inp_txt.shape).long().to(device)
            logits_des = model(inp_txt, att_txt)
            sum_logits_smi.append(logits_des.pooler_output)
    return sum_logits_smi


