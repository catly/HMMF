
import os

import numpy as np

import torch

from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset

from datetime import datetime
import csv
import pickle

import logging


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # Create the directory structure if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Create nested directories

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

class myDataset(InMemoryDataset):
    def __init__(self, root='/save_model', dataset='drug_data_Graph',
                 transform=None, pre_transform=None, simle_graph=None, saliency_map=False):

        super(myDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.saliency_map = saliency_map

        self.data, self.slices = self.process(simle_graph)

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def process(self, simle_graph):
        data_list = []
        for i, (smiles, c_size, features, edge_index, edge_type) in enumerate(simle_graph):
            batch = [i] * c_size
            np_batch = np.array(batch)
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index),
                                y=torch.IntTensor(edge_type * 2).flatten(),
                                batch=torch.LongTensor(np_batch),
                                smiles=smiles)
            GCNData.__setitem__('x_index', torch.LongTensor([0]))
            GCNData.index = [i]
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            data_list.append(GCNData)

        data, slices = self.collate(data_list)
        return data, slices


def create_model_directory(fold):
    model_dir = "./save_model/model/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    file_name = os.path.basename(__file__).split(".")[0]
    path = os.path.join(model_dir, f"{file_name}{fold}.pt")

    return path

def writePredResultToCsv(list1, list2, list3, list4, list5, list6, name, kfold):
    data_list = []
    path = "./save_model/predresult/" + name + "fold_" + str(kfold) + '_' + datetime.now().strftime("%H%M") + ".csv"
    for a, b, c, d, e, f in zip(list1, list2, list3, list4, list5, list6):
        x = {}
        x['ground_drug'] = a
        x['ground_side'] = b
        x['ground_label_truth'] = c
        x['pred1'] = d
        x['ground_truth'] = e
        x['pred2'] = f
        data_list.append(x)

    with open(path, 'w', newline='', encoding='UTF-8') as f_c_csv:
        writer = csv.writer(f_c_csv)
        writer.writerow(['ground_drug', 'ground_side', 'ground_label_truth', 'pred1', 'ground_truth', 'pred2'])
        for nl in data_list:
            writer.writerow(nl.values())


def readdruginfo(path):
    gii = open(path + '/' + 'side_effect_node_label.pkl', 'rb')
    side_effect_label = pickle.load(gii)
    gii.close()
    return side_effect_label


