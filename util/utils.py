import math
import os
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from scipy import stats
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset
from tqdm import trange
import codecs
import pandas as pd
from subword_nmt.apply_bpe import BPE
from datetime import datetime
import csv
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import random


class myDataset(InMemoryDataset):
    def __init__(self, root='/save_model', dataset='drug_data_Graph',
                 transform=None, pre_transform=None, simle_graph=None, saliency_map=False):

        # root is required for save preprocessed data_WS, default is '/data_WS'
        super(myDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset
        self.dataset = dataset
        # self.similarity = similarity
        # self.raw = raw
        self.saliency_map = saliency_map

        # if os.path.isfile(self.processed_paths[0]):
        #     print('Pre_processed data found: {}, loading...'.format(self.processed_paths[0]))
        #     self.data, self.slices = torch.load(self.processed_paths[0])
        # else:
        # print('Pre-processed data {} not found, doing pre_processing...'.format(self.processed_paths[0]))
        # self.process(simle_graph)
        # self.data, self.slices = torch.load(self.processed_paths[0])
        self.data, self.slices = self.process(simle_graph)

    @property
    def raw_file_names(self):  # 返回一个包含没有处理的数据的名字的list
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):  # 返回一个包含所有处理过的数据名字的list
        return [self.dataset + '.pt']

    def download(self):  # 下载数据集函数，不需要的话直接填充pass
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # feature - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data_WS
    def process(self, simle_graph):
        # assert (len(drug_silmes) == len(frequencyMat)), "The two lists must be the same L!"
        data_list = []
        data_len = len(simle_graph)
        # print(data_len)
        # data_len = trange(data_len)
        # data_len.set_description("Processing ")
        for i in range(data_len):
            # data_len.set_description("Processing ")
            # print('Convert SIMLES to graph: {}/{}'.format(i + 1, data_len))
            # labels = frequencyMat[i]
            # Convert SMILES to molecular representation using rdkit
            batch = []
            smiles, c_size, features, edge_index, edge_type = simle_graph[i]
            for j in range(c_size):
                batch.append(i)
            np_batch = np.array(batch)
            # print(type(edge_index), edge_index,i)
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index),
                                y=torch.IntTensor(edge_type * 2).flatten(),
                                batch=torch.LongTensor(np_batch),
                                smiles=smiles)
            # GCNData.__setitem__('edge_type', torch.IntTensor(edge_type * 2 ).flatten())
            # 记录此特征矩阵x的行开始的坐标，为0；
            # 利用DataLoader读取时，返回一个(1 * batch_size)维度的tensor，代表共batch_size个x,每个x的行从x_index[i]开始
            GCNData.__setitem__('x_index', torch.LongTensor([0]))

            # 记录此SMILES对应在所有SMILES的坐标，用于计算loss时查找对应的frequencyMat的位置
            # 利用DataLoader读取时，返回一个(batch_size * 1)的二维列表
            GCNData.index = [i]  # 输出为二维列表

            # 记录每张smile_graph的原子的个数，即特征矩阵x的行数；
            # 利用DataLoader读取时，返回一个(1 * batch_size)维度的tensor，代表共batch_size个x,每个x有c_size[i]的原子
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))

            data_list.append(GCNData)
            # print(data_list)
        # print(data_list[0])
        # 判断数据对象是否应该保存
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        # 保存到磁盘前进行转化
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # print('Graph construction done. Saving to file.')
        # 将数据对象的python列表整理为内部存储格式，torch_geometric.data_WS.InMemoryDataset
        data, slices = self.collate(data_list)
        # save preprocessed data_WS
        # torch.save((data, slices), self.processed_paths[0])
        return data, slices
        pass


def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean())
    return rmse


def mse(y, f):
    mse = ((y - f) ** 2).mean()
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def MAE(y, f):
    rs = sklearn.metrics.mean_absolute_error(y, f)
    return rs


def ci(y, f):
    ind = np.argsort(y)  # argsort函数返回的是数组值从小到大的索引值
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


def draw_loss(train_losses, test_losses, title, result_folder):
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.ylim((0, 10))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # save image
    plt.savefig(result_folder + '/' + title + ".png")  # should before show method


def draw_pearson(pearsons, title, result_folder):
    plt.figure()
    plt.plot(pearsons, label='test pearson')
    plt.ylim((-0.1, 1))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Pearson')
    plt.legend()
    # save image
    plt.savefig(result_folder + '/' + title + ".png")  # should before show method


def my_draw_loss(train_losses, title, result_folder):
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.ylim((0, 10))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # save image
    plt.savefig(result_folder + '/' + title + ".png")  # should before show method


def my_draw_pearson(pearsons, title, result_folder):
    plt.figure()
    plt.plot(pearsons, label='test pearson')
    plt.ylim((-0.1, 1))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Pearson')
    plt.legend()
    # save image
    plt.savefig(result_folder + '/' + title + ".png")  # should before show method


def my_draw_mse(mse, rmse, title, result_folder):
    plt.figure()
    plt.plot(mse, label='test MSE')
    plt.plot(rmse, label='test rMSE')
    plt.ylim((0, 10))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    # save image
    plt.savefig(result_folder + '/' + title + ".png")  # should before show method


def evaluate_others(M, Tr_neg, Te, positions=[1, 5, 10, 15]):
    """
    :param M: 预测值
    :param Tr_neg: dict， 包含Te
    :param Te:  dict
    :param positions:
    :return:
    """
    prec = np.zeros(len(positions))
    rec = np.zeros(len(positions))
    map_value, auc_value, ndcg = 0.0, 0.0, 0.0
    for u in Te:
        val = M[u, :]
        inx = np.array(Tr_neg[u])
        A = set(Te[u])
        B = set(inx) - A
        # compute precision and recall
        ii = np.argsort(val[inx])[::-1][:max(positions)]
        prec += precision(Te[u], inx[ii], positions)
        rec += recall(Te[u], inx[ii], positions)
        ndcg_user = nDCG(Te[u], inx[ii], 10)
        # compute map and AUC
        pos_inx = np.array(list(A))
        neg_inx = np.array(list(B))
        map_user, auc_user = map_auc(pos_inx, neg_inx, val)
        ndcg += ndcg_user
        map_value += map_user
        auc_value += auc_user
        # outf.write(" ".join([str(map_user), str(auc_user), str(ndcg_user)])+"\n")
    # outf.close()
    return map_value / len(Te.keys()), auc_value / len(Te.keys()), ndcg / len(Te.keys()), prec / len(
        Te.keys()), rec / len(Te.keys())


def precision(actual, predicted, N):
    if isinstance(N, int):
        inter_set = set(actual) & set(predicted[:N])
        return float(len(inter_set)) / float(N)
    elif isinstance(N, list):
        return np.array([precision(actual, predicted, n) for n in N])


def recall(actual, predicted, N):
    if isinstance(N, int):
        inter_set = set(actual) & set(predicted[:N])
        return float(len(inter_set)) / float(len(set(actual)))
    elif isinstance(N, list):
        return np.array([recall(actual, predicted, n) for n in N])


def nDCG(Tr, topK, num=None):
    if num is None:
        num = len(topK)
    dcg, vec = 0, []
    for i in range(num):
        if topK[i] in Tr:
            dcg += 1 / math.log(i + 2, 2)
            vec.append(1)
        else:
            vec.append(0)
    vec.sort(reverse=True)
    idcg = sum([vec[i] / math.log(i + 2, 2) for i in range(num)])
    if idcg > 0:
        return dcg / idcg
    else:
        return idcg


def map_auc(pos_inx, neg_inx, val):
    map = 0.0
    pos_val, neg_val = val[pos_inx], val[neg_inx]
    ii = np.argsort(pos_val)[::-1]
    jj = np.argsort(neg_val)[::-1]
    pos_sort, neg_sort = pos_val[ii], neg_val[jj]
    auc_num = 0.0
    for i, pos in enumerate(pos_sort):
        num = 0.0
        for neg in neg_sort:
            if pos <= neg:
                num += 1
            else:
                auc_num += 1
        map += (i + 1) / (i + num + 1)
    return map / len(pos_inx), auc_num / (len(pos_inx) * len(neg_inx))


def create_model_directory(fold):
    model_dir = "./save_model/model/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    file_name = os.path.basename(__file__).split(".")[0]
    path = os.path.join(model_dir, f"{file_name}{fold}.pt")

    return path

def getBPESmiles(smiles, max_d, device):
    vocab_path = './data/drug_codes_chembl.txt'
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    sub_csv = pd.read_csv('./data/subword_units_map_chembl.csv')
    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))
    drug_emb_tensor = torch.zeros(len(smiles), max_d)
    mask_tensor = torch.zeros(len(smiles), max_d)
    for i in range(len(smiles)):
        t1 = dbpe.process_line(smiles[i]).split()
        try:
            intermediary = np.asarray([words2idx_d[i] for i in t1])
        except:
            intermediary = np.array([0])
            print('error:', smiles[i])

        l = len(intermediary)

        if l < max_d:
            drug_emb_tensor[i] = torch.IntTensor(np.pad(intermediary, (0, max_d - l), 'constant', constant_values=0))
            mask_tensor[i] = torch.tensor(([1] * l) + ([0] * (max_d - l)))

        else:
            drug_emb_tensor[i] = torch.IntTensor(intermediary[:max_d])
            mask_tensor[i] = torch.tensor([1] * max_d)
    return drug_emb_tensor.to(device), mask_tensor.to(device)


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


def random_walk_with_restart(similarity_matrix, alpha=0.5, num_steps=11):
    # 确保相似性矩阵是方阵
    assert similarity_matrix.shape[0] == similarity_matrix.shape[1], "输入矩阵必须是方阵"
    n = similarity_matrix.shape[0]
    identity_matrix = np.eye(n)  # 单位矩阵
    # 初始化RWR矩阵
    rwr_matrix = (1 - alpha) * np.linalg.inv(identity_matrix - alpha * similarity_matrix)
    # 进行多步随机游走
    for step in range(num_steps - 1):
        rwr_matrix = (1 - alpha) * np.linalg.inv(identity_matrix - alpha * similarity_matrix.dot(rwr_matrix))
    return rwr_matrix


def read_raw_data(rawdata_dir, data_train, data_test):
    gii = open(rawdata_dir + '/' + 'semantic.pkl', 'rb')
    side_semantic_rwr = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'word.pkl', 'rb')
    glove_word = pickle.load(gii)
    gii.close()
    side_glove_rwr = cosine_similarity(glove_word)

    gii = open(rawdata_dir + '/' + 'drug_side_frequencies.pkl', 'rb')
    drug_side = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'drug_disease_jaccard.pkl', 'rb')
    drug_disease_rwr = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'drug_drug_scores.pkl', 'rb')
    drug_drug_rwr = pickle.load(gii)
    gii.close()

    for i in range(data_test.shape[0]):
        drug_side[data_test[i, 0], data_test[i, 1]] = 0

    drug_side_label = np.zeros((drug_side.shape[0], drug_side.shape[1]))
    for i in range(drug_side.shape[0]):
        for j in range(drug_side.shape[1]):
            if drug_side[i, j] > 0:
                drug_side_label[i, j] = 1

    side_features = []
    side_drug_label_rwr = cosine_similarity(drug_side_label.T)
    side_features.append(side_semantic_rwr)
    side_features.append(side_glove_rwr)
    side_features.append(side_drug_label_rwr)

    side_features_matrix = side_features[0]
    for i in range(1, len(side_features)):
        side_features_matrix = np.hstack((side_features_matrix, side_features[i]))

    drug_features = [drug_disease_rwr, drug_drug_rwr]
    drug_features_matrix = drug_features[0]
    for i in range(1, len(drug_features)):
        drug_features_matrix = np.hstack((drug_features_matrix, drug_features[i]))
    return side_features_matrix, drug_features_matrix


def fold_files(data_train, data_test, data_neg, args):
    data_train = np.array(data_train)
    data_test = np.array(data_test)

    drug_node_test = data_test[:, 0]
    side_node_test = data_test[:, 1]
    test_label = data_test[:, 2]

    drug_node_train = data_train[:, 0]
    side_node_train = data_train[:, 1]
    train_label = data_train[:, 2]

    side_features_matrix, drug_features_matrix = read_raw_data(args.rawpath, data_train, data_test)

    side_test = side_features_matrix[data_test[:, 1]]
    side_train = side_features_matrix[data_train[:, 1]]

    drug_test = drug_features_matrix[data_test[:, 0]]
    drug_train = drug_features_matrix[data_train[:, 0]]

    return drug_node_train, side_node_train, train_label, drug_node_test, side_node_test, \
           test_label, drug_train, side_train, drug_test, side_test


def Extract_positive_negative_samples(DAL, addition_negative_number='all'):
    k = 0
    interaction_target = np.zeros((DAL.shape[0] * DAL.shape[1], 3)).astype(int)
    for i in range(DAL.shape[0]):
        for j in range(DAL.shape[1]):
            interaction_target[k, 0] = i
            interaction_target[k, 1] = j
            interaction_target[k, 2] = DAL[i, j]
            k = k + 1
    data_shuffle = interaction_target[interaction_target[:, 2].argsort()]
    number_positive = len(np.nonzero(data_shuffle[:, 2])[0])
    final_positive_sample = data_shuffle[interaction_target.shape[0] - number_positive::]
    negative_sample = data_shuffle[0:interaction_target.shape[0] - number_positive]
    a = np.arange(interaction_target.shape[0] - number_positive)
    a = list(a)
    if addition_negative_number == 'all':
        b = random.sample(a, (interaction_target.shape[0] - number_positive))
    else:
        b = random.sample(a, (1 + addition_negative_number) * number_positive)
    final_negtive_sample = negative_sample[b[0:number_positive], :]
    addition_negative_sample = negative_sample[b[number_positive::], :]
    final_sample = np.vstack((final_positive_sample, final_negtive_sample))
    np.random.shuffle(final_sample)
    return addition_negative_sample, final_sample
