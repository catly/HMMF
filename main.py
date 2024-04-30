# 最新模型
import os
import random
import pickle
import argparse
import logging
import sys
import copy
import time
from datetime import datetime
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.nn import GATConv, GCNConv, GINConv, RGCNConv
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from util import processSmiles, utils
from util.utils import myDataset
from util.utils import get_logger

base_filename = os.path.basename(__file__).split(".")[0]
log_path = f"./save_model/log/{base_filename}_{datetime.now().strftime('%Y%m%d%H%M')}.log"
loss_path = f"./save_model/loss/{base_filename}_{datetime.now().strftime('%Y%m%d%H%M')}.log"
logger = get_logger(filename=log_path, name="1")
loggerloss = get_logger(filename=loss_path, name="2")


class ConvNCF(nn.Module):
    def __init__(self, drug_embed_description, side_embed_description, sides_dim, embed_dim):
        super(ConvNCF, self).__init__()

        self.drug_embed_description = drug_embed_description
        self.side_embed_description = side_embed_description
        self.sides_dim = sides_dim
        self.embed_dim = embed_dim
        self.outbert_dim = 768
        self.drug_dim = 750
        self.side_dim = sides_dim // 2
        self.atom_dim = 109
        self.gat_embed_dim = 32

        self.dropout = 0.5
        self.dropout1 = 0.6
        self.heads = 10
        self.channel_size = 2 * 4
        self.kernel_size = 2
        self.strides = 2
        self.number_map = 2 * 3

        self.gcn1 = GATConv(self.atom_dim, self.gat_embed_dim, heads=self.heads, dropout=self.dropout)
        self.gcn2 = GATConv(self.gat_embed_dim * self.heads, self.gat_embed_dim, dropout=self.dropout)
        self.GCNFC = nn.Sequential(
            nn.Linear(self.gat_embed_dim, self.embed_dim // 2),
            nn.BatchNorm1d(self.embed_dim // 2, momentum=0.5),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 2, self.embed_dim)
        )
        self.outDrugDescription = self._create_description_layers()
        self.outSideDescription = self._create_description_layers()

        self.outDrugRWR = nn.Sequential(
            nn.Linear(1500, self.embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.outSideRWR = nn.Sequential(
            nn.Linear(2982, self.embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.scalarlayer = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.cnn_interaction = self._create_cnn_layers()

        self.total_layer = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5),
            nn.Dropout(self.dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            nn.Sigmoid(),
        )
        self.con_layer = nn.Linear(self.embed_dim, 1)

    def _create_description_layers(self):
        return nn.Sequential(
            nn.Linear(self.outbert_dim, self.embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5),
            nn.Dropout(self.dropout1),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    def _create_rwr_layers(self):
        return nn.Sequential(
            nn.Linear(self.drug_dim if "Drug" in self.drug_embed_description else self.side_dim, self.embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    def _create_cnn_layers(self):
        layers = []
        in_channels = self.number_map
        for _ in range(5):
            layers += [
                nn.Conv2d(in_channels, self.channel_size, self.kernel_size, stride=self.strides),
                nn.BatchNorm2d(self.channel_size),
                nn.ReLU()
            ]
            in_channels = self.channel_size
        return nn.Sequential(*layers)

    def forward(self, drug_node_train, side_node_train, drug_train, side_train, DrugSmilesGraph, device):
        x, edge_index, batch = DrugSmilesGraph.data.x.to(device), DrugSmilesGraph.data.edge_index.to(device), \
                               DrugSmilesGraph.data.batch.to(device)
        x = F.relu((self.gcn1(x, edge_index)), inplace=True)
        x = F.relu((self.gcn2(x, edge_index)), inplace=True)
        x = global_max_pool(x, batch)

        drug_x = self.GCNFC(x)
        drug_embed_des = self._get_embed_description(drug_node_train, self.drug_embed_description)
        side_embed_des = self._get_embed_description(side_node_train, self.side_embed_description)

        u_embed_description = self.outDrugDescription(drug_embed_des)
        s_embed_description = self.outSideDescription(side_embed_des)

        u_embed_rwr = self.outDrugRWR(drug_train.to(device))
        s_embed_rwr = self.outSideRWR(side_train.to(device))

        maps = self._calculate_interaction_maps(drug_x, u_embed_description, u_embed_rwr, s_embed_description,
                                                s_embed_rwr)
        scalar = self.scalarlayer(torch.sum(maps[1], dim=1))
        feature_map = self.cnn_interaction(maps[0])

        total = self.total_layer(torch.cat((scalar, feature_map.view(-1, self.channel_size * 4 * 4)), dim=1))
        classification = self.classifier(total)
        regression = self.con_layer(total)
        return classification.squeeze(), regression.squeeze()

    def _get_embed_description(self, node_train, embed_description):
        embed_des = []
        for i in range(node_train.size()[0]):
            index = node_train[[i]].cpu().numpy()
            embed_des.append(embed_description[index[0].tolist()])
        return torch.cat(embed_des, 0)

    def _calculate_interaction_maps(self, drug_x, u_embed_description, u_embed_rwr, s_embed_description, s_embed_rwr):
        maps = []
        elementproduct = []
        for i, drug in enumerate([drug_x, u_embed_description, u_embed_rwr]):
            for j, side in enumerate([s_embed_description, s_embed_rwr]):
                maps.append(torch.bmm(drug.unsqueeze(2), side.unsqueeze(1)))
                elementproduct.append(torch.mul(drug, side))

        interaction_map = torch.cat([maps[i].view((-1, 1, self.embed_dim, self.embed_dim)) for i in range(len(maps))],
                                    dim=1)
        elementproduct_map = torch.cat(
            [elementproduct[i].view((-1, 1, self.embed_dim)) for i in range(len(elementproduct))], dim=1)
        return interaction_map, elementproduct_map


def _init_fn(worker_id):
    np.random.seed(int(1024))


def get_embeddings(smiles_list, category, device):
    if category == "side":
        with open("/home/liyang/liuwuyong/HMMF/save_model/processed/pre_side.pkl", "rb") as f:
            embeddings = pickle.load(f)
    if category == "drug":
        with open("/home/liyang/liuwuyong/HMMF/save_model/processed/pre_drug.pkl", "rb") as f:
            embeddings = pickle.load(f)
    embeddings = [embedding.to(device) for embedding in embeddings]
    return embeddings


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


def calculate_precision_recall_at_k(pred_probabilities, ground_truth_labels, precision_at_k_values, recall_at_k_values):
    precision_at_k = {}
    recall_at_k = {}

    for k in precision_at_k_values:
        # 计算Precision@k
        top_k_indices = np.argsort(pred_probabilities)[::-1][:k]
        top_k_labels = ground_truth_labels[top_k_indices]
        precision = np.sum(top_k_labels) / k if k > 0 else 0
        precision_at_k[k] = precision

    for k in recall_at_k_values:
        # 计算Recall@k
        top_k_indices = np.argsort(pred_probabilities)[::-1][:k]
        top_k_labels = ground_truth_labels[top_k_indices]
        relevant_items = np.sum(ground_truth_labels)
        recall = np.sum(top_k_labels) / relevant_items if relevant_items > 0 else 0
        recall_at_k[k] = recall

    return precision_at_k, recall_at_k

def train_test(data_train, data_test, data_neg, fold, args, device):
    drug_node_train, side_node_train, train_label, drug_node_test, side_node_test, test_label, drug_train, side_train, drug_test, side_test = fold_files(
        data_train, data_test, data_neg, args)
    drug_des, drug_smile, drug_combined = processSmiles.load_drug_smile(args.rawpath + '/drug_description.xlsx')
    side_des = processSmiles.side_description(args.rawpath + '/side_description.xlsx')
    simle_graph = processSmiles.convert2graph(drug_smile)
    trainset = torch.utils.data.TensorDataset(torch.LongTensor(drug_node_train), torch.FloatTensor(drug_train),
                                              torch.FloatTensor(train_label), torch.LongTensor(side_node_train),
                                              torch.FloatTensor(side_train))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(drug_node_test), torch.FloatTensor(drug_test),
                                             torch.FloatTensor(test_label), torch.LongTensor(side_node_test),
                                             torch.FloatTensor(side_test))
    _train = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=1, pin_memory=True, worker_init_fn=_init_fn)
    _test = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True,
                                        num_workers=1, pin_memory=True, worker_init_fn=_init_fn)

    u_side_description = get_embeddings(side_des, "side",device)
    u_embed_description = get_embeddings(drug_combined, "drug", device)

    model = ConvNCF(u_side_description, u_embed_description, 2982, args.embed_dim).to(device)
    Regression_criterion = nn.MSELoss()
    Classification_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[250, 300], gamma=0.2)
    AUC_mn, AUPR_mn, endure_count = 0, 0, 0
    best_test_AUC, best_test_AUPR, best_test_rmse, best_test_mae, best_test_pearson, best_test_spearman, best_test_acc, best_test_epoch = 0, 0, 0, 0, 0, 0, 0, 0
    pred1, pred2, ground_truth, ground_label_truth, ground_drug, ground_side = [], [], [], [], [], []

    model_path = utils.create_model_directory(fold)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    for epoch in range(1, args.epochs + 1):
        # ==================== Training ====================
        if not os.path.exists(model_path):
            train(model, _train, optimizer, Classification_criterion, Regression_criterion, simle_graph, device)
        scheduler.step()
        # if epoch < 100:
        #     continue
        test_results = test(model, _test, simle_graph, device)
        t_i_auc, t_iPR_auc, t_rmse, t_mae, t_pearson, t_spearman, t_acc, t_ground_drug, t_ground_side, t_ground_truth, t_ground_label_truth, t_pred1, t_pred2 = test_results
        # Update best test results
        if AUC_mn + AUPR_mn < t_i_auc + t_iPR_auc:
            AUC_mn = t_i_auc
            AUPR_mn = t_iPR_auc
            best_test_AUC = t_i_auc
            best_test_AUPR = t_iPR_auc
            best_test_rmse = t_rmse
            best_test_mae = t_mae
            best_test_pearson = t_pearson
            best_test_spearman = t_spearman
            best_test_acc = t_acc
            best_test_epoch = epoch
            pred1 = t_pred1
            pred2 = t_pred2
            ground_truth = t_ground_truth
            ground_label_truth = t_ground_label_truth
            ground_drug = t_ground_drug
            ground_side = t_ground_side
        else:
            endure_count += 1

    logger.info(
        f'The best AUC/AUPR: Epoch:{best_test_epoch}\t i_auc:{best_test_AUC:.5f}\t iPR_auc={best_test_AUPR:.5f}\t rmse={best_test_rmse:.5f}\t mae:{best_test_mae:.5f}\t pearson:{best_test_pearson:.5f}\t spearman:{best_test_spearman:.5f}\t acc:{best_test_acc:.5f}\t')

    utils.writePredResultToCsv([item for sublist in ground_drug for item in sublist],[item for sublist in ground_side for item in sublist], ground_label_truth, pred1,ground_truth,pred2, os.path.basename(__file__).split(".")[0], fold)
    if args.save_model:
        torch.save(model.state_dict(),
                   "./save_model/model/" + os.path.basename(__file__).split(".")[0] + str(fold) + ".pt")
    return best_test_AUC, best_test_AUPR, best_test_rmse, best_test_mae, best_test_pearson, best_test_spearman, best_test_acc


def train(model, train_loader, optimizer, lossfunction1, lossfunction2, simle_graph, device):
    model.train()
    sum_loss = 0.0
    for i, data in enumerate(train_loader):
        drug_node_train, drug_train, train_label, side_node_train, side_train = data
        batch_labels = (train_label > 0).float()

        optimizer.zero_grad()

        sum_numpy_u_smiles = [simle_graph[idx.item()] for idx in drug_node_train.flatten()]

        DrugSmilesGraph = myDataset(root='save_model', simle_graph=sum_numpy_u_smiles)
        logits, reconstruction = model(drug_node_train, side_node_train, drug_train, side_train,
                                       DrugSmilesGraph,
                                       device)
        loss1 = lossfunction1(logits, batch_labels.to(device))
        one_label_index = np.nonzero(batch_labels.cpu().numpy())[0]
        loss2 = lossfunction2(reconstruction[one_label_index], train_label[one_label_index].to(device))

        total_loss = loss1 * loss2

        total_loss.backward(retain_graph=True)
        optimizer.step()
        sum_loss += total_loss.item()

    logger.info('sum_loss={:.5f}'.format(sum_loss))
    return 0


def test(model, test_loader, simle_graph, device):
    model.eval()
    ground_drug, ground_side, ground_side_node = [], [], []
    ground_truth, ground_label_truth, pred1, pred2 = [], [], [], []

    with torch.no_grad():
        for drug_node_test, drug_test, test_label, side_node_test, side_test in test_loader:
            drug_node_test, drug_test, test_label, side_node_test, side_test = (
                item.to(device) for item in (drug_node_test, drug_test, test_label, side_node_test, side_test)
            )
            sum_numpy_u_smiles = []
            for i in range(drug_node_test.size()[0]):
                index = drug_node_test[[i]].cpu().numpy()
                sum_numpy_u_smiles.append(simle_graph[index[0].tolist()])
            DrugSmilesGraph = myDataset(root='save_model', simle_graph=sum_numpy_u_smiles)

            test_labels = (test_label.clone().long() > 0).long()

            classification, regression = model(drug_node_test, side_node_test, drug_test, side_test,
                                               DrugSmilesGraph,
                                               device)
            pred1.extend(classification.data.cpu().numpy())
            pred2.extend(regression.data.cpu().numpy())
            ground_truth.extend(test_label.data.cpu().numpy())
            ground_label_truth.extend(test_labels.data.cpu().numpy())

            ground_drug.extend(drug_node_test.data.cpu().numpy())
            ground_side.extend(side_node_test.data.cpu().numpy())
            ground_side_node.extend(side_node_test.data.cpu().numpy())

    pred1, pred2 = np.array(pred1, dtype=np.float32), np.array(pred2, dtype=np.float32)
    ground_truth, ground_label_truth = np.array(ground_truth, dtype=np.float32), np.array(ground_label_truth,
                                                                                          dtype=np.float32)

    iprecision, irecall, ithresholds = metrics.precision_recall_curve(ground_label_truth, pred1, pos_label=1,
                                                                      sample_weight=None)

    iPR_auc = metrics.auc(irecall, iprecision)

    try:
        i_auc = metrics.roc_auc_score(ground_label_truth, pred1)
    except ValueError:
        i_auc = 0
    precision_at_k_values = [100, 500]
    recall_at_k_values = [200, 500]
    pred3 = np.where(pred1 > 0.5, 1, 0)

    one_label_index = np.nonzero(ground_label_truth)
    rmse = sqrt(mean_squared_error(pred2[one_label_index], ground_truth[one_label_index]))
    mae = mean_absolute_error(pred2[one_label_index], ground_truth[one_label_index])
    # 计算 Pearson 相关系数
    pearson_corr, pearson_p_value = pearsonr(ground_truth[one_label_index], pred2[one_label_index])

    # 计算 Spearman 相关系数
    spearman_corr, spearman_p_value = spearmanr(ground_truth[one_label_index], pred2[one_label_index])

    precision_at_k, recall_at_k = calculate_precision_recall_at_k(pred3, ground_label_truth,precision_at_k_values,recall_at_k_values)



    acc = accuracy_score(ground_label_truth, pred3)
    threshold_index = next((i for i, t in enumerate(ithresholds) if t > 0.5), -1)
    recall_at_threshold = irecall[threshold_index]
    precision_at_threshold = iprecision[threshold_index]
    loggerloss.info('recall_at_threshold: %.4f\t precision_at_threshold: %.4f\t', recall_at_threshold,precision_at_threshold)
    loggerloss.info('Precision: ' + ', '.join(f'@{k}: {precision_at_k[k]:.4f}' for k in precision_at_k_values))
    loggerloss.info('Recall: ' + ', '.join(f'@{k}: {recall_at_k[k]:.4f}' for k in recall_at_k_values))
    return i_auc, iPR_auc, rmse, mae, pearson_corr, spearman_corr, acc, ground_drug, ground_side_node, ground_truth, ground_label_truth, pred3, pred2


def ten_fold(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    rawpath = args.rawpath
    gii = open(rawpath + '/drug_side_frequencies.pkl', 'rb')
    drug_side = pickle.load(gii)
    gii.close()

    addition_negative_sample, final_sample = Extract_positive_negative_samples(
        drug_side, addition_negative_number='all')

    X = final_sample[:, 0::]

    data_neg = [
        (addition_negative_sample[i, 0], addition_negative_sample[i, 1], int(float(addition_negative_sample[i, 2]))) for
        i in range(addition_negative_sample.shape[0])]
    data = [(X[i, 0], X[i, 1], int(float(X[i, 2]))) for i in range(X.shape[0])]
    data_x = [(X[i, 0], X[i, 1]) for i in range(X.shape[0])]
    data_y = [int(float(X[i, 2])) for i in range(X.shape[0])]
    kfold = StratifiedKFold(5, random_state=1, shuffle=True)
    total_auc, total_pr_auc, total_rmse, total_mae, total_pearson, total_spearman, total_acc = [], [], [], [], [], [], []
    for fold, (train, test) in enumerate(kfold.split(data_x, data_y)):
        logger.info('==================================fold {} start'.format(fold))
        data = np.array(data)
        auc, pr_auc, rmse, mae, pearson, spearman, acc = train_test(data[train].tolist(), data[test].tolist(), data_neg,
                                                                    fold, args, device)
        total_rmse.append(rmse)
        total_mae.append(mae)
        total_auc.append(auc)
        total_pr_auc.append(pr_auc)
        total_pearson.append(pearson)
        total_spearman.append(spearman)
        total_acc.append(acc)
        logger.info("==================================fold {} end".format(fold))
        sys.stdout.flush()

    # Calculate and log the overall statistics
    logger.info(
        'Overall Statistics:\nTotal_AUC: {:.5f}\nTotal_AUPR: {:.5f}\nTotal_RMSE: {:.5f}\nTotal_MAE: {:.5f}\nTotal_pearson: {:.5f}\nTotal_spearman: {:.5f}\nTotal_acc: {:.5f}'.format(
            np.mean(total_auc), np.mean(total_pr_auc), np.mean(total_rmse),
            np.mean(total_mae), np.mean(total_pearson), np.mean(total_spearman), np.mean(total_acc)
        ))


def main():
    # Training settings

    parser = argparse.ArgumentParser(description='Model')
    parser.add_argument('--epochs', type=int, default=2,
                        metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0005,
                        metavar='FLOAT', help='learning rate')
    parser.add_argument('--embed_dim', type=int, default=128,
                        metavar='N', help='embedding dimension')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        metavar='FLOAT', help='weight decay')
    parser.add_argument('--batch_size', type=int, default=128,
                        metavar='N', help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=256,
                        metavar='N', help='input batch size for testing')
    parser.add_argument('--rawpath', type=str, default='./data',
                        metavar='STRING', help='rawpath')
    parser.add_argument('--save_model', action='store_true', default='True', help='save model')
    args = parser.parse_args()

    print('-------------------- Begin --------------------')
    ten_fold(args)


if __name__ == "__main__":
    main()
