# 最新模型
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import pickle
import argparse
from copy import copy
import time

from math import sqrt
import torch.utils.data
from datetime import datetime
import torch.nn.functional as F
import sys
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GATConv, GCNConv, GINConv, RGCNConv
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from torch.optim.lr_scheduler import MultiStepLR

from util.utils import myDataset
from util import processSmiles, utils
from util import getDrug_KVPLM_embed
from util import InfoLog
from util.TransformerUtil import Embeddings, Encoder_MultipleLayers

path = "./save_model/log/" + os.path.basename(__file__).split(".")[0] + '_' + datetime.now().strftime(
    "%Y%m%d%H%M") + '.log'
logger = InfoLog.get_logger(filename=path, name="1")

pathLoss = "./save_model/loss/" + os.path.basename(__file__).split(".")[0] + '_' + datetime.now().strftime(
    "%Y%m%d%H%M") + '.log'
loggerloss = InfoLog.get_logger(filename=pathLoss, name="2")


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
        self.outDrugDescription = nn.Sequential(
            nn.Linear(self.outbert_dim, self.embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5),
            nn.Dropout(self.dropout1),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.outSideDescription = nn.Sequential(
            nn.Linear(self.outbert_dim, self.embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.embed_dim, momentum=0.5),
            nn.Dropout(self.dropout1),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

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
        self.cnn_interaction = nn.Sequential(
            # batch_size * 1 * 64 * 64
            nn.Conv2d(self.number_map, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            # batch_size * 32 * 32 * 32
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            # batch_size * 32 * 16 * 16
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            # batch_size * 32 * 8 * 8
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            # batch_size * 32 * 4 * 4
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
        )
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

    def forward(self, drug_node_train, side_node_train, drug_train, side_train, DrugSmilesGraph, device):
        x, edge_index, edge_type, batch, smiles = (DrugSmilesGraph.data.x).to(device), (
            DrugSmilesGraph.data.edge_index).to(
            device), (DrugSmilesGraph.data.y).to(device), (DrugSmilesGraph.data.batch).to(
            device), DrugSmilesGraph.data.smiles
        x = F.relu((self.gcn1(x, edge_index)), inplace=True)
        x = F.relu((self.gcn2(x, edge_index)), inplace=True)
        x = global_max_pool(x, batch)  # global max pooling
        drug_x = self.GCNFC(x)
        drug_embed_des, side_embed_des = [], []
        for i in range(drug_node_train.size()[0]):
            index = drug_node_train[[i]].cpu().numpy()
            drug_embed_des.append(self.drug_embed_description[index[0].tolist()])
        for i in range(side_node_train.size()[0]):
            index = side_node_train[[i]].cpu().numpy()
            side_embed_des.append(self.side_embed_description[index[0].tolist()])

        u_embed_des = torch.cat(drug_embed_des, 0)
        s_embed_des = torch.cat(side_embed_des, 0)

        u_embed_description = self.outDrugDescription(u_embed_des)
        s_embed_description = self.outSideDescription(s_embed_des)

        u_embed_rwr = self.outDrugRWR(drug_train.to(device))
        s_embed_rwr = self.outSideRWR(side_train.to(device))
        drugs = [drug_x, u_embed_description, u_embed_rwr]

        sides = [s_embed_description, s_embed_rwr]

        cos_sim_1 = 1 - torch.abs(F.cosine_similarity(s_embed_description, s_embed_rwr, dim=1))
        drug_cosine_similarities_1 = cos_sim_1.sum(dim=0)
        drug_contrastive = drug_cosine_similarities_1

        maps = []
        elementproduct = []
        for i in range(len(drugs)):
            for j in range(len(sides)):
                maps.append(torch.bmm(drugs[i].unsqueeze(2), sides[j].unsqueeze(1)))
                elementproduct.append(torch.mul(drugs[i], sides[j]))

        interaction_map = maps[0].view((-1, 1, self.embed_dim, self.embed_dim))

        elementproduct_map = elementproduct[0].view((side_node_train.size()[0], 1, self.embed_dim))
        for i in range(1, len(maps)):
            interaction = maps[i].view((-1, 1, self.embed_dim, self.embed_dim))
            interaction_map = torch.cat([interaction_map, interaction], dim=1)

            elementproductinter = elementproduct[i].view((side_node_train.size()[0], 1, self.embed_dim))
            elementproduct_map = torch.cat([elementproduct_map, elementproductinter], dim=1)

        sumelementproduct = torch.sum(elementproduct_map, dim=1)
        scalar = self.scalarlayer(sumelementproduct)
        feature_map = self.cnn_interaction(interaction_map)  # output: batch_size * 32 * 1 * 1
        h = feature_map.view((-1, self.channel_size * 4 * 4))
        total = self.total_layer(torch.cat((scalar, h), dim=1))
        classification = self.classifier(total)
        regression = self.con_layer(total)
        return classification.squeeze(), regression.squeeze(), drug_contrastive


def _init_fn(worker_id):
    # np.random.seed(int(1024) + worker_id)
    np.random.seed(int(1024))


def train_test(data_train, data_test, data_neg, fold, args, device):
    drug_node_train, side_node_train, train_label, drug_node_test, side_node_test, test_label, drug_train, side_train, drug_test, side_test = utils.fold_files(
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

    u_side_description = getDrug_KVPLM_embed.get_Drug_smiles_embed(side_des, device)
    u_embed_description = getDrug_KVPLM_embed.get_Drug_smiles_embed(drug_combined, device)
    side_effect_label = utils.readdruginfo(args.rawpath)

    model = ConvNCF(u_side_description, u_embed_description, 2982, args.embed_dim).to(device)
    Regression_criterion = nn.MSELoss()
    Classification_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[250, 300], gamma=0.2)
    AUC_mn, AUPR_mn, endure_count = 0, 0, 0
    best_test_AUC, best_test_AUPR, best_test_rmse, best_test_mae, best_test_pearson, best_test_spearman, best_test_acc, best_test_epoch = 0, 0, 0, 0, 0, 0, 0, 0
    pred1, pred2, ground_truth, ground_label_truth, ground_drug, ground_side = [], [], [], [], [], []

    start = time.time()
    model_path = utils.create_model_directory(fold)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    for epoch in range(1, args.epochs + 1):
        # ==================== Training ====================
        if not os.path.exists(model_path):
            train(model, _train, optimizer, Classification_criterion, Regression_criterion, simle_graph, device)
        scheduler.step()

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

    utils.writePredResultToCsv([item for sublist in ground_drug for item in sublist],
                               [item for sublist in ground_side for item in sublist], ground_label_truth, pred1,
                               ground_truth,
                               pred2, os.path.basename(__file__).split(".")[0], fold)
    if args.save_model:
        torch.save(model.state_dict(),
                   "./save_model/model/" + os.path.basename(__file__).split(".")[0] + str(fold) + ".pt")
    return best_test_AUC, best_test_AUPR, best_test_rmse, best_test_mae, best_test_pearson, best_test_spearman, best_test_acc


def train(model, train_loader, optimizer, lossfunction1, lossfunction2, simle_graph, device):
    model.train()
    sum_loss = 0.0
    sumitem = 0
    for i, data in enumerate(train_loader, 0):
        drug_node_train, drug_train, train_label, side_node_train, side_train = data
        batch_labels = train_label.clone().float()
        for k in range(train_label.data.size()[0]):
            if train_label.data[k] > 0:
                batch_labels.data[k] = 1
        optimizer.zero_grad()

        one_label_index = np.nonzero(batch_labels.data.numpy())
        sum_numpy_u_smiles = []
        for i in range(drug_node_train.size()[0]):
            index = drug_node_train[[i]].cpu().numpy()
            sum_numpy_u_smiles.append(simle_graph[index[0].tolist()])
        DrugSmilesGraph = myDataset(root='save_model', simle_graph=sum_numpy_u_smiles)
        logits, reconstruction, drug_contrastive = model(drug_node_train, side_node_train, drug_train, side_train,
                                                         DrugSmilesGraph,
                                                         device)
        loss1 = lossfunction1(logits, batch_labels.to(device))
        loss2 = lossfunction2(reconstruction[one_label_index], train_label[one_label_index].to(device))

        total_loss = loss1 * loss2
        sumitem += 1
        if (sumitem % 100 == 0):
            loggerloss.info('loss1:{:.5f}\t loss2:{:.5f}\t'.format(loss1, loss2))
        total_loss.backward(retain_graph=True)
        optimizer.step()
        sum_loss += total_loss.item()
        # logger.info('batch:{}\t loss={:.5f}\t avg_loss={:.3f}'.format(i, total_loss, avg_loss))
    logger.info('sum_loss={:.5f}'.format(sum_loss))
    return 0


def test(model, test_loader, simle_graph, device):
    model.eval()
    ground_drug, ground_side, ground_side_node = [], [], []
    ground_truth, ground_label_truth, pred1, pred2 = [], [], [], []

    with torch.no_grad():
        for drug_node_test, drug_test, test_label, side_node_test, side_test in test_loader:
            drug_node_test, drug_test, test_label, side_node_test, side_test = (
                drug_node_test.to(device),
                drug_test.to(device),
                test_label.to(device),
                side_node_test.to(device),
                side_test.to(device),
            )
            sum_numpy_u_smiles = []
            for i in range(drug_node_test.size()[0]):
                index = drug_node_test[[i]].cpu().numpy()
                sum_numpy_u_smiles.append(simle_graph[index[0].tolist()])
            DrugSmilesGraph = myDataset(root='save_model', simle_graph=sum_numpy_u_smiles)

            test_labels = test_label.clone().long()
            test_labels[test_label.data > 0] = 1

            classification, regression, drug_contrastive = model(drug_node_test, side_node_test, drug_test, side_test,
                                                                 DrugSmilesGraph,
                                                                 device)
            pred1.append(list(classification.data.cpu().numpy()))
            pred2.append(list(regression.data.cpu().numpy()))
            ground_truth.append(list(test_label.data.cpu().numpy()))
            ground_label_truth.append(list(test_labels.data.cpu().numpy()))

            ground_drug.append(list(drug_node_test.data.cpu().numpy()))
            ground_side.append(list(side_node_test.data.cpu().numpy()))
            ground_side_node.append(list(side_node_test.data.cpu().numpy()))

    pred1 = np.array(sum(pred1, []), dtype=np.float32)
    pred2 = np.array(sum(pred2, []), dtype=np.float32)
    ground_truth = np.array(sum(ground_truth, []), dtype=np.float32)
    ground_label_truth = np.array(sum(ground_label_truth, []), dtype=np.float32)

    iprecision, irecall, ithresholds = metrics.precision_recall_curve(ground_label_truth, pred1, pos_label=1,
                                                                      sample_weight=None)
    average_precision = metrics.average_precision_score(ground_label_truth, pred1)
    iPR_auc = metrics.auc(irecall, iprecision)

    try:
        i_auc = metrics.roc_auc_score(ground_label_truth, pred1)
    except ValueError:
        i_auc = 0

    one_label_index = np.nonzero(ground_label_truth)
    rmse = sqrt(mean_squared_error(pred2[one_label_index], ground_truth[one_label_index]))
    mae = mean_absolute_error(pred2[one_label_index], ground_truth[one_label_index])
    pearson = utils.pearson(ground_truth[one_label_index], pred2[one_label_index])
    spearman = utils.spearman(ground_truth[one_label_index], pred2[one_label_index])

    pred3 = copy(pred1)
    pred3[pred1 > 0.5] = 1
    pred3[pred1 <= 0.5] = 0

    acc = accuracy_score(ground_label_truth, pred3)

    return i_auc, iPR_auc, rmse, mae, pearson, spearman, acc, ground_drug, ground_side_node, ground_truth, ground_label_truth, pred3, pred2


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

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    rawpath = args.rawpath
    gii = open(rawpath + '/drug_side_frequencies.pkl', 'rb')
    drug_side = pickle.load(gii)
    gii.close()

    addition_negative_sample, final_sample = utils.Extract_positive_negative_samples(
        drug_side, addition_negative_number='all')

    X = final_sample[:, 0::]

    data_neg = [
        (addition_negative_sample[i, 0], addition_negative_sample[i, 1], int(float(addition_negative_sample[i, 2]))) for
        i in range(addition_negative_sample.shape[0])]
    data = [(X[i, 0], X[i, 1], int(float(X[i, 2]))) for i in range(X.shape[0])]
    data_x = [(X[i, 0], X[i, 1]) for i in range(X.shape[0])]
    data_y = [int(float(X[i, 2])) for i in range(X.shape[0])]
    kfold = StratifiedKFold(10, random_state=1, shuffle=True)
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
    parser.add_argument('--epochs', type=int, default=400,
                        metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0005,
                        metavar='FLOAT', help='learning rate')
    parser.add_argument('--embed_dim', type=int, default=128,
                        metavar='N', help='embedding dimension')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        metavar='FLOAT', help='weight decay')
    parser.add_argument('--N', type=int, default=30000,
                        metavar='N', help='L0 parameter')
    parser.add_argument('--droprate', type=float, default=0.5,
                        metavar='FLOAT', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        metavar='N', help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=256,
                        metavar='N', help='input batch size for testing')
    parser.add_argument('--dataset', type=str, default='hh',
                        metavar='STRING', help='dataset')
    parser.add_argument('--rawpath', type=str, default='./data',
                        metavar='STRING', help='rawpath')
    parser.add_argument('--save_model', action='store_true', default='True', help='save model')
    args = parser.parse_args()

    print('-------------------- Begin --------------------')
    ten_fold(args)


if __name__ == "__main__":
    main()
