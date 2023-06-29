import os
import os.path as osp
import sys
import glob
import numpy as np
import scipy.sparse as sp
import scipy.linalg as spl
import torch
import random
from torch.nn import Parameter

import test
import utils
import logging
import pickle
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import cat

from torch.autograd import Variable
from model import NetworkGNN as Network
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull, Reddit
from utils import gen_uniform_60_20_20_split, save_load_split
from sklearn.model_selection import StratifiedKFold
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from logging_util import init_logger

parser = argparse.ArgumentParser("sane-test")
parser.add_argument('--data', type=str, default='CiteSeer', help='location of the data corpus')
parser.add_argument('--record_time', action='store_true', default=False, help='used for run_with_record_time func')
parser.add_argument('--num_layers', type=int, default=3, help='num of GNN layers in SANE')
parser.add_argument('--hidden_size', type=int, default=64, help='embedding size in NN')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=3000, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--epsilon', type=float, default=0.0, help='the explore rate in the gradient descent process')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--with_conv_linear', type=bool, default=False, help=' in NAMixOp with linear op')
parser.add_argument('--fix_last', type=bool, default=False, help='fix last layer in design architectures.')
parser.add_argument('--with_linear', action='store_true', default=False, help='whether to use linear in NaOp')
parser.add_argument('--with_layernorm', action='store_true', default=False, help='whether to use layer norm')

test_args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def attack(adj, n_perturbations=622, dim=32, window_size=5,
           attack_type="remove", min_span_tree=False, n_candidates=None, seed=None, **kwargs):

    assert attack_type in ["remove", "add", "add_by_remove"], \
        "attack_type can only be `remove` or `add`"

    if attack_type == "remove":
        if min_span_tree:
            candidates = generate_candidates_removal_minimum_spanning_tree(adj)
        else:
            candidates = generate_candidates_removal(adj, seed)

    elif attack_type == "add" or attack_type == "add_by_remove":

        assert n_candidates, "please specify the value of `n_candidates`, " \
                             + "i.e. how many candiate you want to genereate for addition"
        candidates = generate_candidates_addition(adj, n_candidates, seed)

    n_nodes = adj.shape[0]

    if attack_type == "add_by_remove":
        candidates_add = candidates
        adj_add = flip_candidates(adj, candidates_add)
        vals_org_add, vecs_org_add = spl.eigh(adj_add.toarray(), np.diag(adj_add.sum(1).A1))
        flip_indicator = 1 - 2 * adj_add[candidates[:, 0], candidates[:, 1]].A1

        loss_est = estimate_loss_with_delta_eigenvals(candidates_add, flip_indicator,
                                                      vals_org_add, vecs_org_add, n_nodes, dim, window_size)

        loss_argsort = loss_est.argsort()
        top_flips = candidates_add[loss_argsort[:n_perturbations]]

    else:
        # vector indicating whether we are adding an edge (+1) or removing an edge (-1)
        delta_w = 1 - 2 * adj.toarray()[candidates[:, 0], candidates[:, 1]].ravel()

        # generalized eigenvalues/eigenvectors
        deg_matrix = np.diag(adj.sum(1).A1)
        vals_org, vecs_org = spl.eigh(adj.toarray(), deg_matrix)

        loss_for_candidates = estimate_loss_with_delta_eigenvals(candidates, delta_w, vals_org, vecs_org, n_nodes, dim,
                                                                 window_size)
        top_flips = candidates[loss_for_candidates.argsort()[-n_perturbations:]]

    assert len(top_flips) == n_perturbations

    modified_adj = flip_candidates(adj, top_flips)
    return modified_adj

def generate_candidates_removal(adj, seed=None):
    n_nodes = adj.shape[0]
    if seed is not None:
        np.random.seed(seed)
    deg = np.where(adj.sum(1).ravel() == 1)[0]
    hiddeen = np.column_stack(
        (np.arange(n_nodes), np.fromiter(map(np.random.choice, adj.tolil().rows), dtype=np.int32)))

    adj_hidden = edges_to_sparse(hiddeen, adj.shape[0])
    adj_hidden = adj_hidden.maximum(adj_hidden.T)

    adj_keep = adj - adj_hidden

    candidates = np.column_stack((sp.triu(adj_keep).nonzero()))

    candidates = candidates[np.logical_not(np.in1d(candidates[:, 0], deg) | np.in1d(candidates[:, 1], deg))]

    return candidates

def edges_to_sparse(edges, num_nodes, weights=None):
    if weights is None:
        weights = np.ones(edges.shape[0])

    return sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes)).tocsr()

def generate_candidates_removal_minimum_spanning_tree(adj):
    mst = sp.csgraph.minimum_spanning_tree(adj)
    mst = mst.maximum(mst.T)
    adj_sample = adj - mst
    candidates = np.column_stack(sp.triu(adj_sample, 1).nonzero())

    return candidates

def generate_candidates_addition(adj, n_candidates, seed=None):
    if seed is not None:
        np.random.seed(seed)

    num_nodes = adj.shape[0]

    candidates = np.random.randint(0, num_nodes, [n_candidates * 5, 2])
    candidates = candidates[candidates[:, 0] < candidates[:, 1]]
    candidates = candidates[adj[candidates[:, 0], candidates[:, 1]].A1 == 0]
    candidates = np.array(list(set(map(tuple, candidates))))
    candidates = candidates[:n_candidates]

    assert len(candidates) == n_candidates

    return candidates

def flip_candidates(adj, candidates):
    adj_flipped = adj.copy().tolil()
    adj_flipped[candidates[:, 0], candidates[:, 1]] = 1 - adj.toarray()[candidates[:, 0], candidates[:, 1]]
    adj_flipped[candidates[:, 1], candidates[:, 0]] = 1 - adj.toarray()[candidates[:, 1], candidates[:, 0]]
    adj_flipped = adj_flipped.tocsr()
    adj_flipped.eliminate_zeros()

    return adj_flipped

def estimate_loss_with_delta_eigenvals(candidates, flip_indicator, vals_org, vecs_org, n_nodes, dim, window_size):
    loss_est = np.zeros(len(candidates))
    for x in range(len(candidates)):
        i, j = candidates[x]
        vals_est = vals_org + flip_indicator[x] * (
                2 * vecs_org[i] * vecs_org[j] - vals_org * (vecs_org[i] ** 2 + vecs_org[j] ** 2))

        vals_sum_powers = sum_of_powers(vals_est, window_size)

        loss_ij = np.sqrt(np.sum(np.sort(vals_sum_powers ** 2)[:n_nodes - dim]))
        loss_est[x] = loss_ij

    return loss_est

def sum_of_powers(x, power):
    n = x.shape[0]
    sum_powers = np.zeros((power, n))

    for i, i_power in enumerate(range(1, power + 1)):
        sum_powers[i] = np.power(x, i_power)

    return sum_powers.sum(0)

def generate_adj(data):
    adj = np.zeros((data.num_nodes, data.num_nodes))
    for i, j in zip(data.edge_index[0], data.edge_index[1]):
        adj[i][j] = 1
        adj[j][i] = 1
    adj = torch.from_numpy(adj)
    return adj

def generate_edge_index(adj):
    adj = adj.detach().cpu().numpy()
    adj = sp.coo_matrix(adj)
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    adj = torch.LongTensor(indices).cuda()
    return adj

def generate_indices(data):
    train_indices = []
    val_indices = []
    test_indices = []
    for i in range(len(data.train_mask)):
        if data.train_mask[i] == True:
            train_indices.append(i)
    for i in range(len(data.val_mask)):
        if data.val_mask[i] == True:
            val_indices.append(i)
    for i in range(len(data.test_mask)):
        if data.test_mask[i] == True:
            test_indices.append(i)
    return train_indices, val_indices, test_indices

def infer(data, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    logits = F.log_softmax(model(data.to(device)), dim=-1)

  input = logits[data.test_mask].to(device)
  target = data.y[data.test_mask].to(device)

  #logits, _ = model(input)
  loss = criterion(input, target)

  pred = logits[data.test_mask].max(1)[1]
  acc = (pred == target).sum().item() / data.test_mask.sum().item()

  prec1, prec5 = utils.accuracy(input, target, topk=(1, 3))
  n = input.size(0)
  objs.update(loss.data.item(), n)
  top1.update(prec1.data.item(), n)
  top5.update(prec5.data.item(), n)

  return top1.avg, objs.avg

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    # np.random.seed(test_args.seed)
    torch.cuda.set_device(test_args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(test_args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(test_args.seed)

    if test_args.data == 'Cora':
        dataset = Planetoid('/home/yuqi/data/', 'Cora')
    elif test_args.data == 'CiteSeer':
        dataset = Planetoid('/home/yuqi/data/', 'CiteSeer')

    num_features = dataset.num_features
    num_classes = dataset.num_classes

    raw_dir = dataset.raw_dir
    data = dataset[0]

    data = save_load_split(data, raw_dir, 1, gen_uniform_60_20_20_split)
    edge_index, _ = add_self_loops(data.edge_index)
    data.edge_index = edge_index

    hidden_size = test_args.hidden_size
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    genotype = 'gat_cos||gin||gat_generalized_linear||skip||skip||skip||l_concat'
    model = Network(genotype, criterion, num_features, num_classes, hidden_size,
                    num_layers=test_args.num_layers, is_mlp=False, args=test_args)
    model = model.cuda()
    utils.load(model, '/home/yuqi/citeseeradv/rep.pt')

    row = data.edge_index[0]
    col = data.edge_index[1]
    edge = np.array([])
    for i in range(len(row)):
        edge = np.append(edge, [1])
    ori_adj = sp.coo_matrix((edge, (row.cpu().numpy(), col.cpu().numpy())), shape=(3327, 3327))
    modified_adj = attack(ori_adj)
    modified_adj = torch.from_numpy(modified_adj.toarray())
    data.edge_index = generate_edge_index(modified_adj)
    test_acc, test_obj = infer(data, model, criterion)
    print('test_acc=', test_acc)


if __name__ == '__main__':
    main()
