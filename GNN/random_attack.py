import os
import os.path as osp
import sys
import glob
import numpy as np
import scipy.sparse as sp
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
from sklearn.model_selection import StratifiedKFold
from utils import gen_uniform_60_20_20_split, save_load_split
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import add_self_loops
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

def perturb_adj(adj, n_perturbations, type='add'):
    modified_adj = adj.tolil()

    type = type.lower()
    assert type in ['add', 'remove', 'flip']

    if type == 'flip':
        # sample edges to flip
        edges = random_sample_edges(adj, n_perturbations, exclude=set())
        for n1, n2 in edges:
            modified_adj[n1, n2] = 1 - modified_adj[n1, n2]
            modified_adj[n2, n1] = 1 - modified_adj[n2, n1]

    if type == 'add':
        # sample edges to add
        nonzero = set(zip(*adj.nonzero()))
        edges = random_sample_edges(adj, n_perturbations, exclude=nonzero)
        for n1, n2 in edges:
            modified_adj[n1, n2] = 1
            modified_adj[n2, n1] = 1

    if type == 'remove':
        # sample edges to remove
        nonzero = np.array(sp.triu(adj, k=1).nonzero()).T
        indices = np.random.permutation(nonzero)[: n_perturbations].T
        modified_adj[indices[0], indices[1]] = 0
        modified_adj[indices[1], indices[0]] = 0

    return modified_adj

def random_sample_edges(adj, n, exclude):
    itr = sample_forever(adj, exclude=exclude)
    return [next(itr) for _ in range(n)]

def sample_forever(adj, exclude):
    while True:
        # t = tuple(np.random.randint(0, adj.shape[0], 2))
        # t = tuple(random.sample(range(0, adj.shape[0]), 2))
        t = tuple(np.random.choice(adj.shape[0], 2, replace=False))
        if t not in exclude:
            yield t
            exclude.add(t)
            exclude.add((t[1], t[0]))

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
    modified_adj = perturb_adj(ori_adj, 622)
    modified_adj = torch.from_numpy(modified_adj.toarray())
    data.edge_index = generate_edge_index(modified_adj)
    test_acc, test_obj = infer(data, model, criterion)
    print('test_acc=', test_acc)


if __name__ == '__main__':
    main()
