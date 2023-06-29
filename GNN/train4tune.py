import os
import os.path as osp
import sys
import time
import glob
import pickle
import numpy as np
import scipy.sparse as sp
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch import cat
from sklearn.metrics import f1_score

from torch.autograd import Variable
import genotypes
from model import NetworkGNN as Network
from utils import gen_uniform_60_20_20_split, save_load_split
#from dice import attack, generate_adj, generate_indices, generate_edge_index
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull, Reddit, PPI
from sklearn.model_selection import StratifiedKFold
from torch_geometric.utils import add_self_loops
from logging_util import init_logger
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

parser = argparse.ArgumentParser("sane-train")
parser.add_argument('--data', type=str, default='Cora', help='location of the data corpus')
parser.add_argument('--record_time', action='store_true', default=False, help='used for run_with_record_time func')
parser.add_argument('--num_layers', type=int, default=3, help='num of GNN layers in SANE')
parser.add_argument('--hidden_size', type=int, default=64, help='embedding size in NN')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.25, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
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

train_args = parser.parse_args()

def main():
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #np.random.seed(train_args.seed)
    torch.cuda.set_device(train_args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(train_args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(train_args.seed)

    if train_args.data == 'Cora':
        dataset = Planetoid('/home/yuqi/data/', 'Cora')
    elif train_args.data == 'CiteSeer':
        dataset = Planetoid('/home/yuqi/data/', 'CiteSeer')

    genotype = 'gat_cos||gin||gat_generalized_linear||skip||skip||skip||l_concat'
    hidden_size = train_args.hidden_size

    raw_dir = dataset.raw_dir
    data = dataset[0]
    data = save_load_split(data, raw_dir, 1, gen_uniform_60_20_20_split)
    edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
    data.edge_index = edge_index
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    '''
    row = data.edge_index[0]
    col = data.edge_index[1]
    edge = np.array([])
    for i in range(len(row)):
        edge = np.append(edge, [1])
    ori_adj = sp.coo_matrix((edge, (row.cpu().numpy(), col.cpu().numpy())), shape=(2708, 2708))
    labels = data.y.cpu().numpy()
    modified_adj = attack(ori_adj, labels, 326)
    modified_adj = torch.from_numpy(modified_adj.toarray())
    data.edge_index = generate_edge_index(modified_adj)
    '''

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(genotype, criterion, num_features, num_classes, hidden_size,
                    num_layers=train_args.num_layers, is_mlp=False, args=train_args)
    model = model.cuda()

    print("genotype = {}, param size = {}MB, args = {}".format(genotype, utils.count_parameters_in_MB(model), train_args.__dict__))


    optimizer = torch.optim.SGD(
        model.parameters(),
        train_args.learning_rate,
        momentum=train_args.momentum,
        weight_decay=train_args.weight_decay)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(train_args.epochs))

    val_res = 0
    best_val_acc = best_test_acc = 0
    training_cost = 0
    for epoch in range(train_args.epochs):
        t1 = time.time()
        train_acc, train_obj = train(train_args.data, data, model, criterion, optimizer)
        scheduler.step()
        t2 = time.time()
        training_cost += t2 - t1

        valid_acc, valid_obj = infer(train_args.data, data, model, criterion)
        test_acc, test_obj = infer(train_args.data, data, model, criterion, test=True)

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_test_acc = test_acc
            utils.save(model, '/home/yuqi/coraadv/rep.pt')

        if epoch % 1 == 0:
            print('epoch={}, lr={}, train_obj={}, train_acc={}, valid_acc={}, test_acc={}'.format(epoch, scheduler.get_lr()[0], train_obj, train_acc, best_val_acc, best_test_acc))

    print('best validation acc: {}, best test acc: {}'.format(best_val_acc, best_test_acc))
    print('training cost: ', training_cost)

def train(dataset_name, data, model, criterion, optimizer):
    return train_trans(data, model, criterion, optimizer)

def infer(dataset_name, data, model, criterion, test=False):
    return infer_trans(data, model, criterion, test=test)

def train_trans(data, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    model.train()
    target = data.y[data.train_mask].to(device)

    optimizer.zero_grad()
    logits = model(data.to(device))

    input = logits[data.train_mask].to(device)

    loss = criterion(input, target)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), train_args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(input, target, topk=(1, 3))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    return top1.avg, objs.avg

def infer_trans(data, model, criterion, test=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        logits = model(data.to(device))
    if test:
        mask = data.test_mask
    else:
        mask = data.val_mask
    input = logits[mask].to(device)
    target = data.y[mask].to(device)
    loss = criterion(input, target)

    prec1, prec5 = utils.accuracy(input, target, topk=(1, 3))
    n = data.val_mask.sum().item()
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    return top1.avg, objs.avg

if __name__ == '__main__':
    main()


