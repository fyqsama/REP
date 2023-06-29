import os
import os.path as osp
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import cat
import pickle
from sklearn.metrics import f1_score

from torch.autograd import Variable
from model_search import Network
from architect import Architect
from utils import gen_uniform_60_20_20_split, save_load_split
from torch_geometric.data import DataLoader

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull, Reddit, PPI
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.utils import add_self_loops
from sklearn.model_selection import StratifiedKFold
from logging_util import init_logger

parser = argparse.ArgumentParser("sane-train-search")
parser.add_argument('--data', type=str, default='Cora', help='location of the data corpus')
parser.add_argument('--record_time', action='store_true', default=False, help='used for run_with_record_time func')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--epsilon', type=float, default=0.0, help='the explore rate in the gradient descent process')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--transductive', action='store_true', help='use transductive settings in train_search.')
parser.add_argument('--with_conv_linear', type=bool, default=False, help=' in NAMixOp with linear op')
parser.add_argument('--fix_last', type=bool, default=False, help='fix last layer in design architectures.')

args = parser.parse_args()

def main():
    global device
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    '''args.save = 'logs/search-{}'.format(args.save)
    if not os.path.exists(args.save):
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_filename = os.path.join(args.save, 'log.txt')
    init_logger('', log_filename, logging.INFO, False)
    print('*************log_filename=%s************' % log_filename)'''

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    print("args = %s", args.__dict__)

    if args.data == 'Cora':
        dataset = Planetoid('/home/yuqi/data/', 'Cora')
    elif args.data == 'CiteSeer':
        dataset = Planetoid('/home/yuqi/data/', 'CiteSeer')

    raw_dir = dataset.raw_dir
    data = dataset[0]
    data = save_load_split(data, raw_dir, 1, gen_uniform_60_20_20_split)

    edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
    data.edge_index = edge_index
    hidden_size = 32

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(criterion, dataset.num_features, dataset.num_classes, hidden_size, epsilon=args.epsilon, args=args)

    model = model.cuda()
    print("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)# send model to compute validation loss
    # test_acc_with_time = []
    # cur_t = 0
    search_cost = 0
    for epoch in range(args.epochs):
        t1 = time.time()
        lr = scheduler.get_lr()[0]
        if epoch % 1 == 0:
            print('epoch {} lr {}'.format(epoch, lr))
            genotype = model.genotype()
            print('genotype = ', genotype)


        train_acc, train_obj = train(args.data, data, model, architect, criterion, optimizer, lr)
        scheduler.step()
        t2 = time.time()
        search_cost += (t2 - t1)

        valid_acc, valid_obj = infer(args.data, data, model, criterion)
        test_acc,  test_obj = infer(args.data, data, model, criterion, test=True)

        if epoch % 1 == 0:
            #logging.info('epoch=%s, train_acc=%f, valid_acc=%f, test_acc=%f, explore_num=%s', epoch, train_acc, valid_acc,test_acc, model.explore_num)
            print('epoch={}, train_acc={:.04f}, valid_acc={:.04f}, test_acc={:.04f},explore_num={}'.format(epoch, train_acc, valid_acc, test_acc, model.explore_num))
        #utils.save(model, '/home/yuqi/weights.pt')
    print('The search process costs {:.02f} s', format(search_cost))
    return genotype

def train(dataset_name, data, model, architect, criterion, optimizer, lr):
    return train_trans(data, model, architect, criterion, optimizer, lr)

def infer(dataset_name, data, model, criterion, test=False):
    return infer_trans(data, model, criterion, test=test)

def train_trans(data, model, architect, criterion, optimizer, lr):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    model.train()
    target = Variable(data.y[data.train_mask], requires_grad=False).to(device)


    #architecture send input or send logits, which are important for computation in architecture
    architect.step(data.to(device), lr, optimizer, unrolled=args.unrolled)

    #train loss
    logits = model(data.to(device))
    input = logits[data.train_mask].to(device)

    optimizer.zero_grad()
    loss = criterion(input, target)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
        input = logits[data.test_mask].to(device)
        target = data.y[data.test_mask].to(device)
        loss = criterion(input, target)
        print('test_loss:', loss.item())
    else:
        input = logits[data.val_mask].to(device)
        target = data.y[data.val_mask].to(device)
        loss = criterion(input, target)
        print('valid_loss:', loss.item())
    prec1, prec5 = utils.accuracy(input, target, topk=(1, 3))
    n = data.val_mask.sum().item()
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    return top1.avg, objs.avg

def run_by_seed():
    res = []
    for i in range(1):
        print('searched {}-th for {}...'.format(i+1, args.data))
        seed = np.random.randint(0, 10000)
        args.seed = seed
        genotype = main()
        res.append('seed={},genotype={}'.format(seed, genotype))


if __name__ == '__main__':
    run_by_seed()

