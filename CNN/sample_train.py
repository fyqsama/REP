import numpy as np
import torch
import utils
import time
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import arch_sampled

from torch.autograd import Variable
from model import NetworkCIFAR as Network
from loss import trades_loss, madry_loss

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/home/yuqi/data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--epochs', type=int, default=5, help='num of training epochs')
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation')
parser.add_argument('--num_steps', type=int, default=7, help='perturb number of steps')
parser.add_argument('--step_size', type=float, default=0.01, help='perturb step size')
parser.add_argument('--beta', type=float, default=6.0, help='regularization in TRADES')
parser.add_argument('--adv_loss', type=str, default='pgd', help='experiment name')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()


CIFAR_CLASSES = 10


def main():
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    print('gpu device: ', args.gpu)
    print("args: ", args)

    train_transform, valid_transform = utils._data_transforms_train(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    total_cost = 0
    for arch_num in range(0, 24):
        arch = 'arch' + str(arch_num)
        genotype = eval("arch_sampled.%s" % arch)
        model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
        model = model.cuda()

        print("param size: ", utils.count_parameters_in_MB(model), 'MB')

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.learning_rate_min)

        best_acc = 0.0
        for epoch in range(args.epochs):
            lr = scheduler.get_last_lr()
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

            time1 = time.time()
            train_acc, train_obj = train(train_queue, model, criterion, optimizer)
            print('epoch: ', epoch, 'lr: ', lr, 'train_acc: ', train_acc)
            time2 = time.time()
            total_cost += time2 - time1
            print('total_cost: ', total_cost)

            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(model.state_dict(), '/home/yuqi/models/REP_DARTS_' + str(arch_num) + '.pt')
            print('epoch: ', epoch, 'valid_acc: ', valid_acc, 'best_acc: ', best_acc)
            scheduler.step()


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = Variable(input).cuda(non_blocking=True)
        target = Variable(target).cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, _ = model(input)
        if args.adv_loss == 'pgd':
            loss = madry_loss(
                model,
                input,
                target,
                optimizer,
                step_size=args.step_size,
                epsilon=args.epsilon,
                perturb_steps=args.num_steps)
        elif args.adv_loss == 'trades':
            loss = trades_loss(model,
                               input,
                               target,
                               optimizer,
                               step_size=args.step_size,
                               epsilon=args.epsilon,
                               perturb_steps=args.num_steps,
                               beta=args.beta,
                               distance='l_inf')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input, requires_grad=False).cuda(non_blocking=True)
            target = Variable(target, requires_grad=False).cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
