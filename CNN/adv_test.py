import argparse
import torch
import torchvision
import utils
from torch.autograd import Variable
from torchvision import transforms
import genotypes
import torch.backends.cudnn as cudnn
from model_test import NetworkCIFAR
import torchattacks


parser = argparse.ArgumentParser(description='PyTorch CIFAR Clean Evaluation')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--test-batch-size', type=int, default=150, metavar='N', help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=True, help='disables CUDA training')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--data_type', type=str, default='cifar10', help='which dataset to use')
args = parser.parse_args()


torch.cuda.set_device(args.gpu)
cudnn.benchmark = True


if args.data_type == 'cifar10':
    transform_list = [transforms.ToTensor()]
    transform_test = transforms.Compose(transform_list)
    testset = torchvision.datasets.CIFAR10(root='/home/yuqi/data', train=False, download=True, transform=transform_test)
elif args.data_type == 'cifar100':
    transform_list = [transforms.ToTensor()]
    transform_test = transforms.Compose(transform_list)
    testset = torchvision.datasets.CIFAR100(root='/home/yuqi/data', train=False, download=True, transform=transform_test)

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)


def cal_acc(model, X, y):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    return err


def eval_std_acc(model, test_loader):
    model.eval()
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural = cal_acc(model, X, y)
        natural_err_total += err_natural
        print('batch err: ', natural_err_total)

    print('natural_err_total: ', natural_err_total)
    return natural_err_total


def eval_adv_acc(model, test_loader, type):
    model.eval()
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        if type == 'FGSM':
            attack = torchattacks.FGSM(model, eps=8/255)
        if type == 'PGD20':
            attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20)
        if type == 'PGD100':
            attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=100)
        if type == 'CW':
            attack = torchattacks.CW(model, c=0.5, steps=100)
        if type == 'APGD':
            attack = torchattacks.APGD(model, eps=8/255, steps=20)
        if type == 'AA':
            attack = torchattacks.AutoAttack(model, eps=8/255)
        adv_images = attack(data, target)
        X, y = Variable(adv_images, requires_grad=True), Variable(target)
        err_natural = cal_acc(model, X, y)
        natural_err_total += err_natural
        print('batch err: ', natural_err_total)

    print('natural_err_total: ', natural_err_total)
    return natural_err_total


def eval_diff_strength(model, test_loader, steps):
    model.eval()
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=steps)
        adv_images = attack(data, target)
        X, y = Variable(adv_images, requires_grad=True), Variable(target)
        err_natural = cal_acc(model, X, y)
        natural_err_total += err_natural
        #print('batch err: ', natural_err_total)

    print('natural_err_total: ', natural_err_total)
    return natural_err_total


def main():
    if args.data_type == 'cifar100':
        CIFAR_CLASSES = 100
    elif args.data_type == 'cifar10':
        CIFAR_CLASSES = 10


    genotype = genotypes.REP_DARTS
    model = NetworkCIFAR(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    print("param size: ", utils.count_parameters_in_MB(model), 'MB')
    model.drop_path_prob = args.drop_path_prob

    model.load_state_dict(torch.load('/home/yuqi/REP_R2/rep_darts_c10_adv/model.pt', map_location=torch.device('cuda:' + str(args.gpu))))
    model = model.cuda()
    eval_std_acc(model, test_loader)
    eval_adv_acc(model, test_loader, 'FGSM')
    eval_adv_acc(model, test_loader, 'PGD20')
    eval_adv_acc(model, test_loader, 'APGD')
    eval_adv_acc(model, test_loader, 'CW')


if __name__ == '__main__':
    main()