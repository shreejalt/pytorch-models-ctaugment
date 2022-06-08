import argparse
import gc
import os
import random
import shutil
import time
import warnings
import json
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from sklearn.metrics import confusion_matrix, classification_report
import torchvision.transforms as transforms
import torchvision.models as models
from ctaugment import CTUpdater, CTAugment
from data import DataList, MyDataset
from randaugment import RandAugment
from torch.optim.lr_scheduler import LambdaLR
import math
from prettytable import PrettyTable
from torch.utils.data import WeightedRandomSampler


model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description='PyTorch Training with support of CTAugment Dataset')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--use-weighted', action='store_true', help='Use WeightedRandomSampler')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-nc', '--num-classes', default=10, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('--data-percent', default=1.0, type=float, metavar='N',
                    help='percentage of training data to take for training')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--output-dir', default='outputs', type=str, metavar='V',
                    help='directory to store output weights and logs')
parser.add_argument('--size', default=64, type=int, metavar='N',
                    help='image size')
parser.add_argument('--mu-ratio', default=3, type=int, metavar='N',
                    help='multiplicative ratio for ct augment probe loader')
parser.add_argument('--min-step-lr', default=90, type=int, metavar='N',
                    help='minimum for step lr')
parser.add_argument('--max-step-lr', default=120, type=int, metavar='N',
                    help='maxiumum for step lr')
parser.add_argument('--save-every', default=5, type=int, metavar='S',
                    help='save checkpoints every N epochs')
parser.add_argument('--rand-depth', default=2, type=int,
                    help='depth of RandAugment')
parser.add_argument('--rand-magnitude', default=5, type=int,
                    help='magnitude of RandAugment')
parser.add_argument('--ct-depth', default=2, type=int,
                    help='depth of CT Augment')
parser.add_argument('--ct-decay', default=0.999, type=float,
                    help='decay of CT Augment')
parser.add_argument('--ct-thresh', default=0.85, type=float,
                    help='thresh of CT Augment')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='batch size for training/testing')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--use-scheduler', action='store_true',
                    help='Flag to use the scheduler during the training')                
parser.add_argument('--use-cosine', action='store_true',
                    help='use Cosine Scheduler')               
parser.add_argument('--use-ct', action='store_true',
                    help='use CTAugment strategy')  
parser.add_argument('--no-update-ct', action='store_true',
                    help='Flag that will disable to update the CTAug.')                  
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

best_acc1 = 0
best_epoch = -1
best_table, best_cr = '', ''


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
   
    def _lr_lambda(current_step):
   
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main_worker(args.gpu, args)

def main_worker(gpu, args):
    global best_acc1, best_epoch
    args.gpu = gpu

    if args.use_ct:
        print('=> Training with CTAugment ...')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # Adjust the FC layer's output features
    model.fc.out_features = args.num_classes

    print(model)
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = None
    if args.use_scheduler:
        if not args.use_cosine:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.min_step_lr, args.max_step_lr], gamma=0.1)
        else:
            print('=> Using the cosine scheduler')
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=args.epochs)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    dataset_list = DataList(args.data, percent=args.data_percent)
    classnames = dataset_list.classnames

    print('=> Data Distribution \n')
    print(dataset_list.class_distribution)

    train_sampler = None
    if args.use_weighted:

        print('=> Using Weighted Random Sampler \n')
        weight_class = dataset_list.weight_class
        weight_class = torch.from_numpy(weight_class).type('torch.DoubleTensor')
        if args.gpu is not None:
            weight_class = weight_class.cuda(args.gpu)
        train_sampler = WeightedRandomSampler(weights=weight_class, num_samples=len(weight_class))

    # ImageNet Normalization...
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = [
        transforms.RandomResizedCrop((args.size, args.size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(args.rand_depth, args.rand_magnitude),
        transforms.ToTensor(),
        normalize
    ]

    test_transforms = [
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        normalize
    ]

    if args.use_ct:
        train_transforms = [
            transforms.Resize((args.size, args.size)),
            transforms.RandomHorizontalFlip(),
            CTAugment(depth=args.ct_depth, decay=args.ct_decay, thresh=args.ct_thresh),
            transforms.ToTensor(),
            normalize
        ]
    
    train_transforms = transforms.Compose(train_transforms)
    test_transforms = transforms.Compose(test_transforms)

    updater = None
    if args.use_ct:
        updater = CTUpdater(
            args.data,
            train_transforms=train_transforms,
            batch_size=args.batch_size * args.mu_ratio,
            num_workers=args.workers,
            gpu=args.gpu
        )

    # optionally resume from ct checkpoint
    if args.use_ct and args.resume:
        if os.path.isfile(args.resume):
            state_dict = torch.load(args.resume)
            if state_dict['ct_state_dict'] is not None:
                print("=> loading ct checkpoint '{}'".format(args.resume))
                updater.CTClass.load_state_dict(state_dict['ct_state_dict'])
            else:
                print("=> no ct state dict checkpoint found at '{}'".format(args.resume))
        else:
            warnings.warn('=> %s path is not a valid file to load dictionary. Continuing without loading the checkpoints...' % args.resume)

    train_dataset = MyDataset(
        data_source=dataset_list.train,
        train_transform=train_transforms,
        is_train=True
    )

    val_dataset = MyDataset(
        data_source=dataset_list.val,
        test_transform=test_transforms,
        is_train=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    print('=> Dataset statistics...')
    print('TRAIN: %d | VAL: %d | Classes: %d' % (len(train_dataset), len(val_dataset), dataset_list.num_classes))

    print('=> Training statistics...')
    print('Initial LR: %.3f | Epochs: %d | Batch Size: %d | No. Iters: %d' % (args.lr, args.epochs, args.batch_size, args.epochs * len(train_loader)))

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, updater=updater)

        # evaluate on validation set
        acc1, table, cr = validate(val_loader, model, criterion, args, classnames=classnames)
        
        # update the scheduler
        if args.use_scheduler:
            scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_epoch = epoch if is_best else best_epoch
        best_table = table if is_best else best_table
        best_cr = cr if is_best else best_cr

        if (epoch + 1) % args.save_every == 0 or is_best:
            if is_best:
                print('Saving the best model Top1 acc = %.3f ...' % best_acc1)
                if args.use_ct:
                    print(' CT Weights')
                    print(updater.CTClass.stats)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'ct_state_dict': updater.CTClass.state_dict() if args.use_ct else None
            }, is_best, output_dir=args.output_dir)

    print('Training finished ...')
    print('Best Epoch: %d Best Accuracy: %.3f' % (best_epoch, best_acc1))
    print(f'=> Classification Report \n {best_cr}')
    print(f'=> Confusion Matrix \n {best_table}')


def parse_batch(batch, cuda_flag=None):

    input = batch['img']
    label = batch['label']

    if cuda_flag is not None:
        input = input.cuda(cuda_flag, non_blocking=True)
        label = label.cuda(cuda_flag, non_blocking=True)

    return input, label


def train(train_loader, model, criterion, optimizer, epoch, args, updater=None):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for i, batch in enumerate(train_loader):

        images, target = parse_batch(batch, cuda_flag=args.gpu)
        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        (acc1, acc5), _, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        # Update the CTAugment stats
        if args.use_ct and not args.no_update_ct:
            updater.update(model)


def validate(val_loader, model, criterion, args, classnames=None):

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    stats = DisplayStats(classnames=classnames)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):

            images, target = parse_batch(batch, cuda_flag=args.gpu)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            (acc1, acc5), pred, truth = accuracy(output, target, topk=(1, 5))

            stats.update(truth, pred)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display_summary()
        table, cr = stats.display()

        del stats
        gc.collect()

    return top1.avg, table, cr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', output_dir='outputs'):

    torch.save(state, os.path.join(output_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(output_dir, filename), os.path.join(output_dir, 'model_best.pth.tar'))


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class DisplayStats(object):

    def __init__(self, classnames):

        self.classnames = classnames
        self.preds = list()
        self.truths = list()

    def reset(self):
        self.preds, self.truths = list(), list()

    def update(self, truth, pred):
        truth = truth.cpu().numpy().reshape(1, -1).tolist()
        pred = pred.cpu().numpy().reshape(1, -1).tolist()
        self.truths.append(truth)
        self.preds.append(pred)

    def display(self):

        self.truths = sum(sum(self.truths, []), [])
        self.preds = sum(sum(self.preds, []), [])
        cm = confusion_matrix(self.truths, self.preds)
        cr = classification_report(self.truths, self.preds, target_names=self.classnames, zero_division=0)

        print('=> Confusion Matrix \n')
        table = PrettyTable([' '] + self.classnames)
        table.add_rows([[self.classnames[i]] + cm[i].tolist() for i in range(len(self.classnames))])
        print(table)

        print('=> Classification Report \n')
        print(cr + '\n')

        return table, cr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        _, max_pred = torch.max(output, dim=-1)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, max_pred, target


if __name__ == '__main__':
    main()
