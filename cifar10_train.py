import os
import tqdm
import argparse
import pprint

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np
import skimage
import os
import glob
from skimage.io import imread
import skimage
import math
import time
from models import get_model
from losses import get_loss
from optimizers import get_optimizer, get_q_optimizer
from schedulers import get_scheduler
from tensorboardX import SummaryWriter
from evaluators import accuracy

import utils.config
import utils.checkpoint
from utils import AverageMeter

from torch.utils.data import DataLoader
import torchvision

device = None

def train_single_epoch(config, model, dataloader, criterion,
                       optimizer, q_optimizer, epoch, writer, postfix_dict):
    model.train()
    batch_size = config.train.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    log_dict = {}

    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    for i, (imgs, labels) in tbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        q_optimizer.zero_grad()


        pred_dict = model(imgs)
        loss = criterion['train'](pred_dict['out'], labels)

        for k, v in loss.items():
            log_dict[k] = v.item()

        loss['loss'].backward()

        optimizer.step()
        q_optimizer.step()

        ## logging
        f_epoch = epoch + i / total_step
        log_dict['lr'] = optimizer.param_groups[0]['lr']
        for key, value in log_dict.items():
            postfix_dict['train/{}'.format(key)] = value

        desc = '{:5s}'.format('train')
        desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
        tbar.set_description(desc)
        tbar.set_postfix(**postfix_dict)

        # tensorboard
        if i % 10 == 0:
            log_step = int(f_epoch * 1280)
            if writer is not None:
                for key, value in log_dict.items():
                    writer.add_scalar('train/{}'.format(key), value, log_step)

def evaluate_single_epoch(config, model,
                          dataloader, criterion, epoch, writer,
                          postfix_dict, eval_type):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    with torch.no_grad():
        batch_size = config.eval.batch_size
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)

        for i, (imgs, labels) in tbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            pred_dict = model(imgs)
            train_loss = criterion['val'](pred_dict['out'], labels)
            prec1, prec5 = accuracy(pred_dict['out'].data, labels.data, topk=(1,5))
            prec1 = prec1[0]
            prec5 = prec5[0]

            losses.update(train_loss.item(), labels.size(0))
            top1.update(prec1, labels.size(0))
            top5.update(prec5, labels.size(0))

            ## Logging
            f_epoch = epoch + i / total_step
            desc = '{:5s}'.format(eval_type)
            desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
            tbar.set_description(desc)
            tbar.set_postfix(**postfix_dict)

        ## logging
        log_dict = {}
        log_dict['loss'] = losses.avg
        log_dict['top1'] = top1.avg.item()
        log_dict['top5'] = top5.avg.item()

        print(log_dict)

        for key, value in log_dict.items():
            if writer is not None:
                writer.add_scalar('{}/{}'.format(eval_type, key), value, epoch)
            postfix_dict['{}/{}'.format(eval_type, key)] = value

        return log_dict['top1'], log_dict['top5']

def train(config, model, dataloaders, criterion,
          optimizer, q_optimizer, scheduler, q_scheduler, writer, start_epoch):
    num_epochs = config.train.num_epochs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    postfix_dict = {'train/lr': 0.0,
                    'train/loss': 0.0,
                    'train/accuracy': 0.0,
                    'test/accuracy':0.0,
                    'test/loss':0.0}

    best_accuracy = 0.0

    for epoch in range(start_epoch, num_epochs):
        # train phase
        train_single_epoch(config, model, dataloaders['train'],
                           criterion, optimizer, q_optimizer, epoch, writer,
                           postfix_dict)

        # test phase
        top1, top5 = evaluate_single_epoch(config, model,
                                           dataloaders['test'],
                                           criterion, epoch, writer,
                                           postfix_dict, eval_type='test')

        scheduler.step()
        q_scheduler.step()

        if best_accuracy < top1:
            best_accuracy = top1

    utils.checkpoint.save_checkpoint(config, model, optimizer, scheduler, q_optimizer,
                                     q_scheduler, None, None, epoch, 0, 'model')

    return {'best_accuracy': best_accuracy}

def qparam_extract(model):

    var = list()

    for m in model._modules:
        if len(model._modules[m]._modules) > 0:
            var = var + qparam_extract(model._modules[m])

        else:
            if hasattr(model._modules[m], 'init'):
                var = var + list(model._modules[m].parameters())[1:]

    return var

def param_extract(model):

    var = list()

    for m in model._modules:
        if len(model._modules[m]._modules) > 0:
            var = var + param_extract(model._modules[m])

        else:
            if hasattr(model._modules[m], 'init'):
                var = var + list(model._modules[m].parameters())[0:1]
            else:
                var = var + list(model._modules[m].parameters())

    return var

def run(config):

    model = get_model(config).to(device)
    print("The number of parameters : %d" % count_parameters(model))
    criterion = get_loss(config)

    q_param = qparam_extract(model)
    param = param_extract(model)

    optimizer = get_optimizer(config, param)
    q_optimizer = get_q_optimizer(config, q_param)

    # Loading the full-precision model
    if config.model.pretrain.pretrained:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(config.model.pretrain.dir)['state_dict']

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('load the pretrained model')

    checkpoint = utils.checkpoint.get_initial_checkpoint(config, model_type)

    last_epoch, step = -1, -1
    print('model from checkpoint: {} last epoch:{}'.format(
        checkpoint, last_epoch))

    scheduler = get_scheduler(config, optimizer, last_epoch)
    q_scheduler = get_scheduler(config, q_optimizer, last_epoch)

    # Data augmentation
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])

    dataloader = torchvision.datasets.CIFAR10

    trainset = dataloader(root='./data', train=True, download=True, transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=config.train.batch_size, shuffle=True,
                             num_workers=config.data.num_workers, pin_memory=config.data.pin_memory)

    testset = dataloader(root='./data', train=False, download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=config.eval.batch_size, shuffle=False,
                            num_workers=config.data.num_workers)

    dataloaders = {'train': trainloader,
                   'test': testloader}

    writer = SummaryWriter(config.train['model' + '_dir'])

    train(config, model, dataloaders, criterion, optimizer, q_optimizer,
          scheduler, q_scheduler, writer, last_epoch+1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_args():
    parser = argparse.ArgumentParser(description='quantization network')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()

def main():
    global device
    global model_type
    model_type = 'model'
    import warnings
    warnings.filterwarnings("ignore")

    print('train %s network'%model_type)
    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')

    config = utils.config.load(args.config_file)

    os.environ["CUDA_VISIBLE_DEVICES"]= str(config.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pprint.PrettyPrinter(indent=2).pprint(config)
    utils.prepare_train_directories(config, model_type)
    run(config)

    print('success!')

if __name__ == '__main__':
    main()
