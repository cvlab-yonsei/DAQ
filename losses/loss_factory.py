from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy_dist_epoch(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    l1_fn = torch.nn.L1Loss(reduction=reduction)

    def loss_fn(outputs, outputs_f, labels, epoch, **_):
        loss_dict = dict()
        full_gt_loss = cross_entropy_fn(outputs_f['out'], labels)
        gt_loss = cross_entropy_fn(outputs['out'], labels)
        dist_loss = 0
        layer_names = outputs.keys()
        len_layer = len(layer_names)

        for i, layer_name in enumerate(layer_names):
            if i == len_layer - 1:
                continue
            dist_loss += l1_fn(outputs[layer_name], outputs_f[layer_name])

        scale = epoch / 100
        if epoch == 100:
            scale = 1

        loss_dict['loss'] = scale*(gt_loss + dist_loss) + full_gt_loss
        loss_dict['gt_loss'] = gt_loss
        loss_dict['full_gt_loss'] = full_gt_loss
        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def cross_entropy_dist(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    l1_fn = torch.nn.L1Loss(reduction=reduction)

    def loss_fn(outputs, outputs_f, labels, **_):
        loss_dict = dict()
        full_gt_loss = cross_entropy_fn(outputs_f['out'], labels)
        gt_loss = cross_entropy_fn(outputs['out'], labels)
        dist_loss = 0
        layer_names = outputs.keys()
        len_layer = len(layer_names)

        for i, layer_name in enumerate(layer_names):
            if i == len_layer - 1:
                continue
            dist_loss += l1_fn(outputs[layer_name], outputs_f[layer_name])

        loss_dict['loss'] = gt_loss + dist_loss + full_gt_loss
        loss_dict['gt_loss'] = gt_loss
        loss_dict['full_gt_loss'] = full_gt_loss
        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def cross_entropy(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)

    def loss_fn(outputs, labels, **_):
        loss_dict = dict()
        gt_loss = cross_entropy_fn(outputs, labels)
        loss_dict['loss'] = gt_loss
        loss_dict['gt_loss'] = gt_loss
        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def regularization(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)

    def loss_fn(outputs, labels, reg_factors, **_):
        loss_dict = dict()
        gt_loss = cross_entropy_fn(outputs, labels)
        reg_loss = 0
        for i in range(len(reg_factors)):
            reg_loss += torch.mean((torch.pow(reg_factors[i]-1, 2)*torch.pow(reg_factors[i]+1, 2)))
        reg_loss = reg_loss / len(reg_factors)
        loss_dict['loss'] = gt_loss + reg_loss
        loss_dict['gt_loss'] = gt_loss
        loss_dict['reg_loss'] = reg_loss
        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}

def regularization_temp(reduction='mean', **_):
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction=reduction)

    def loss_fn(outputs, labels, reg_factors, **_):
        loss_dict = dict()
        gt_loss = cross_entropy_fn(outputs, labels)
        reg_loss = 0
        for i in range(len(reg_factors)):
            reg_loss += torch.mean((torch.pow(reg_factors[i]-1, 2)*torch.pow(reg_factors[i]+1, 2)))
        reg_loss = reg_loss / len(reg_factors)
        loss_dict['loss'] = gt_loss + reg_loss
        loss_dict['gt_loss'] = gt_loss
        loss_dict['reg_loss'] = reg_loss
        return loss_dict


    return {'train':loss_fn,
            'val':cross_entropy_fn}


def get_loss(config):
    f = globals().get(config.loss.name)
    return f(**config.loss.params)
