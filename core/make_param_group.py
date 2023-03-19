import numpy as np
import shutil
import torch
import os
import io
import logging
from collections import defaultdict


from torch.nn import BatchNorm2d

def param_group_no_wd(model):
    pgroup_no_wd = []
    names_no_wd = []
    pgroup_normal = []

    type2num = defaultdict(lambda : 0)
    for name,m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            if m.bias is not None:
                pgroup_no_wd.append(m.bias)
                names_no_wd.append(name+'.bias')
                type2num[m.__class__.__name__+'.bias'] += 1
        elif isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                pgroup_no_wd.append(m.bias)
                names_no_wd.append(name+'.bias')
                type2num[m.__class__.__name__+'.bias'] += 1
        elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            if m.weight is not None:
                pgroup_no_wd.append(m.weight)
                names_no_wd.append(name+'.weight')
                type2num[m.__class__.__name__+'.weight'] += 1
            if m.bias is not None:
                pgroup_no_wd.append(m.bias)
                names_no_wd.append(name+'.bias')
                type2num[m.__class__.__name__+'.bias'] += 1

    for name,p in model.named_parameters():
        if not name in names_no_wd:
            pgroup_normal.append(p)

    return [{'params': pgroup_normal}, {'params': pgroup_no_wd, 'weight_decay': 0.0}], type2num

def param_group_fc(model):
    logits_w_id = id(model.module.logits.weight)
    fc_group = []
    normal_group = []
    for p in model.parameters():
        if id(p) == logits_w_id:
            fc_group.append(p)
        else:
            normal_group.append(p)
    param_group = [{'params': fc_group}, {'params': normal_group}]

    return param_group

def param_group_multitask(model):
    backbone_group = []
    neck_group = []
    decoder_group = []
    other_group = []
    for name, p in model.named_parameters():
        if 'module.backbone_module' in name:
            backbone_group.append(p)
        elif 'module.neck_module' in name:
            neck_group.append(p)
        elif 'module.decoder_module' in name:
            decoder_group.append(p)
        else:
            other_group.append(p)

    if len(other_group) > 0:
        param_group = [{'params': backbone_group}, {'params': neck_group}, \
                        {'params': decoder_group}, {'params', other_group}]
    else:
        param_group = [{'params': backbone_group}, {'params': neck_group}, \
                        {'params': decoder_group}]
    return param_group
