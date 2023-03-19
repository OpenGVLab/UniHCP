import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils import (add_task_specific, add_neck_specific, add_decoder_specific, add_backbone_specific,
                        add_aio_decoder_specific, add_aio_backbone_specific, add_aio_neck_specific)

class model_entry(nn.Module):
    def __init__(self, backbone_module, neck_module, decoder_module):
        super(model_entry, self).__init__()
        self.backbone_module = backbone_module
        self.neck_module = neck_module
        self.decoder_module = decoder_module
        add_task_specific(self, False)
        add_backbone_specific(self.backbone_module, True)
        add_neck_specific(self.neck_module, True)
        add_decoder_specific(self.decoder_module, True)
        if hasattr(self.decoder_module, 'loss'):
            if hasattr(self.decoder_module.loss, 'classifier'):
                add_task_specific(self.decoder_module.loss, True)

    def forward(self, input_var, current_step):
        x = self.backbone_module(input_var) # {'image': img_mask, 'label': target_mask, 'filename': img_name, 'backbone_output':xxx}
        x = self.neck_module(x)
        decoder_feature = self.decoder_module(x)
        return decoder_feature


class aio_entry(nn.Module):
    def __init__(self, backbone_module, neck_module, decoder_module):
        super(aio_entry, self).__init__()
        self.backbone_module = backbone_module
        self.neck_module = neck_module
        self.decoder_module = decoder_module
        add_task_specific(self, False)
        add_aio_backbone_specific(self.backbone_module, True, self.backbone_module.task_sp_list)
        add_aio_neck_specific(self.neck_module, True, self.neck_module.task_sp_list)
        add_aio_decoder_specific(self.decoder_module, True, self.decoder_module.task_sp_list,
                                 self.decoder_module.neck_sp_list)

    def forward(self, input_var, current_step):
        if current_step < self.backbone_module.freeze_iters:
            with torch.no_grad():
                x = self.backbone_module(input_var)  # {'image': img_mask, 'label': target_mask, 'filename': img_name, 'backbone_output':xxx}
        else:
            x = self.backbone_module(input_var) # {'image': img_mask, 'label': target_mask, 'filename': img_name, 'backbone_output':xxx}
        x = self.neck_module(x)
        decoder_feature = self.decoder_module(x)
        return decoder_feature


class aio_entry_v2(aio_entry):
    def __init__(self, backbone_module, neck_module, decoder_module):
        super(aio_entry, self).__init__()
        self.backbone_module = backbone_module
        self.neck_module = neck_module
        self.decoder_module = decoder_module
        add_task_specific(self, False)
        add_aio_backbone_specific(self.backbone_module, True, self.backbone_module.task_sp_list,
                                  self.backbone_module.neck_sp_list)
        add_aio_backbone_specific(self.neck_module, True, self.neck_module.task_sp_list)
        add_aio_decoder_specific(self.decoder_module, True, self.decoder_module.task_sp_list,
                                 self.decoder_module.neck_sp_list)


