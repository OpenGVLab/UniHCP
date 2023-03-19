import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import HungarianMatcher, DirectMatcher, RedundantQMatcher, POSDirectMatcher
from .criterion import SetCriterion, POSSetCriterion

class BasePosLoss(nn.Module):
    def __init__(self, target_type, use_target_weight=True, cfg=None):
        super(BasePosLoss, self).__init__()
        self.criterion = nn.MSELoss()

        self.target_type = target_type
        self.use_target_weight = use_target_weight

        self.cfg = cfg

    def get_loss(self, num_joints, heatmaps_pred, heatmaps_gt, target_weight):
        loss = 0.
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred * target_weight[:, idx],
                                       heatmap_gt * target_weight[:, idx])
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)
        return loss

    def forward(self, outputs, target, target_weight):  # {"aux_outputs": xx, 'xx': xx}
        """Forward function."""
        output = outputs['pred_masks']  # {'pred_logits':'pred_masks':}

        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = self.get_loss(num_joints, heatmaps_pred, heatmaps_gt, target_weight)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs and self.cfg.get('aux_loss', True):
            for aux_outputs in outputs["aux_outputs"]:
                heatmaps_pred = aux_outputs['pred_masks'].reshape((batch_size, num_joints, -1)).split(1, 1)

                loss = loss + self.get_loss(num_joints, heatmaps_pred, heatmaps_gt, target_weight)

        return loss / num_joints

class POS_FocalDiceLoss_bce_cls_emb(nn.Module):
    def __init__(self, target_type, use_target_weight=True, cfg=None):
        super(POS_FocalDiceLoss_bce_cls_emb, self).__init__()
        self.target_type = target_type
        self.use_target_weight = use_target_weight

        matcher = POSDirectMatcher()

        weight_dict = {"loss_bce_pos": cfg.class_weight,
                       "loss_mask_pos": cfg.mask_weight,
                       }

        if cfg.get('deep_supervision', False):
            aux_weight_dict = {}
            for i in range(cfg.dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})   # {loss_ce_i : cfg.class_weight ...}
            weight_dict.update(aux_weight_dict)

        self.fd_loss = POSSetCriterion(
            cfg.num_classes,
            ginfo=cfg.ginfo,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=[
                "pos_mask",
                "pos_bce_labels",
            ],
            eos_coef=cfg.get('eos_coef', 0.1),
            aux=cfg.get('deep_supervision', False),
            ignore_blank=cfg.get('ignore_blank', True),
            sample_weight=cfg.get('sample_weight', None)
        )

        self.cfg = cfg

    def forward(self, outputs, targets, target_weight, **kwargs):  # {"aux_outputs": xx, 'xx': xx}
        losses = self.fd_loss(outputs, targets)

        for k in list(losses.keys()):
            if k in self.fd_loss.weight_dict:
                losses[k] *= self.fd_loss.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        return losses
