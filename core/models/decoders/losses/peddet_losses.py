import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import DetectionHungarianMatcher
from .criterion import DetSetCriterion

class DetFocalDiceLoss(nn.Module):
    def __init__(self, cfg):
        super(DetFocalDiceLoss, self).__init__()
        matcher = DetectionHungarianMatcher(
            cost_class=cfg.class_weight,
            cost_bbox=cfg.bbox_weight,
            cost_giou=cfg.giou_weight,
        )

        weight_dict = {"loss_ce": cfg.class_weight,
                       "loss_bbox": cfg.bbox_weight,
                       "loss_giou": cfg.giou_weight}

        if cfg.deep_supervision:
            aux_weight_dict = {}
            for i in range(cfg.dec_layers-1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})   # {loss_ce_i : cfg.class_weight ...}
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)



        self.fd_loss = DetSetCriterion(
            cfg.num_classes,
            ginfo=cfg.ginfo,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=["labels", "boxes"],
            focal_alpha=cfg.focal_alpha,
            ign_thr=cfg.ign_thr,
        )

        self.cfg = cfg

    def forward(self, outputs, targets, **kwargs): # {"aux_outputs": xx, 'xx': xx}
        losses = self.fd_loss(outputs, targets)
        for k in list(losses.keys()):
            if k in self.fd_loss.weight_dict:
                losses[k] *= self.fd_loss.weight_dict[k]
            elif 'loss' in k:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        return losses


class DetFocalDiceLoss_hybrid(DetFocalDiceLoss):
    def forward(self, outputs, targets, **kwargs): # {"aux_outputs": xx, 'xx': xx}
        multi_targets = copy.deepcopy(targets)
        losses = self.fd_loss(outputs, targets)

        for target in multi_targets:
            target["boxes"] = target["boxes"].repeat(self.cfg.k_one2many, 1)
            target["labels"] = target["labels"].repeat(self.cfg.k_one2many)
            assert len(target["iscrowd"].shape) == 1, f"len(target['iscrowd'].shape) == 1: {len(target['iscrowd'].shape) == 1}"
            target["iscrowd"] = target["iscrowd"].repeat(self.cfg.k_one2many)
        outputs_one2many = dict()
        outputs_one2many["pred_logits"] = outputs["pred_logits_one2many"]
        outputs_one2many["pred_boxes"] = outputs["pred_boxes_one2many"]
        outputs_one2many["aux_outputs"] = outputs["aux_outputs_one2many"]
        outputs_one2many["mask"] = outputs["mask"]
        losses_one2many = self.fd_loss(outputs_one2many, multi_targets)


        for k in list(losses.keys()):
            if k in self.fd_loss.weight_dict:
                losses[k] *= self.fd_loss.weight_dict[k]
            elif 'loss' in k:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        for k in list(losses_one2many.keys()):  # repeat
            if k in self.fd_loss.weight_dict:
                losses_one2many[k] *= self.fd_loss.weight_dict[k]
            elif 'loss' in k:
                losses_one2many.pop(k)

        for key, value in losses_one2many.items():
            if key + "_one2many" in losses.keys():
                losses[key + "_one2many"] += value * self.cfg.get('lambda_one2many', 1)
            else:
                losses[key + "_one2many"] = value * self.cfg.get('lambda_one2many', 1)

        return losses