import warnings
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from core.memory import retry_if_cuda_oom
from core.data.transforms.post_transforms import pose_pck_accuracy, flip_back, transform_preds
from core.models.ops import box_ops
from ..transformer_decoder import TransformerDecoder
from ...losses import loss_entry
from ast import literal_eval


class AIOHead(nn.Module):
    def __init__(self,
                 transformer_predictor_cfg,
                 loss_cfg,
                 num_classes,
                 backbone,  # placeholder
                 neck,  # placeholder
                 loss_weight,
                 ignore_value,
                 ginfo,
                 bn_group,  # placeholder
                 task_sp_list=(),
                 neck_sp_list=(),
                 task='seg',
                 test_cfg=None,
                 predictor='td',
                 feature_only=False # redundant param in compliance with past reid test code
                 ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes to predict
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            task_sp_list: specify params/buffers in decoder that should be treated task-specific in reduce_gradients()
            neck_sp_list: specify params/buffers in decoder that should be treated neck-specific in reduce_gradients()
        """
        super().__init__()
        self.task = task
        self.task_sp_list = task_sp_list
        self.neck_sp_list = neck_sp_list

        self.backbone = [backbone]  # avoid recursive specific param register
        self.neck = [neck]  # avoid recursive specific param register

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        if predictor == 'td':
            self.predictor = TransformerDecoder(in_channels=neck.vis_token_dim,
                                                mask_dim=neck.mask_dim,
                                                mask_classification=True,
                                                num_classes=num_classes,
                                                ginfo=ginfo,
                                                backbone_pose_embed=backbone.pos_embed,
                                                **transformer_predictor_cfg)
        else:
            raise
        loss_cfg.kwargs.cfg.num_classes = num_classes
        loss_cfg.kwargs.cfg.ignore_value = ignore_value
        loss_cfg.kwargs.cfg.ginfo = ginfo
        self.loss = loss_entry(loss_cfg)

        self.test_cfg = {} if test_cfg is None else test_cfg

        self.num_classes = num_classes

    def prepare_targets(self, targets, images):  # for seg
        h_pad, w_pad = images.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt TODO: seems duplicated?
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def prepare_detection_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            new_targets.append(
                {
                    "boxes": targets_per_image.boxes,
                    "labels": targets_per_image.labels,
                    "area": targets_per_image.area,
                    "iscrowd": targets_per_image.iscrowd,
                }
            )
        return new_targets

    def get_accuracy(self, output, target, target_weight):  # for pos
        """Calculate accuracy for top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K
            heatmaps height: H
            heatmaps weight: W

        Args:
            output (torch.Tensor[NxKxHxW]): Output heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """

        if self.loss.target_type == 'GaussianHeatMap':
            _, avg_acc, _ = pose_pck_accuracy(
                output.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                target_weight.detach().cpu().numpy().squeeze(-1) > 0)
            accuracy = avg_acc
        else:
            raise NotImplementedError(f"Unknown target type: {self.loss.target_type}")

        return torch.Tensor([accuracy]).cuda()

    def forward(self, features):  # input -> loss, top1, etc.
        # {'image': img_mask, 'label': target_mask, 'filename': img_name, 'backbone_output':xxx, 'neck_output': xxx}
        images = features['image']  # double padded image(s)

        # --> {'pred_logits':'pred_masks':
        #      'aux_outputs':[{'pred_logits':'pred_masks':}, ...]}}
        if self.task == 'seg':
            outputs = self.predictor(features['neck_output']['multi_scale_features'],
                                     features['neck_output']['mask_features'])
            if self.training:
                # mask classification target
                assert "instances" in features
                gt_instances = features["instances"]
                targets = self.prepare_targets(gt_instances, images)

                # bipartite matching-based loss
                losses = self.loss(outputs, targets)
                for k in losses:
                    losses[k] = losses[k] * self.loss_weight

                return {'loss': losses, 'top1': torch.FloatTensor([0]).cuda()}
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.shape[-2], images.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )
                del outputs

                processed_results = []

                mask_cls_result = mask_cls_results[0]
                mask_pred_result = mask_pred_results[0]
                input_per_image = features  # hot fixes
                image_size = features['prepad_input_size']

                height = input_per_image.get("height", None)
                width = input_per_image.get("width", None)  # image_size[1])
                processed_results.append({})

                # semantic segmentation inference
                r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                # if not self.sem_seg_postprocess_before_inference:
                r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                processed_results[-1]["sem_seg"] = r

                return processed_results
        elif self.task == 'pos':
            outputs = self.predictor(features['neck_output']['multi_scale_features'],
                                     features['neck_output']['mask_features'])
            if self.training:
                target = features['label']
                target_weight = features['target_weight']

                pos_loss = self.loss(outputs, target, target_weight) * self.loss_weight

                acc = self.get_accuracy(outputs['pred_masks'], target, target_weight)
                return {'feature': outputs, 'loss': pos_loss, 'top1': acc}
            else:
                assert features['image'].size(0) == len(features['img_metas'])
                batch_size, _, img_height, img_width = features['image'].shape
                if batch_size > 1:
                    assert 'bbox_id' in features['img_metas'][0].data
                output_heatmap = self.pose_inference(outputs["pred_masks"], flip_pairs=None)

                if self.test_cfg.get('flip_test', True):  # True
                    img_flipped = {'image': features['image'].flip(3)}
                    features_flipped = self.backbone[0](img_flipped)
                    features_flipped = self.neck[0](features_flipped)
                    features_flipped = self.predictor(features_flipped['neck_output']['multi_scale_features'],
                                                      features_flipped['neck_output']['mask_features'])

                    output_flipped_heatmap = self.pose_inference(features_flipped["pred_masks"],
                                                                 features['img_metas'][0].data['flip_pairs'])
                    output_heatmap = (output_heatmap + output_flipped_heatmap) * 0.5

                keypoint_result = self.pose_decode(features['img_metas'], output_heatmap)
                return keypoint_result
        elif self.task == 'pos_bce':
            outputs = self.predictor(features['neck_output']['multi_scale_features'],
                                     features['neck_output']['mask_features'])
            if self.training:
                target = features
                target_weight = features['target_weight']

                pos_losses = self.loss(outputs, target, target_weight)

                for k in pos_losses:
                    pos_losses[k] = pos_losses[k] * self.loss_weight

                acc = self.get_accuracy(outputs['pred_masks'], features['label'], target_weight)
                return {'feature': outputs, 'loss': pos_losses, 'top1': acc}
            else:
                assert features['image'].size(0) == len(features['img_metas'])

                batch_size, _, img_height, img_width = features['image'].shape
                if batch_size > 1:
                    assert 'bbox_id' in features['img_metas'][0].data
                output_heatmap = self.pose_inference(outputs["pred_masks"], flip_pairs=None)

                if self.test_cfg.get('flip_test', True):  # True
                    img_flipped = {'image': features['image'].flip(3)}
                    features_flipped = self.backbone[0](img_flipped)
                    features_flipped = self.neck[0](features_flipped)
                    features_flipped = self.predictor(features_flipped['neck_output']['multi_scale_features'],
                                                      features_flipped['neck_output']['mask_features'])

                    output_flipped_heatmap = self.pose_inference(features_flipped["pred_masks"],
                                                                 features['img_metas'][0].data['flip_pairs'])
                    output_heatmap = (output_heatmap + output_flipped_heatmap) * 0.5
                ### enisum with cls branch
                # cls = outputs["pred_logits"].sigmoid().reshape(outputs["pred_logits"].shape[0],-1)
                # output_heatmap = torch.einsum('kq,kqhw->kqhw', cls,
                #                      torch.tensor(output_heatmap.copy(), device=cls.device)).cpu().numpy()
                ###
                keypoint_result = self.pose_decode(features['img_metas'], output_heatmap)
                keypoint_result['pred_logits'] = outputs['pred_logits'].sigmoid().cpu().numpy()
                return keypoint_result
        elif self.task == 'par':
            outputs = self.predictor(features['neck_output']['multi_scale_features'],
                                     features['neck_output']['mask_features'])
            if self.training:
                # mask classification target
                assert "instances" in features
                gt_instances = features["instances"]
                targets = self.prepare_targets(gt_instances, images)

                # bipartite matching-based loss
                losses = self.loss(outputs, targets)
                for k in losses:
                    losses[k] = losses[k] * self.loss_weight

                return {'loss': losses, 'top1': torch.FloatTensor([0]).cuda()}
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.shape[-2], images.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )
                del outputs

                processed_results = []
                # import pdb;
                # pdb.set_trace()

                # re = retry_if_cuda_oom(self.semantic_inference_batch)(mask_cls_results, mask_pred_results)
                # image_size = (images.shape[-2], images.shape[-1])
                #
                # for _idx in range(len(mask_cls_results)):
                #     height = features.get("height", None)[_idx].item()
                #     width = features.get("width", None)[_idx].item()
                #     processed_results.append({})
                #     r = retry_if_cuda_oom(sem_seg_postprocess)(re[_idx], image_size, height, width)
                #     processed_results[-1]["sem_seg"] = r
                for _idx, (mask_cls_result, mask_pred_result,) in enumerate(zip(  # currently, only batch_size == 1 is supported
                        mask_cls_results, mask_pred_results
                )):
                    image_size = (images.shape[-2], images.shape[-1])
                    # height = features.get("height", None)[_idx].item()
                    # width = features.get("width", None)[_idx].item()
                    # import pdb;pdb.set_trace()
                    try:
                        height = features.get("gt", None).shape[-2] #.item()
                        width = features.get("gt", None).shape[-1] #.item()
                    except:
                        height = features['height'][_idx].item()
                        width = features['width'][_idx].item() #np.array(literal_eval(features['gt'][_idx])).shape[-1]

                # input_per_image = features  # hot fixes

                # image_size = features['prepad_input_size']

                # height = input_per_image.get("height", None)
                # width = input_per_image.get("width", None)  # image_size[1])
                    processed_results.append({})

                # semantic segmentation inference
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    # if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                return processed_results
        elif self.task == 'par_bce_cls_emb':
            outputs = self.predictor(features['neck_output']['multi_scale_features'],
                                     features['neck_output']['mask_features'], mask_label=features.get('mask', None))
            if self.training:
                # mask classification target
                assert "instances" in features

                gt_instances = features["instances"]
                targets = self.prepare_targets(gt_instances, images)

                # bipartite matching-based loss
                losses = self.loss(outputs, targets)

                for k in losses:
                    losses[k] = losses[k] * self.loss_weight
                # pdb.set_trace()
                return {'loss': losses, 'top1': torch.FloatTensor([0]).cuda()}
            else:
                mask_cls_results = outputs["pred_logits"]
                # mask_cls_results = torch.nn.functional.sigmoid(mask_cls_results)
                mask_pred_results = outputs["pred_masks"]  # [bs, queries, h, w]
                bs, queries, h,w = mask_pred_results.shape
                # import pdb;pdb.set_trace()
                redundant_queries = int(queries / self.num_classes)
                mask = mask_pred_results.clone()
                if redundant_queries > 1:

                    # select = list(range(1,queries+1, redundant_queries))
                    mask = mask.reshape(bs, self.num_classes, redundant_queries, h,w)
                    pooling = self.test_cfg.get('pooling', 'max')
                    if pooling == 'max':
                        mask = torch.max(mask, 2)[0]
                    elif pooling == 'avg':
                        mask = torch.sum(mask,2)

                # use predicted logits to remove not existing classes, TODO support redundant_queries
                # for i in range(len(mask)):
                #     pred_logits = mask_cls_results[i].reshape(-1)
                #     remove = torch.nonzero(torch.where(pred_logits<0.5, 1,0)).reshape(-1)
                #     mask[i][remove]=-1e15
                # use gt to remove not existing classes
                # for i in range(len(mask)):
                #     gt_label = features['gt'][i].data.astype(np.int)
                #     label = np.unique(gt_label).tolist()
                #     remove = list(set(list(range(self.num_classes)))-set(label))
                #     mask[i][remove]=-1e15

                mask_pred_results = mask
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.shape[-2], images.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )
                del outputs

                processed_results = []

                for _idx, (mask_cls_result, mask_pred_result) in enumerate(  # currently, only batch_size == 1 is supported
                        zip( mask_cls_results, mask_pred_results )
                ):
                    image_size = (images.shape[-2], images.shape[-1])
                    # height = features.get("height", None)[_idx].item()
                    # width = features.get("width", None)[_idx].item()
                    # import pdb;pdb.set_trace()
                    try:
                        height = features.get("gt", None).shape[-2] #.item()
                        width = features.get("gt", None).shape[-1] #.item()
                    except:
                        height = features['height'][_idx].item()
                        width = features['width'][_idx].item() #np.array(literal_eval(features['gt'][_idx])).shape[-1]

                # input_per_image = features  # hot fixes

                # image_size = features['prepad_input_size']

                # height = input_per_image.get("height", None)
                # width = input_per_image.get("width", None)  # image_size[1])
                    processed_results.append({})

                # semantic segmentation inference
                    r= retry_if_cuda_oom(self.semantic_inference_bce)(mask_cls_result, mask_pred_result)
                    # r = mask_pred_result
                    # if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                return processed_results
        elif self.task == 'reid':
            output = self.predictor.forward_reid(features['neck_output']['multi_scale_features'],
                                                 self.loss.norm)
            # if self.feature_bn:  place within predictior& loss
            #     features = self.feat_bn(features_nobn)
            output['label'] = features['label']
            if self.training:
                logits = self.loss(output)
                output.update(logits)
                return output
            else:
                return output
        elif self.task == 'pedattr':
            output = self.predictor.forward_attr(features['neck_output']['multi_scale_features'])  # {'logit':xxx}
            output['label'] = features['label']
            if self.training:
                losses = self.loss(output)
                output.update(losses)
                return output
            else:
                return output
        elif self.task == 'peddet':
            outputs = self.predictor.forward_peddet(features['neck_output']['multi_scale_features'],
                                                    features['neck_output']['mask_features'])
            if self.training:
                # pedestrain detection target
                assert "instances" in features
                gt_instances = features["instances"]
                targets = self.prepare_detection_targets(gt_instances)

                # bipartite matching-based loss for object detection
                losses = self.loss(outputs, targets)
                for k in losses:
                    if 'loss' in k:
                        losses[k] = losses[k] * self.loss_weight
                return {'loss': losses, 'top1': losses['top1']}  #torch.FloatTensor([0]).cuda()} #losses['top1']}
            else:
                # pedestrain detection target
                processed_results = ped_det_postprocess(outputs, features['orig_size'])
                return processed_results
        else:
            raise NotImplementedError

    @staticmethod
    def semantic_inference_bce(mask_cls, mask_pred):
        mask_cls = mask_cls.sigmoid().reshape(-1)
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("q,qhw->qhw", mask_cls, mask_pred)
        # semseg = torch.einsum("kqc,kqhw->kchw", mask_cls, mask_pred)
        return semseg

    @staticmethod
    def semantic_inference(mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        # semseg = torch.einsum("kqc,kqhw->kchw", mask_cls, mask_pred)
        return semseg

    @staticmethod
    def semantic_inference_batch(mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        # semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        semseg = torch.einsum("kqc,kqhw->kchw", mask_cls, mask_pred)
        return semseg

    def pose_inference(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[NxKxHxW]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """
        if flip_pairs is not None:
            output_heatmap = flip_back(
                x.detach().cpu().numpy(),
                flip_pairs,
                target_type=self.loss.target_type)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):  # True
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = x.detach().cpu().numpy()
        return output_heatmap

    def pose_decode(self, img_metas, output, **kwargs):
        """Decode keypoints from heatmaps.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, H, W]): model predicted heatmaps.
        """
        batch_size = len(img_metas)

        if 'bbox_id' in img_metas[0].data:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i].data['center']
            s[i, :] = img_metas[i].data['scale']
            image_paths.append(img_metas[i].data['image_file'])

            if 'bbox_score' in img_metas[i].data:
                score[i] = np.array(img_metas[i].data['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i].data['bbox_id'])

        preds, maxvals = keypoints_from_heatmaps(
            output,
            c,
            s,
            unbiased=self.test_cfg.get('unbiased_decoding', False),
            post_process=self.test_cfg.get('post_process', 'default'),
            kernel=self.test_cfg.get('modulate_kernel', 11),
            valid_radius_factor=self.test_cfg.get('valid_radius_factor',
                                                  0.0546875),
            use_udp=self.test_cfg.get('use_udp', False),
            target_type=self.test_cfg.get('target_type', 'GaussianHeatMap'))

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}

        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        return result

def ped_det_postprocess(outputs, target_sizes):
    """ Perform the computation
    Parameters:
        outputs: raw outputs of the model
        target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                      For evaluation, this must be the original image size (before any data augmentation)
                      For visualization, this should be the image size after data augment, but before padding
    """
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
    assert len(out_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2

    prob = out_logits.sigmoid()
    # find the topk predictions
    num = out_logits.view(out_logits.shape[0], -1).shape[1]
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    return results


def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]
    return result


def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def _taylor(heatmap, coord):
    """Distribution aware coordinate decoding method.

    Note:
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmap (np.ndarray[H, W]): Heatmap of a particular joint type.
        coord (np.ndarray[2,]): Coordinates of the predicted keypoints.

    Returns:
        np.ndarray[2,]: Updated coordinates.
    """
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (
            heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
        dxy = 0.25 * (
            heatmap[py + 1][px + 1] - heatmap[py - 1][px + 1] -
            heatmap[py + 1][px - 1] + heatmap[py - 1][px - 1])
        dyy = 0.25 * (
            heatmap[py + 2 * 1][px] - 2 * heatmap[py][px] +
            heatmap[py - 2 * 1][px])
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy**2 != 0:
            hessianinv = np.linalg.inv(hessian)
            offset = -hessianinv @ derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


def post_dark_udp(coords, batch_heatmaps, kernel=3):
    """DARK post-pocessing. Implemented by udp. Paper ref: Huang et al. The
    Devil is in the Details: Delving into Unbiased Data Processing for Human
    Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
    Representation for Human Pose Estimation (CVPR 2020).

    Note:
        - batch size: B
        - num keypoints: K
        - num persons: N
        - height of heatmaps: H
        - width of heatmaps: W

        B=1 for bottom_up paradigm where all persons share the same heatmap.
        B=N for top_down paradigm where each person has its own heatmaps.

    Args:
        coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
        batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
        kernel (int): Gaussian kernel size (K) for modulation.

    Returns:
        np.ndarray([N, K, 2]): Refined coordinates.
    """
    if not isinstance(batch_heatmaps, np.ndarray):
        batch_heatmaps = batch_heatmaps.cpu().numpy()
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    assert (B == 1 or B == N)
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)

    batch_heatmaps_pad = np.pad(
        batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)),
        mode='edge').flatten()

    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + W + 2]
    ix1y1 = batch_heatmaps_pad[index + W + 3]
    ix1_y1_ = batch_heatmaps_pad[index - W - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - W]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()
    return coords


def _gaussian_blur(heatmaps, kernel=11):
    """Modulate heatmap distribution with Gaussian.
     sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
     sigma~=3 if k=17
     sigma=2 if k=11;
     sigma~=1.5 if k=7;
     sigma~=1 if k=3;

    Note:
        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([N, K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border),
                          dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps


def keypoints_from_heatmaps(heatmaps,
                            center,
                            scale,
                            unbiased=False,
                            post_process='default',
                            kernel=11,
                            valid_radius_factor=0.0546875,
                            use_udp=False,
                            target_type='GaussianHeatMap'):
    """Get final keypoint predictions from heatmaps and transform them back to
    the image.

    Note:
        batch size: N
        num keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.
        post_process (str/None): Choice of methods to post-process
            heatmaps. Currently supported: None, 'default', 'unbiased',
            'megvii'.
        unbiased (bool): Option to use unbiased decoding. Mutually
            exclusive with megvii.
            Note: this arg is deprecated and unbiased=True can be replaced
            by post_process='unbiased'
            Paper ref: Zhang et al. Distribution-Aware Coordinate
            Representation for Human Pose Estimation (CVPR 2020).
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.
        valid_radius_factor (float): The radius factor of the positive area
            in classification heatmap for UDP.
        use_udp (bool): Use unbiased data processing.
        target_type (str): 'GaussianHeatMap' or 'CombinedTarget'.
            GaussianHeatMap: Classification target with gaussian distribution.
            CombinedTarget: The combination of classification target
            (response map) and regression target (offset map).
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Returns:
        tuple: A tuple containing keypoint predictions and scores.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    # Avoid being affected
    heatmaps = heatmaps.copy()

    # detect conflicts
    if unbiased:
        assert post_process not in [False, None, 'megvii']
    if post_process in ['megvii', 'unbiased']:
        assert kernel > 0
    if use_udp:
        assert not post_process == 'megvii'

    # normalize configs
    if post_process is False:
        warnings.warn(
            'post_process=False is deprecated, '
            'please use post_process=None instead', DeprecationWarning)
        post_process = None
    elif post_process is True:
        if unbiased is True:
            warnings.warn(
                'post_process=True, unbiased=True is deprecated,'
                " please use post_process='unbiased' instead",
                DeprecationWarning)
            post_process = 'unbiased'
        else:
            warnings.warn(
                'post_process=True, unbiased=False is deprecated, '
                "please use post_process='default' instead",
                DeprecationWarning)
            post_process = 'default'
    elif post_process == 'default':
        if unbiased is True:
            warnings.warn(
                'unbiased=True is deprecated, please use '
                "post_process='unbiased' instead", DeprecationWarning)
            post_process = 'unbiased'

    # start processing
    if post_process == 'megvii':
        heatmaps = _gaussian_blur(heatmaps, kernel=kernel)

    N, K, H, W = heatmaps.shape
    if use_udp:
        assert target_type in ['GaussianHeatMap', 'CombinedTarget']
        if target_type == 'GaussianHeatMap':
            preds, maxvals = _get_max_preds(heatmaps)
            preds = post_dark_udp(preds, heatmaps, kernel=kernel)
        elif target_type == 'CombinedTarget':
            for person_heatmaps in heatmaps:
                for i, heatmap in enumerate(person_heatmaps):
                    kt = 2 * kernel + 1 if i % 3 == 0 else kernel
                    cv2.GaussianBlur(heatmap, (kt, kt), 0, heatmap)
            # valid radius is in direct proportion to the height of heatmap.
            valid_radius = valid_radius_factor * H
            offset_x = heatmaps[:, 1::3, :].flatten() * valid_radius
            offset_y = heatmaps[:, 2::3, :].flatten() * valid_radius
            heatmaps = heatmaps[:, ::3, :]
            preds, maxvals = _get_max_preds(heatmaps)
            index = preds[..., 0] + preds[..., 1] * W
            index += W * H * np.arange(0, N * K / 3)
            index = index.astype(np.int).reshape(N, K // 3, 1)
            preds += np.concatenate((offset_x[index], offset_y[index]), axis=2)
        else:
            raise ValueError('target_type should be either '
                             "'GaussianHeatMap' or 'CombinedTarget'")
    else:
        preds, maxvals = _get_max_preds(heatmaps)
        if post_process == 'unbiased':  # alleviate biased coordinate
            # apply Gaussian distribution modulation.
            heatmaps = np.log(
                np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
            for n in range(N):
                for k in range(K):
                    preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        elif post_process is not None:
            # add +/-0.25 shift to the predicted locations for higher acc.
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[n][k]
                    px = int(preds[n][k][0])
                    py = int(preds[n][k][1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array([
                            heatmap[py][px + 1] - heatmap[py][px - 1],
                            heatmap[py + 1][px] - heatmap[py - 1][px]
                        ])
                        preds[n][k] += np.sign(diff) * .25
                        if post_process == 'megvii':
                            preds[n][k] += 0.5

    # Transform back to the image
    for i in range(N):
        preds[i] = transform_preds(preds[i], center[i], scale[i], [W, H], use_udp=use_udp)

    if post_process == 'megvii':
        maxvals = maxvals / 255.0 + 0.5

    return preds, maxvals