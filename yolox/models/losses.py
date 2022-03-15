#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from loguru import logger
try:
    import spring.linklink as link
except:   # noqa
    link = None

class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        iou = (area_i) / (area_p + area_g - area_i + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"   
    "can be use" 
    def __init__(self, reduction, alpha=.25, gamma=2):
            super(FocalLoss, self).__init__()        
            self.alpha = alpha     
            self.gamma = gamma
            self.reduction=reduction
            
    def forward(self, inputs, targets): #(num_anchor,9) (num_anchor,9)
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class DistBackend():
    def __init__(self):
        self.backend = 'linklink'

DIST_BACKEND = DistBackend()

def allreduce(*args, **kwargs):
    if DIST_BACKEND.backend == 'linklink':
        return link.allreduce(*args, **kwargs)
    elif DIST_BACKEND.backend == 'dist':
        return dist.all_reduce(*args, **kwargs)
    else:
        raise NotImplementedError

def _reduce(loss, reduction, **kwargs):
    if reduction == 'none':
        ret = loss
    elif reduction == 'mean':
        normalizer = loss.numel()
        if kwargs.get('normalizer', None):
            normalizer = kwargs['normalizer']
        ret = loss.sum() / normalizer
    elif reduction == 'sum':
        ret = loss.sum()
    else:
        raise ValueError(reduction + ' is not valid')
    return ret


class BaseLoss(_Loss):
    # do not use syntax like `super(xxx, self).__init__,
    # which will cause infinited recursion while using class decorator`
    def __init__(self,
                 name='base',
                 reduction='none',
                 loss_weight=1.0):
        r"""
        Arguments:
            - name (:obj:`str`): name of the loss function
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
        """
        _Loss.__init__(self, reduction=reduction)
        self.loss_weight = loss_weight
        self.name = name

    def __call__(self, input, target, reduction_override=None, normalizer_override=None, **kwargs):
        r"""
        Arguments:
            - input (:obj:`Tensor`)
            - reduction (:obj:`Tensor`)
            - reduction_override (:obj:`str`): choice of 'none', 'mean', 'sum', override the reduction type
            defined in __init__ function
            - normalizer_override (:obj:`float`): override the normalizer when reduction is 'mean'
        """
        reduction = reduction_override if reduction_override else self.reduction
        assert (normalizer_override is None or reduction == 'mean'), \
            f'normalizer is not allowed when reduction is {reduction}'
        loss = _Loss.__call__(self, input, target, reduction, normalizer=normalizer_override, **kwargs)
        return loss * self.loss_weight

    def forward(self, input, target, reduction, normalizer=None, **kwargs):
        raise NotImplementedError


class GeneralizedCrossEntropyLoss(BaseLoss):
    def __init__(self,
                 name='generalized_cross_entropy_loss',
                 reduction='none',
                 loss_weight=1.0,
                 activation_type='softmax',
                 ignore_index=-1,):
        BaseLoss.__init__(self,
                          name=name,
                          reduction=reduction,
                          loss_weight=loss_weight)
        self.activation_type = activation_type
        self.ignore_index = ignore_index


class EqualizedFocalLoss(GeneralizedCrossEntropyLoss):
    def __init__(self,
                 name='equalized_focal_loss',
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=-1,
                 num_classes=1204,
                 focal_gamma=2.0,
                 focal_alpha=0.25,
                 scale_factor=8.0,
                 fpn_levels=5):
        activation_type = 'sigmoid'
        GeneralizedCrossEntropyLoss.__init__(self,
                                             name=name,
                                             reduction=reduction,
                                             loss_weight=loss_weight,
                                             activation_type=activation_type,
                                             ignore_index=ignore_index)

        # Focal Loss的超参数
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # ignore bg class and ignore idx
        self.num_classes = num_classes - 1

        # EFL损失函数的超参数
        self.scale_factor = scale_factor
        # 初始化正负样本的梯度变量
        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))
        # 初始化正负样本变量
        self.register_buffer('pos_neg', torch.ones(self.num_classes))

        # grad collect
        self.grad_buffer = []
        self.fpn_levels = fpn_levels

        logger.info("build EqualizedFocalLoss, focal_alpha: {focal_alpha}, focal_gamma: {focal_gamma},scale_factor: {scale_factor}")

    def forward(self, input, target, reduction, normalizer=None):
        self.n_c = input.shape[-1]
        self.input = input.reshape(-1, self.n_c)
        self.target = target.reshape(-1)
        self.n_i, _ = self.input.size()

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c + 1)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target[:, 1:]

        expand_target = expand_label(self.input, self.target)
        sample_mask = (self.target != self.ignore_index)

        inputs = self.input[sample_mask]
        targets = expand_target[sample_mask]
        self.cache_mask = sample_mask
        self.cache_target = expand_target

        pred = torch.sigmoid(inputs)
        pred_t = pred * targets + (1 - pred) * (1 - targets)
  # map_val为：1-g^j
        map_val = 1 - self.pos_neg.detach()
        # dy_gamma为：gamma^j
        dy_gamma = self.focal_gamma + self.scale_factor * map_val
        
        # focusing factor
        ff = dy_gamma.view(1, -1).expand(self.n_i, self.n_c)[sample_mask]
        
        # weighting factor
        wf = ff / self.focal_gamma

        # ce_loss
        ce_loss = -torch.log(pred_t)
        cls_loss = ce_loss * torch.pow((1 - pred_t), ff.detach()) * wf.detach()

        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
            cls_loss = alpha_t * cls_loss

        if normalizer is None:
            normalizer = 1.0

        return _reduce(cls_loss, reduction, normalizer=normalizer)
    
 # 收集梯度，用于梯度引导的机制
    def collect_grad(self, grad_in):
        bs = grad_in.shape[0]
        self.grad_buffer.append(grad_in.detach().permute(0, 2, 3, 1).reshape(bs, -1, self.num_classes))
        if len(self.grad_buffer) == self.fpn_levels:
            target = self.cache_target[self.cache_mask]
            grad = torch.cat(self.grad_buffer[::-1], dim=1).reshape(-1, self.num_classes)

            grad = torch.abs(grad)[self.cache_mask]
            pos_grad = torch.sum(grad * target, dim=0)
            neg_grad = torch.sum(grad * (1 - target), dim=0)

            allreduce(pos_grad)
            allreduce(neg_grad)
   # 正样本的梯度
            self.pos_grad += pos_grad
            # 负样本的梯度
            self.neg_grad += neg_grad
            # self.pos_neg=g_j:表示第j类正样本与负样本的累积梯度比
            self.pos_neg = torch.clamp(self.pos_grad / (self.neg_grad + 1e-10), min=0, max=1)
            self.grad_buffer = []

