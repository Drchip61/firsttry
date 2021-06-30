# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 12:57:49 2021

@author: 严天宇
"""

def diceCoeff(pred, gt, smooth=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """
 
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")
 
    pred = activation_fn(pred)
 
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
 
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)
    return loss
pred = torch.rand(1,1,4,4)
gt = torch.ones(1,1,4,4)
dice_baldder1 = diceCoeff(pred[:, 0:1, :], gt[:, 0:1, :], smooth=1, activation=None)
dice_baldder2 = diceCoeff(pred[:, 0:1, :], gt[:, 0:1, :], smooth=1e-5, activation=None)
dice_baldder2 = diceCoeff(pred, gt, smooth=1e-5, activation='sigmoid')
print('smooth=1 : dice={:.4}'.format(dice_baldder1.item()))
print('smooth=1e-5 : dice={:.4}'.format(dice_baldder2.item()))
