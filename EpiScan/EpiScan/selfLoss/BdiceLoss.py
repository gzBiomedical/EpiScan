import torch
from torch import nn
import numpy as np

class BinaryDiceLoss(nn.Module):
	def __init__(self):
		super(BinaryDiceLoss, self).__init__()
	
	def forward(self, input, targets):
		N = targets.size()[0]
		smooth = 1
		input_flat = input.view(N, -1)
		targets_flat = targets.view(N, -1)
	
		intersection = input_flat * targets_flat 
		N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
		loss = 1 - N_dice_eff.sum() / N
		return loss



#Dice系数
def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
 
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


#Dice numpy
def dice_coef_np(y_true, y_pred):
	smooth = 1.0
	y_true_f = y_true.flatten()
	y_pred_f = y_pred.flatten()
	intersection = np.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
