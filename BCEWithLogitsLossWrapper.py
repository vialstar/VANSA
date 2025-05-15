import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .Loss import Loss


class BCEWithLogitsLossWrapper(Loss):
    
    def __init__(self, adv_temperature=None, alpha=0.5):
        super(BCEWithLogitsLossWrapper, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.adv_flag = False
        if adv_temperature is not None:
            self.adv_temperature = nn.Parameter(torch.tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        self.alpha = alpha

    def get_weights(self, n_score):
        """计算负样本的权重"""
        return F.softmax(n_score * self.adv_temperature, dim=-1).detach()

    def compute_loss(self, scores, targets):
        """计算单侧损失"""
        return self.criterion(scores, targets)

    def dynamic_combined_loss(self, real_output, fake_output, epoch, max_epochs):
        """动态组合损失"""
        weight = (epoch / max_epochs) * self.alpha
        mse_loss = F.mse_loss(fake_output, real_output)
        l1_loss = F.l1_loss(fake_output, real_output)
        return weight * mse_loss + (1 - weight) * l1_loss

    def forward(self, p_score, n_score, epoch=None, max_epochs=None):
        """前向传播，计算总损失"""
        # 正样本的目标标签为1，负样本的目标标签为0
        pos_targets = torch.ones_like(p_score)
        neg_targets = torch.zeros_like(n_score)

        pos_loss = self.compute_loss(p_score, pos_targets)
        neg_loss = self.compute_loss(n_score, neg_targets)

        if self.adv_flag:
            weights = self.get_weights(n_score)
            neg_loss = (weights * neg_loss).sum(dim=-1).mean()

        if epoch is not None and max_epochs is not None:
            neg_loss += self.dynamic_combined_loss(p_score, n_score, epoch, max_epochs)

        total_loss = (pos_loss + neg_loss) / 2
        return total_loss

    def predict(self, p_score, n_score, epoch=None, max_epochs=None):
        """预测，计算损失并返回NumPy数组"""
        score = self.forward(p_score, n_score, epoch, max_epochs)
        return score.cpu().data.numpy()