#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, s=64.0, m=0.2, **kwargs):
        super(LossFunction, self).__init__()
        
        self.fc = nn.Linear(nOut, nClasses)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.s = s # scaling factor
        self.m = m # margin

        print('Initialized ArcFace loss')

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.fc.weight))
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))

        target_logit = torch.cos(theta + self.m)
        target_logit = target_logit.gather(1, label.view(-1, 1))

        logits = cosine.clone()
        logits.scatter_(1, label.view(-1, 1), target_logit)

        logits *= self.s
        loss = self.criterion(logits, label)
        return loss