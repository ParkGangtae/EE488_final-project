import math
import torch
from torch import nn
import torch.nn.functional as F

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, s=64.0, m=0.2, **kwargs):
        super(LossFunction, self).__init__()
        
        self.s = s  # Scaling factor
        self.m = m  # Margin
        self.fc 		= nn.Linear(nOut,nClasses)
        self.weight = nn.Parameter(torch.Tensor(nClasses, nOut))
        nn.init.xavier_uniform_(self.weight)

        self.criterion = nn.CrossEntropyLoss()

        # Precompute constants
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        print('Initialized Combined loss')

    def forward(self, x, label):

        out_x = self.fc(x)
        sft_loss = self.criterion(out_x, label)
        # sft_loss = sft_loss / sft_loss.mean().detach()

        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(1e-7, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        
        arcface_loss = self.criterion(logits, label)
        # arcface_loss = arcface_loss / arcface_loss.mean().detach()

        loss = 0.2*sft_loss + 0.8*arcface_loss

        return loss
