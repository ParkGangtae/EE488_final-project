#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, test_interval, max_epoch, **kwargs):

	sche_fn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=test_interval, eta_min=0.0001)

	lr_step = 'epoch'

	print('Initialised Cosine Annealing LR scheduler')

	return sche_fn, lr_step