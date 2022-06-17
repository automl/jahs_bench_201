""" Taken as-is under extant license conditions from the original NAS-Bench-201 repo's
code at
https://github.com/D-X-Y/AutoDL-Projects/tree/bc4c4692589e8ee7d6bab02603e69f8e5bd05edc """

import math, torch
import torch.nn as nn
from torch.optim import Optimizer

class _LRScheduler(object):

  def __init__(self, optimizer, warmup_epochs, epochs):
    if not isinstance(optimizer, Optimizer):
      raise TypeError('{:} is not an Optimizer'.format(type(optimizer).__name__))
    self.optimizer = optimizer
    for group in optimizer.param_groups:
      group.setdefault('initial_lr', group['lr'])
    self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
    self.max_epochs = epochs
    self.warmup_epochs  = warmup_epochs
    self.current_epoch  = 0
    self.current_iter   = 0

  def extra_repr(self):
    return ''

  def __repr__(self):
    return ('{name}(warmup={warmup_epochs}, max-epoch={max_epochs}, current::epoch={current_epoch}, iter={current_iter:.2f}'.format(name=self.__class__.__name__, **self.__dict__)
              + ', {:})'.format(self.extra_repr()))

  def state_dict(self):
    return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)

  def get_lr(self):
    raise NotImplementedError

  def get_min_info(self):
    lrs = self.get_lr()
    return '#LR=[{:.6f}~{:.6f}] epoch={:03d}, iter={:4.2f}#'.format(min(lrs), max(lrs), self.current_epoch, self.current_iter)

  def get_min_lr(self):
    return min( self.get_lr() )

  def update(self, cur_epoch, cur_iter):
    if cur_epoch is not None:
      assert isinstance(cur_epoch, int) and cur_epoch>=0, 'invalid cur-epoch : {:}'.format(cur_epoch)
      self.current_epoch = cur_epoch
    if cur_iter is not None:
      assert isinstance(cur_iter, float) and cur_iter>=0, 'invalid cur-iter : {:}'.format(cur_iter)
      self.current_iter  = cur_iter
    for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
      param_group['lr'] = lr



class CosineAnnealingLR(_LRScheduler):

  def __init__(self, optimizer, warmup_epochs, epochs, T_max, eta_min):
    self.T_max = T_max
    self.eta_min = eta_min
    super(CosineAnnealingLR, self).__init__(optimizer, warmup_epochs, epochs)

  def extra_repr(self):
    return 'type={:}, T-max={:}, eta-min={:}'.format('cosine', self.T_max, self.eta_min)

  def get_lr(self):
    lrs = []
    for base_lr in self.base_lrs:
      if self.current_epoch >= self.warmup_epochs and self.current_epoch < self.max_epochs:
        last_epoch = self.current_epoch - self.warmup_epochs
        #if last_epoch < self.T_max:
        #if last_epoch < self.max_epochs:
        lr = self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * last_epoch / self.T_max)) / 2
        #else:
        #  lr = self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * (self.T_max-1.0) / self.T_max)) / 2
      elif self.current_epoch >= self.max_epochs:
        lr = self.eta_min
      else:
        lr = (self.current_epoch / self.warmup_epochs + self.current_iter / self.warmup_epochs) * base_lr
      lrs.append( lr )
    return lrs
