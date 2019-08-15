'''
Created on Jun 27, 2018

@author: david
'''

import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions

import numpy as np

import dapnn.model_helpers as mh

class model(nn.Module):
    
    def __init__(self, name='model'):
        super(model, self).__init__()
        
        self.name = name
        
        # loss trackers
        self.register_buffer('train_batch_loss_y', torch.tensor([], requires_grad=False))
        self.register_buffer('train_batch_loss_x', torch.tensor([], requires_grad=False, dtype=torch.long))
        
        self.register_buffer('val_ckpt_loss_y', torch.tensor([], requires_grad=False))
        self.register_buffer('checkpoint_x', torch.tensor([], requires_grad=False, dtype=torch.long))
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_seed(self, seed):
        torch.manual_seed(seed)

    def set_name(self, name):
        self.name = name

    def reset_parameters(self):
        self.apply(mh.weight_reset)

    def load_state_dict(self, state_dict, strict=True):
        self.train_batch_loss_y.resize_(state_dict['train_batch_loss_y'].shape)
        self.train_batch_loss_x.resize_(state_dict['train_batch_loss_x'].shape)
        self.val_ckpt_loss_y.resize_(state_dict['val_ckpt_loss_y'].shape)
        self.checkpoint_x.resize_(state_dict['checkpoint_x'].shape)
        
        super().load_state_dict(state_dict, strict)

    def loss(self, batch): #abstract
        return NotImplemented

    def accuracy(self, batch): #abstract
        return NotImplemented
        

class sequential(model):
    
    # image_shape in form (C, W, H)
    def __init__(self, modules, criterion=None, in_batch_idx=0, out_batch_idx=1, weight_decay=0, name='sequential'):
        
        super(sequential, self).__init__(name=name)

        self.in_batch_idx = in_batch_idx
        self.out_batch_idx = out_batch_idx
        
        self.layers = nn.ModuleList()

        for i in range(0, len(modules)):
            self.layers.append(modules[i])

        # criterion
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=weight_decay)

    def forward(self, x):
        h = x
        for i in range(0, len(self.layers)):
            h = self.layers[i](h)
        return h
    
    def loss(self, batch):
        outputs = self.forward(batch[self.in_batch_idx])
        return self.criterion(outputs, batch[self.out_batch_idx])

    def accuracy(self, batch):
        output = self.forward(batch[self.in_batch_idx])
        correct = (output.argmax(dim=-1, keepdim=False) == batch[self.out_batch_idx].argmax(dim=-1, keepdim=False)).sum().item()
        total = torch.cumprod(torch.Tensor(list(batch[self.out_batch_idx].shape)), 0)[-2].item()
        return correct, total
