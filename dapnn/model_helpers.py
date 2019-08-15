'''
Created on Jun 29, 2018

@author: david
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
import matplotlib.pyplot as plt


def weight_reset(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, self_attn):
        m.reset_parameters()


class residual(nn.Module):

    def __init__(self, module):
        super(residual, self).__init__()
        self.module = module
        self.module.apply(weight_reset)

    def forward(self, x):
        return x + self.module(x)


class reshape(nn.Module):
    
    def __init__(self, size):
        super(reshape, self).__init__()
        self.size = size
        
    def forward(self, x):
        return x.reshape(self.size)


class ReLU255(nn.Module):   
    
    def __init__(self):
        super(ReLU255, self).__init__()
        
    def forward(self, x):
        return torch.clamp(x,0,255)


class pad2d(nn.Module):

    def __init__(self, kernel_size, dilation):
        
        super(pad2d, self).__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        
    def forward(self, x):


        pad_along_height = (self.kernel_size + (self.kernel_size-1) * (self.dilation-1) - 1)
        pad_along_width = (self.kernel_size + (self.kernel_size-1) * (self.dilation-1) - 1)

        pad_top = int(pad_along_height // 2)
        pad_bottom = int(pad_along_height - pad_top)
        pad_left = int(pad_along_width // 2)
        pad_right = int(pad_along_width - pad_left)
    
        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

class pad1d(nn.Module):

    def __init__(self, kernel_size, dilation):
        
        super(pad1d, self).__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        
    def forward(self, x):

        pad_along_width = (self.kernel_size + (self.kernel_size-1) * (self.dilation-1) - 1)

        pad_left = int(pad_along_width // 2)
        pad_right = int(pad_along_width - pad_left)
    
        return F.pad(x, (pad_left, pad_right))


class pos1d(nn.Module):

    def __init__(self, pos_size):
        super().__init__()

        self.pos_size = pos_size

    def forward(self, num_samples):
        half_step_size = np.pi / num_samples
        samples = torch.linspace(half_step_size, 2 * np.pi - half_step_size, num_samples)

        all_samples = []

        for i in range(1, self.pos_size + 1):
            all_samples.append(samples * ((i + 1) / 2.))

        outputs = torch.sin(torch.stack(all_samples)).transpose(1, 0)

        return outputs

    def plot_encoding(self, num_samples):
        half_step_size = np.pi / num_samples
        samples = torch.linspace(half_step_size, 2 * np.pi - half_step_size, num_samples)

        over_samples = torch.linspace(0, 2 * np.pi, 1000)

        for i in range(1, self.pos_size + 1):
            plt.subplot(self.pos_size, 1, i)
            plt.plot(over_samples.numpy(), torch.sin(over_samples * ((i + 1) / 2.)).numpy())
            print(torch.sin(samples * ((i + 1) / 2.)).numpy())
            for sample in samples:
                plt.axvline(sample.numpy(), color='red')

        plt.show()


class attention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, querys, keys, values):

        pre_weights = torch.einsum('bxj,byj->bxy', (querys, keys)) / math.sqrt(float(keys.shape[-1]))
        weights = F.softmax(pre_weights, dim=2)

        outputs = torch.einsum('bey,byj->bej', (weights, values))

        return outputs


class mq_attn(nn.Module):

    def reset_parameters(self):
        q_stdv = 1. / math.sqrt(self.query_proj.size(-1))
        self.query_proj.data.uniform_(-q_stdv, q_stdv)

        k_stdv = 1. / math.sqrt(self.key_proj.size(-1))
        self.key_proj.data.uniform_(-k_stdv, k_stdv)

        v_stdv = 1. / math.sqrt(self.value_proj.size(-1))
        self.value_proj.data.uniform_(-v_stdv, v_stdv)

    def __init__(self, in_dim, query_dim, out_dim, key_dim, residual=False):
        super().__init__()

        self.residual = residual

        self.query_proj = Parameter(torch.Tensor(query_dim, key_dim))
        self.key_proj = Parameter(torch.Tensor(in_dim, key_dim))
        self.value_proj = Parameter(torch.Tensor(in_dim, out_dim))

        self.attn = attention()

        self.reset_parameters()

    def forward(self, x, querys):  # input should be size (batch_cnt, entity_cnt, in_dim)

        # we get a query + key + value for each entity
        querys = torch.einsum('bei,ij->bej', (querys, self.query_proj))
        keys = torch.einsum('bei,ij->bej', (x, self.key_proj))
        values = torch.einsum('bei,ij->bej', (x, self.value_proj))

        outputs = self.attn(querys, keys, values)

        if self.residual:
            outputs += x

        return outputs


class self_attn(nn.Module):
    
    def reset_parameters(self):
        q_stdv = 1. / math.sqrt(self.query_proj.size(-1))
        self.query_proj.data.uniform_(-q_stdv, q_stdv)
        
        k_stdv = 1. / math.sqrt(self.key_proj.size(-1))
        self.key_proj.data.uniform_(-k_stdv, k_stdv)
        
        v_stdv = 1. / math.sqrt(self.value_proj.size(-1))
        self.value_proj.data.uniform_(-v_stdv, v_stdv)

    def __init__(self, in_dim, out_dim, key_dim, residual=False):
        
        super().__init__()
        
        self.residual = residual

        self.query_proj = Parameter(torch.Tensor(in_dim, key_dim))
        self.key_proj = Parameter(torch.Tensor(in_dim, key_dim))
        self.value_proj = Parameter(torch.Tensor(in_dim, out_dim))

        self.attn = attention()

        self.reset_parameters()

    def forward(self, x): # input should be size (batch_cnt, entity_cnt, in_dim)

        # we get a query + key + value for each entity
        querys = torch.einsum('bei,ij->bej', (x, self.query_proj))
        keys = torch.einsum('bei,ij->bej', (x, self.key_proj))
        values = torch.einsum('bei,ij->bej', (x, self.value_proj))

        outputs = self.attn(querys, keys, values)
        
        if self.residual:
            outputs += x
        
        return outputs

