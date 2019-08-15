
import dapnn.models
import dapnn.model_helpers as mh
import torch
import torch.nn as nn
import numpy as np


class baseline(dapnn.models.model):

    # image_shape in form (C, W, H)
    def __init__(self, feature_extraction_modules, clas_layers, criterion=None, weight_decay=0, name='mdm_baseline'):

        super(baseline, self).__init__(name=name)

        self.feature_extraction_layers = nn.ModuleList()

        for i in range(0, len(feature_extraction_modules)):
            self.feature_extraction_layers.append(feature_extraction_modules[i])

        self.clas_layers = nn.ModuleList()

        for i in range(0, len(clas_layers)):
            self.clas_layers.append(clas_layers[i])

        self.sm = nn.LogSoftmax(-1)

        # criterion
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=weight_decay)

    def forward(self, x, num_chars):
        h = x
        for i in range(0, len(self.feature_extraction_layers)):
            h = self.feature_extraction_layers[i](h)

        h = h.reshape(h.shape[0], -1)

        char_probs = []
        for c in range(1, len(self.clas_layers) + 1):
            inputs = h[num_chars == c]
            logits = self.clas_layers[c-1](inputs).reshape(inputs.shape[0], c, 10)
            char_probs.append(self.sm(logits))

        return char_probs

    def loss(self, batch):
        outputs = self.forward(batch[0], batch[2].sum(dim=1))

        loss = 0
        for c in range(0, len(self.clas_layers)):
            char_probs = outputs[c]
            labels = batch[1][batch[2].sum(dim=1) == (c+1)][:, 0:(c+1), :]
            loss += self.criterion(char_probs, labels)

        return loss

    def accuracy(self, batch):
        outputs = self.forward(batch[0], batch[2].sum(dim=1))

        correct = 0.
        total = 0.

        for c in range(0, len(self.clas_layers)):
            char_probs = outputs[c]
            correct += (char_probs.argmax(dim=-1, keepdim=False) == batch[1][batch[2].sum(dim=1) == (c+1)][:, 0:(c+1), :].argmax(dim=-1, keepdim=False)).sum().item()
            total += torch.cumprod(torch.Tensor(list(char_probs.shape)), 0)[-2].item()

        return correct, total


class attn_test(dapnn.models.model):

    # image_shape in form (C, W, H)
    def __init__(self, feature_extraction_modules, criterion=None, weight_decay=0, name='attn_test'):

        super(attn_test, self).__init__(name=name)

        self.feature_extraction_layers = nn.ModuleList()

        for i in range(0, len(feature_extraction_modules)):
            self.feature_extraction_layers.append(feature_extraction_modules[i])

        self.pos_encoding_len = 10
        self.feature_size = 32

        self.query_generator = mh.pos1d(self.pos_encoding_len)
        #self.query_generator.plot_encoding(6)

        self.attn_module = mh.mq_attn(in_dim=self.feature_size+self.pos_encoding_len, query_dim=self.pos_encoding_len, out_dim=self.feature_size, key_dim=32, residual=False)

        self.linear = nn.Linear(in_features=self.feature_size, out_features=10)
        self.ReLU = nn.ReLU()

        self.sm = nn.LogSoftmax(-1)

        # criterion
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=weight_decay)

    def forward(self, x, num_chars):
        h = x
        for i in range(0, len(self.feature_extraction_layers)):
            h = self.feature_extraction_layers[i](h)

        h = h.reshape(h.shape[0], h.shape[1], h.shape[-1]).transpose(-2, -1)

        pos_encoding = self.query_generator(h.shape[1]).expand(x.shape[0], -1, -1).to(h.device)
        attn_input = torch.cat([h, pos_encoding], dim=-1)

        u_num_chars = torch.unique(num_chars)

        char_probs = []

        for nc in u_num_chars:
            nc_inputs = attn_input[num_chars == nc]
            querys = self.query_generator(int(nc.item())).expand(nc_inputs.shape[0], -1, -1).to(nc_inputs.device)
            attn_out = self.attn_module(nc_inputs, querys)
            attn_out = self.ReLU(attn_out)
            logits = self.linear(attn_out)
            char_probs.append(self.sm(logits))

        return char_probs

    def loss(self, batch):
        outputs = self.forward(batch[0], batch[2].sum(dim=1))

        loss = 0
        for nc in torch.unique(batch[2].sum(dim=1)):
            c = int(nc.item())
            char_probs = outputs[c - 1]
            labels = batch[1][batch[2].sum(dim=1) == c][:, 0:c, :]
            loss += self.criterion(char_probs, labels)

        return loss

    def accuracy(self, batch):
        outputs = self.forward(batch[0], batch[2].sum(dim=1))

        correct = 0.
        total = 0.

        for nc in torch.unique(batch[2].sum(dim=1)):
            c = int(nc.item())
            char_probs = outputs[c - 1]
            correct += (char_probs.argmax(dim=-1, keepdim=False) == batch[1][batch[2].sum(dim=1) == c][:, 0:c, :].argmax(dim=-1, keepdim=False)).sum().item()
            total += torch.cumprod(torch.Tensor(list(char_probs.shape)), 0)[-2].item()

        return correct, total