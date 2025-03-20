import os
from random import shuffle
from turtle import forward
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
class Expert(nn.Module):
    def __init__(self, n_kernel, window, n_multiv, hidden_size, output_size, drop_out):
        super(Expert, self).__init__()
        self.conv1 = nn.Conv2d(1, n_kernel, (window, 1))
        self.conv2 = nn.Conv2d(1, n_kernel, (window, 1))
        # self.LKA = LKA(1)
        # self.conv1 = Inception_dilated_Conv(n_kernel,window)
        # self.conv2 = Inception_dilated_Conv(n_kernel,window)
        self.gate = Sigmoid()
        self.dropout = nn.Dropout(drop_out)
        self.fc1 = nn.Linear(n_kernel * n_multiv, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(dim=1).contiguous()
        # x = self.LKA(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1 = self.gate(x1)
        x2 = self.relu(x2)
        x = x1*x2
        # x = F.relu(self.conv(x))
        x = self.dropout(x)

        out = torch.flatten(x, start_dim=1).contiguous()

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out
class  Inception_dilated_Conv(nn.Module):
    def __init__(self,n_kernel,window):
        super(Inception_dilated_Conv,self).__init__()
        self.conv1 = nn.Conv2d(1, n_kernel , (14, 1), dilation=3,padding=0)
        self.conv2 = nn.Conv2d(1, n_kernel , (14, 1), dilation=4,padding=0)
        self.conv3 = nn.Conv2d(1, n_kernel , (14, 1),dilation=1)

    def forward(self,x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out =out1+out2+out3
        return out

class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv1d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv1d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv1d(dim, dim, 1)


    def forward(self, x):
        x = x.permute(0,2,1)
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        attn = u * attn
        return attn.permute(0, 2, 1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn
# class Inception(nn.Module):
#     def __init__(self,n_kernel,window):
#         super(Inception,self).__init__()
#         self.incep1=Inception_dilated_Conv(n_kernel,window)
#         self.incep2=Inception_dilated_Conv(n_kernel,window)
#         self.relu=nn.Relu()
#         self.gate=nn.
class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)






class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, drop_out):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(drop_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class MMoE(pl.LightningModule):
    def __init__(self):
        super(MMoE, self).__init__()
        # self.hp = hparams
        self.sg_ratio = 0.7
        # self.seed = seed
        self.n_multiv = 128
        self.n_kernel = 16
        self.window = 20
        self.num_experts = 5
        self.experts_out = 55
        self.experts_hidden = 256
        self.towers_hidden = 32

        # task num = n_multiv
        self.tasks = 128
        # self.criterion = hparams.criterion
        self.exp_dropout = 0.1
        self.tow_dropout = 0.1
        self.conv_dropout = 0.1
        self.lr = 0.0001

        self.softmax = nn.Softmax(dim=1)

        self.experts = nn.ModuleList(
            [Expert(self.n_kernel, self.window, self.n_multiv, self.experts_hidden, self.experts_out, self.exp_dropout) \
             for i in range(self.num_experts)])
        self.w_gates = nn.ParameterList([nn.Parameter(torch.randn(self.window, self.num_experts), requires_grad=True) \
                                         for i in range(self.tasks)])
        self.share_gate = nn.Parameter(torch.randn(self.window, self.num_experts), requires_grad=True)
        self.towers = nn.ModuleList([Tower(self.experts_out, 1, self.towers_hidden, self.tow_dropout) \
                                     for i in range(self.tasks)])

    def forward(self, x):
        experts_out = [e(x) for e in self.experts]
        experts_out_tensor = torch.stack(experts_out)

        gates_out = [self.softmax(
            (x[:, :, i] @ self.w_gates[i]) * (1 - self.sg_ratio) + (x[:, :, i] @ self.share_gate) * self.sg_ratio)
                     for i in range(self.tasks)]
        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_out_tensor for g in gates_out]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]

        tower_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        tower_output = torch.stack(tower_output, dim=0).permute(1, 2, 0)

        final_output = tower_output
        return final_output