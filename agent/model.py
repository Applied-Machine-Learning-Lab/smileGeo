# -*- coding: utf-8 -*-
# @Time : 2024/3/19 18:22
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: GLM
# @File : agents.py
# @Software: PyCharm

import torch
from torch import nn
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, N):
        super(GAT, self).__init__()
        self.N = N
        self.hidden_dim = 1024
        # self.X = nn.Embedding(self.N, 1024)
        self.X = nn.Parameter(torch.randn(size=(self.N, 1024)), requires_grad=True)
        self.W = nn.Parameter(torch.randn(size=(4096, self.hidden_dim)), requires_grad=True)
        self.a = nn.Parameter(torch.randn(size=(2*self.hidden_dim, 1)), requires_grad=True)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.linear = nn.Linear(self.N * self.hidden_dim, self.N+1)
    
    def forward(self, adj, img):
        x = img.unsqueeze(1).expand(-1, self.N, -1)
        comb_x = self.X + x
        final_x = comb_x.reshape(self.N, -1)

        h = torch.mm(final_x, self.W)

        e = torch.zeros(self.N, self.N).to(img.device)

        adj_indices = adj._indices()
        row, col = adj_indices[0], adj_indices[1]
        a_input = torch.cat((h[row], h[col]), dim=1)
        e_val = self.leakyrelu(torch.matmul(a_input, self.a).squeeze())
        
        e[row, col] = e_val
    
        attention = F.softmax(e, dim=1)

        # 计算X的最大值和最小值
        adj2 = adj.to_dense() * 2 
        adj_min = adj2.min()
        adj_max = adj2.max()

        # 执行Min-Max归一化
        adj_norm = (adj2 - adj_min) / (adj_max - adj_min)
        
        new_adj = attention * adj_norm

        h_prime = torch.mm(attention, h)

        out_flatten = h_prime.flatten()

        out = self.linear(out_flatten)
        # 计算最小值和最大值
        min_val = torch.min(out)
        max_val = torch.max(out)

        # 检查最大值和最小值是否相同，以避免除以零
        if max_val == min_val:
            # 如果最大值等于最小值，返回全0.5张量
            normalized_tensor = torch.full_like(out, 0.5)
        else:
            # 正常归一化处理
            normalized_tensor = (out - min_val) / (max_val - min_val)

        return new_adj, normalized_tensor