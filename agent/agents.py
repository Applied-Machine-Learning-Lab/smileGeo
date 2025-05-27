# -*- coding: utf-8 -*-
# @Time : 2024/3/17 18:22
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: GLM
# @File : agents.py
# @Software: PyCharm
import os.path
import random
import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from PIL import Image
import torchvision.transforms as transforms
from agent.model import GAT
from diffusers.models import AutoencoderKL
from utils.my_utils import sparse_mx_to_torch_sparse_tensor
import torch.nn.functional as F
import torch

class Agents():
    def __init__(self, args, logger, adj=None, mode="train", save_path='saved', load_checkpoint=False):
        self.G = nx.Graph()
        self.logger = logger
        self.link_threshold = args.link_threshold
        self.device = args.device
        self.last = {}
        if adj is None:
            df = pd.read_csv(os.path.join('agent','agent_settings.csv'), sep='#')
            num_nodes = len(df)
            # Create a fully connected graph with weights 0.5
            for i in range(num_nodes):
                self.last[i] = {}
                for j in range(num_nodes):
                    if i != j:
                        self.G.add_edge(i, j, weight=0.5)
                        self.last[i][j] = 'None'
            self.logger.log('Graph initialized: '+str(self.G))
        else:
            # 使用稀疏矩阵构建图
            adj_matrix = csr_matrix(adj)  # 确保adj_matrix是一个稀疏矩阵
            adj_matrix_coo = adj_matrix.tocoo()  # 转换为COO格式，方便迭代
            for i, j, v in zip(adj_matrix_coo.row, adj_matrix_coo.col, adj_matrix_coo.data):
                if i not in self.last.keys():
                    self.last[i] = {}
                self.last[i][j] = 'None'
                self.G.add_edge(i, j, weight=v)
        self.model = GAT(self.get_Nnodes())
        if self.device != torch.device('cpu'):
            self.model = self.model.to(self.device)
        if mode == "inference":
            self.model_load(save_path)
            self.model.eval()
        else:
            if load_checkpoint:
                self.model_load('graph_tmp_data')
            self.model.train()
        self.logger.log(f"model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.image = None
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 调整图片大小为512x512
            transforms.ToTensor()  # 转换为Tensor
        ])
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=args.stepsize, gamma=args.gamma)
        self.exp_dec_cnt = 1
    
    def update_graph(self, adj):
        try:
            adj_np = adj.umpy()
            self.G = nx.from_numpy_array(adj_np)
            return True
        except Exception as e:
            self.logger.log(e, 'error')
            return False

    def set_image(self, img):
        try:
            image = self.transform(img).unsqueeze(0).to(self.device)
            # print('image shape:', image.shape)
            self.image = self.vae.encode(image).latent_dist.sample().mul_(0.18215)
            # print('self.image shape:', self.image.shape)
            self.image = self.image.squeeze().view(4, -1)
            # print('self.image2 shape:', self.image.shape)
            return True
        except Exception as e:
            self.logger.log(e, 'error')
            return False

    def get_neighbors_with_max_weight_paths(self, start_node, max_hops=3):
        """
        获取从start_node出发，最多max_hops跳内的所有节点，
        以及到达这些节点的路径上权重乘积的最大值。
        """
        def dfs(node, hops, weight_product):
            if hops > max_hops:
                return
            # 更新到达当前节点的最大权重乘积
            if node != start_node:
                if node in max_weights:
                    max_weights[node] = max(max_weights[node], weight_product)
                else:
                    max_weights[node] = weight_product
            for neighbor in self.G[node]:
                edge_weight = self.G[node][neighbor]['weight']
                dfs(neighbor, hops + 1, weight_product * edge_weight)

        max_weights = {}
        dfs(start_node, 0, 1)  # 从start_node开始，权重乘积初始为1
        return max_weights
    
    def get_neighbors_with_random_walk(self, start_node, k=3):
        path = [start_node]
        visited = set(path)
        current_node = start_node
        for _ in range(k-1):  # 减1因为起始节点已在路径中
            edges = self.G.edges(current_node, data=True)
            if not edges:
                break  # 如果没有更多的边，结束游走
            valid_edges = [(n, attr['weight']) for _, n, attr in edges if n not in visited]
            if not valid_edges:
                break  # 如果所有可能的节点都已经被访问过，结束游走
            total_weight = sum(weight for _, weight in valid_edges)
            cumulative_weights = [sum(weight for _, weight in valid_edges[:i+1])/total_weight for i in range(len(valid_edges))]
            rand_val = random.random()
            for i, (neighbor, _) in enumerate(valid_edges):
                if rand_val <= cumulative_weights[i]:
                    current_node = neighbor
                    if current_node not in path:  # 检查是否已经记录在路径中
                        path.append(current_node)
                    visited.add(current_node)  # 添加到已访问集合
                    break
        if len(path) > 1:
            return path[1:]
        else:
            return path

    def get_Nnodes(self):
        return self.G.number_of_nodes()

    def get_adj(self):
        return nx.adjacency_matrix(self.G)

    def enhance_link(self, i, j):
        if i == j:
            return False
        if self.last[i][j] == 'enhance':
            self.exp_dec_cnt += 1
        else:
            self.exp_dec_cnt = 1
            self.last[i][j] = 'enhance'
        try:
            self.G[i][j]['weight'] = 1 / (2 * self.exp_dec_cnt) + self.G[i][j]['weight'] * (2 * self.exp_dec_cnt - 1) / (2 * self.exp_dec_cnt)
            return True
        except Exception as e:
            self.logger.log(e, 'error')
            return False

    def weaken_link(self, i, j):
        if i == j:
            return False
        if self.last[i][j] == 'weaken':
            self.exp_dec_cnt += 1
        else:
            self.exp_dec_cnt = 1
            self.last[i][j] = 'weaken'
        try:
            self.G[i][j]['weight'] = self.G[i][j]['weight'] * (2 * self.exp_dec_cnt - 1) / (2 * self.exp_dec_cnt)
            if self.G[i][j]['weight'] <= self.link_threshold:
                self.G.remove_edge(i, j)
            return True
        except Exception as e:
            self.logger.log('Edge not exists! Skip ...', 'error')
            return False

    def social_network_inference(self, adj_pre):
        adj_pre = sparse_mx_to_torch_sparse_tensor(adj_pre).to(self.device)
        with torch.no_grad():
            adj_next, soft_one_hot = self.model.forward(adj_pre, self.image)
        self.update_graph(adj_next)
        return True
        
    
    def social_network_training(self, adj_pre, adj_next, select_agent):
        flag = 0.0
        for ind in select_agent:
            if ind == 1:
                break
            else:
                flag = 1.0
        select_agent.append(flag)
        adj_pre = sparse_mx_to_torch_sparse_tensor(adj_pre).to(self.device)
        adj_next = sparse_mx_to_torch_sparse_tensor(adj_next)
        torch_select_agent = torch.tensor(select_agent).to(self.device)
        self.logger.log('select agent: '+str(torch_select_agent.tolist()), 'debug')
        dense_tensor_adj_next = adj_next.to_dense().to(self.device)
        self.model.train()
        out, soft_one_hot = self.model.forward(adj_pre, self.image)
        # self.logger.log('soft_one_hot shape: '+str(soft_one_hot.shape), 'debug')
        loss = F.mse_loss(out, dense_tensor_adj_next) + F.mse_loss(soft_one_hot, torch_select_agent)  #F.cosine_similarity(soft_one_hot.unsqueeze(0), torch_select_agent.unsqueeze(0), dim=1)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.scheduler.step()
        
        self.logger.log('loss: '+str(loss.item()), 'debug')
        self.logger.log(str(soft_one_hot)+'\t'+str(torch_select_agent), 'debug')
        return
    
    def model_save(self, path):
        torch.save(self.model.state_dict(), os.path.join(path,'model.pth'))
    
    def model_load(self, path):
        state_dict=torch.load(os.path.join(path,'model.pth'), map_location=self.device)
        self.model.load_state_dict(state_dict)