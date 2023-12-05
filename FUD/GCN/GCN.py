import numpy as np
import networkx as nx
import torch
import 数据集引用

# 生成一个随机无向图
# G = nx.karate_club_graph()
G = 数据集引用.G

# 获取节点数量
num_nodes = len(G.nodes)

# 计算邻接矩阵
adj_matrix = nx.adjacency_matrix(G,dtype=float)
adj_matrix = adj_matrix.toarray()+np.identity(len(G.nodes))

# 计算度矩阵
degree_matrix = np.diag(np.sum(adj_matrix, axis=1))

# 将邻接矩阵和度矩阵转换为PyTorch张量
adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
degree_matrix = torch.tensor(degree_matrix, dtype=torch.float32)

# 创建GCN模型
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = torch.matmul(adj, x)  # GCN操作
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class GCNResidual(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNResidual, self).__init__()
        self.gcn1 = GCNLayer(in_features, out_features)
        self.gcn2 = GCNLayer(out_features, out_features)

    def forward(self, x, adj):
        residual = x
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        x += residual  # 残差连接
        return x

class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes,):
        super(GraphConvolutionalNetwork, self).__init__()
        self.gcn1 = GCNLayer(num_features, hidden_dim)
        self.gcn2 = GCNResidual(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, adj, degree_matrix):
        x = degree_matrix  # 使用度矩阵代替特征矩阵
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        x = self.fc(x)
        return x

# 定义模型参数
num_features = num_nodes
hidden_dim = 32
num_classes = 16

# 创建模型实例
model = GraphConvolutionalNetwork(num_features, hidden_dim, num_classes)

# 进行前向传播
node_representations = model(adj_matrix, degree_matrix)

# tensor转换成数组
# print(node_representations.detach().numpy())


