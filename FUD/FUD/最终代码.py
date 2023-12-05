import networkx as nx
import LBLD_AlgorithmFinal
from 论文综合代码.GCN import t1
import numpy as np
import 数据集引用
import time



# ------------------------ 数据集选择 ------------------------
dataset_name = 数据集引用.dataset_name
path = 数据集引用.path
G = 数据集引用.G
final_code_GRG = 数据集引用.final_code_GRG
# ------------------------ 社区划分 ------------------------
comm1 = LBLD_AlgorithmFinal.LBLD(dataset_name, path)
# comm1 = nx.community.louvain_communities(G)

# ------------------------ 打印社区 ------------------------
# print('LBLD社区划分结果如下')
# for index, c in enumerate(comm1.values()):
#     print(f'第{index + 1}个社区:{c}')
# print('Louvain社区划分结果如下：')
# for index, c in enumerate(comm2):
#     print(f'第{index + 1}个社区：{list(c)}')

# ------------------------ 求GR=度中心性值 ------------------------

def GR(G, com):
    dict_nbr = {}  # 存放节点的邻居
    for node in G.nodes():
        dict_nbr[node] = list(G.neighbors(node))
    remove_nodes = []  # 要删除的节点
    # 遍历社区
    if isinstance(com, list):
        for c in com:
            # 遍历社区里的节点
            for n in c:
                list1 = list(set(dict_nbr[n]) - set(c))
                if len(list1) == 0:
                    remove_nodes.append(n)
    else:
        com = list(com.values())
        for c in com:
            # 遍历社区里的节点
            for n in c:
                if n in list(G.nodes):
                    list1 = list(set(dict_nbr[n]) - set(c))
                    if len(list1) == 0:
                        remove_nodes.append(n)
    # print('需要删除的社区内部节点：', remove_nodes)
    for i in remove_nodes:
        G.remove_node(i)
    return G.degree()


remove_node_degree = dict(GR(final_code_GRG, comm1))
max_val = max(remove_node_degree.values())
min_val = min(remove_node_degree.values()) - 1
# 记录GR
GR_value = {}
GR_degree_centrality = {}
# 把每个节点的GR值都设置为0
for node in G.nodes():
    G.nodes[node]['GR'] = 0
    GR_value[node] = 0
    GR_degree_centrality[node] = 0
for k, v in remove_node_degree.items():
    G.nodes[k]['GR'] = (v - min_val) / (max_val - min_val)
    GR_value[k] = (v - min_val) / (max_val - min_val)
    GR_degree_centrality[k] = v / (len(final_code_GRG.nodes) - 1)

# ------------------------ 局部特征 ------------------------
# 计算每个节点的局部聚类系数
clustering_coefficients = nx.clustering(G)
# d=dict(G.degree)

# ------------------------ 全局特征 ------------------------
# 特征向量中心性
ec = nx.eigenvector_centrality_numpy(G)  # 字典类型
# ------------------------ 提取节点表征向量 ------------------------
input_dim = 100  # 输入特征维度
hidden_dims = [160, 320, 640]  # 隐藏层维度
output_dim = len(G.nodes)  # 输出特征维度，设置为节点数
reduction_dim = 300  # 降维的维度
num_epochs = 100  # 迭代次数
learning_rate = 0.001  # 学习率
# node_features_pca = GCN提取节点表征向量.extract_node_features(G, input_dim, hidden_dims, output_dim, reduction_dim,
#                                                               num_epochs, learning_rate)
node_features_pca = t1.node_representations.detach().numpy()
# node_features_pca = GCN一号.node_features
# ------------------------ Pearson correlation coefficient ------------------------
# 向量个数大于向量的维数，一定线性相关
corr = np.corrcoef(node_features_pca)
# ------------------------ Mahalanobis Distance Mahalanobis Distance ------------------------
# 协方差矩阵
cov_matrix = np.cov(np.array(node_features_pca).T)
# 计算马氏距离
n = len(node_features_pca)
mahalanobis_distances = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        diff = node_features_pca[i] - node_features_pca[j]
        mahalanobis_distance = np.sqrt(np.dot(np.dot(diff, np.linalg.inv(cov_matrix)), diff))
        mahalanobis_distances[i, j] = mahalanobis_distance


# 矩阵归一化
def normalize_matrix(matrix, min_value=0, max_value=1):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val) * (max_value - min_value) + min_value
    return normalized_matrix


# 马氏距离矩阵归一化
normalize_mahalanobis_distances = 2 - normalize_matrix(mahalanobis_distances)
corr_plus_mahalanobis = corr + normalize_mahalanobis_distances
# ------------------------ cmdis ------------------------
cp_dict = {}
for num in range(2):
    for cmnode in list(G.nodes):
        cmnode_nbr = list(G.neighbors(cmnode))
        cmdis = 0
        for cp in cmnode_nbr:
            cmdis = cmdis + corr_plus_mahalanobis[list(G.nodes).index(cmnode)][cmnode_nbr.index(cp)]
        cp_dict[cmnode] = cmdis

# ------------------------ 节点重叠影响 ------------------------
num = len(G.nodes())
for node in G.nodes:
    nodes = list(G.neighbors(node))
    oc = 0
    for nbr in nodes:
        nbrs = list(G.neighbors(nbr))
        intersection = len(set(nodes) & set(nbrs))
        union = len(set(nodes) | set(nbrs))
        similarity = 1-intersection / union
        oc = oc + similarity
    G.nodes[node]['oc'] = oc

for node in G.nodes:
    ooc = 0
    for nbr in G.neighbors(node):
        ooc = ooc + G.nodes[nbr]['oc']
    G.nodes[node]['ooc'] = ooc

overlap_dict = {}
max_value = 0
for node in G.nodes:
    oooc = 0
    for nbr in G.neighbors(node):
        oooc = oooc + G.nodes[nbr]['ooc']
    G.nodes[node][oooc] = oooc
    overlap_dict[node] = oooc
    if max_value <= oooc:
        max_value = oooc

max_value = int(max_value)
len_max_value = len(str(max_value))
num = pow(10, len_max_value - 4)
for key in overlap_dict.keys():
    overlap_dict[key] = overlap_dict[key] / num
# ------------------------ 综合特征 ------------------------
# 创建一个空字典来存放合并后的结果
merged_dict = {}
# 合并字典
for d in [GR_degree_centrality, clustering_coefficients, ec, cp_dict, overlap_dict]:
    for key, value in d.items():
        if key in merged_dict:
            # 如果键已经存在于合并字典中，将值添加到已存在的值列表中
            if isinstance(merged_dict[key], list):
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [merged_dict[key], value]
        else:
            # 如果键不存在于合并字典中，直接添加键值对
            merged_dict[key] = value

# print(merged_dict)


