import networkx as nx
import 数据集引用
import time


# --------------------------------- 数据集 ---------------------------------
G = 数据集引用.G
CIFR_GRG = 数据集引用.CIFR_GRG

# --------------------------------- 社区发现 ---------------------------------
all_nodes = len(G.nodes)
comms = nx.community.louvain_communities(G)

# --------------------------------- 排名 ---------------------------------
def rank_dict(dict,rank,temp):
    for key in dict.keys():
        if dict[key] > temp:
            temp = dict[key]
            dict[key] = rank
        elif dict[key] < temp:
            temp=dict[key]
            rank = rank + 1
            dict[key] = rank
        else:
            dict[key] = rank
    return dict


# --------------------------------- LC ---------------------------------
LC = nx.betweenness_centrality(G)  # dict
for c in comms:
    c_len = len(c)
    Ac = c_len / all_nodes
    for node in c:
        LC[node] = LC[node] * Ac
order_LC = sorted(LC.items(), key=lambda x: x[1], reverse=True)  # list类型
order_LC = dict(order_LC)


# --------------------------------- LR ---------------------------------

def GR_value(G, com):
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
                list1 = list(set(dict_nbr[n]) - set(c))
                if len(list1) == 0:
                    remove_nodes.append(n)
    # print('需要删除的社区内部节点：', remove_nodes)
    for i in remove_nodes:
        G.remove_node(i)
    return G


CIFR_GRG = GR_value(CIFR_GRG, comms)
GR = nx.degree_centrality(CIFR_GRG)
for i in list(G.nodes):
    if i not in GR.keys():
        GR[i] = 0
order_GR = sorted(GR.items(), key=lambda x: x[1], reverse=True)  # list类型
order_GR = dict(order_GR)

# --------------------------------- CR ---------------------------------
CR={}
for c in comms:
    dc=0
    for node in c:
        dc=G.degree(node)+dc
    for node in c:
        CR[node]=dc
print('--')
order_CR = sorted(CR.items(), key=lambda x: x[1], reverse=True)  # list类型
order_CR = dict(order_CR)
# --------------------------------- 划分等级 ---------------------------------
rank =1
temp=0
order_LC=rank_dict(order_LC,rank,temp)
order_GR=rank_dict(order_GR,rank,temp)
order_CR=rank_dict(order_CR,rank,temp)
# --------------------------------- CIFR ---------------------------------
CIFR={}
for node in G.nodes:
    CIFR[node]=1/(order_CR[node]+order_GR[node]+order_LC[node])
order_CIFR = sorted(CIFR.items(), key=lambda x: x[1], reverse=True)  # list类型
order_CIFR=dict(order_CIFR)

seed_nodes = list(order_CIFR.keys())
# print(seed_nodes)
