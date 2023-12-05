import copy
import networkx as nx
import numpy as np
import 数据集引用
import time



G= 数据集引用.ECRM_GG
ECRM_G= 数据集引用.ECRM_G

min_degree = 1
IT = 1
shell = 1
while G:
    degree = dict(G.degree).values()
    nodes = dict(G.degree).keys()
    remove_nodes = []
    if min(degree) <= min_degree:
        for index, d in enumerate(degree):
            if d <= min_degree:
                node = list(nodes)[index]
                ECRM_G.nodes[node]['IT'] = IT
                ECRM_G.nodes[node]['shell'] = shell
                remove_nodes.append(node)
        G.remove_nodes_from(remove_nodes)
        IT = IT + 1
    else:
        min_degree = min(degree)
        shell = shell + 1
        for index, d in enumerate(degree):
            if d == min_degree:
                node = list(nodes)[index]
                ECRM_G.nodes[node]['IT'] = IT
                ECRM_G.nodes[node]['shell'] = shell
                remove_nodes.append(node)
        G.remove_nodes_from(remove_nodes)
        IT = IT + 1

print('shell层数：',shell)
nbr_dict = {}
ECRM_G_nodes = list(ECRM_G.nodes)
for n in ECRM_G_nodes:
    vector = [0] * IT
    for nbr in ECRM_G.neighbors(n):
        vector[ECRM_G.nodes[nbr]['IT'] - 1] = vector[ECRM_G.nodes[nbr]['IT'] - 1] + 1
    nbr_dict[n] = vector
# print(f'节点壳层向量表示为\n{nbr_dict}')

vects = np.array(list(nbr_dict.values()))
vects_corr = np.corrcoef(vects)
# 节点和邻居之间，相关系数越低越好
# print('节点壳层向量皮尔逊相关系数\n', vects_corr)
print('维度：',vects_corr.shape)

degrees = dict(ECRM_G.degree)
max_degree = degrees[max(degrees,key=degrees.get)]
# 计算SCC
for scnode in ECRM_G_nodes:
    scnode_nbr = list(ECRM_G.neighbors(scnode))
    SCC=0
    for sc in scnode_nbr:
        SCC=SCC+((2-vects_corr[ECRM_G_nodes.index(scnode)][ECRM_G_nodes.index(sc)])+(2*ECRM_G.degree[sc]/max_degree+1))
    ECRM_G.nodes[scnode]['SCC'] =SCC

# 计算CRM
for crnode in ECRM_G.nodes:
    crnode_nbr=list(ECRM_G.neighbors(crnode))
    CRM=0
    for cr in crnode_nbr:
        CRM = CRM +ECRM_G.nodes[cr]['SCC']
    ECRM_G.nodes[crnode]['CRM']=CRM

# 计算ECRM
ECRM_dict={}
for ecnode in ECRM_G.nodes:
    ecnode_nbr=list(ECRM_G.neighbors(ecnode))
    ECRM=0
    for ec in ecnode_nbr:
        ECRM = ECRM +ECRM_G.nodes[cr]['SCC']
    ECRM_G.nodes[ecnode]['CRM']=ECRM
    ECRM_dict[ecnode]=ECRM

sorted_ECRM = sorted(ECRM_dict.items(), key=lambda x: x[1],reverse=True)
# print(ECRM_dict)
# print(sorted_ECRM)

# 取前5%
sorted_ECRM_keys=dict(sorted_ECRM).keys()
sorted_ECRM_values=dict(sorted_ECRM).values()

keys_list_new = list(map(int, sorted_ECRM_keys))
values_list_new = list(map(int, sorted_ECRM_values))







