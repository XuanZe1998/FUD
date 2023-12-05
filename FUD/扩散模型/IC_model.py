import math
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np


# 无向图G,种子集合seed,IC模拟次数n
def IC_model(G, seed, n):
    sum_ndoes = len(G.nodes)    # 总节点数目
    diffusion = []  # 传播的次数
    active_nodes = []  # 激活节点数
    diffusion_and_actnodes={}

    for i in range(n):
        sum_diffusion = 0
        current_node_set = seed
        sum_active_ndoes = len(seed)
        act_nodes = [seed]
        # 为每条边赋传播概率值
        for edge in G.edges:
            G.edges[edge[0], edge[1]]['p'] = random.uniform(0, 1)
        # 用state标识状态 state=0 未激活，state=1 激活
        for node in G:
            if node in seed:
                G.nodes[node]['state'] = 1
            else:
                G.nodes[node]['state'] = 0
        for nn in range(50):
            temp = []
            for node in current_node_set:
                for nbr in G.neighbors(node):
                    # 这个值可以随便设置，p=任何
                    if G.edges[node, nbr]['p'] > random.uniform(0, 1) and G.nodes[nbr]['state'] == 0:
                        G.nodes[nbr]['state'] = 1
                        temp.append(nbr)
            if len(temp) != 0:
                current_node_set = temp
                act_nodes.append(temp)
            # 总传播次数，总激活节点数目
            sum_diffusion = len(act_nodes)
            sum_active_ndoes = sum_active_ndoes + len(temp)
        diffusion.append(sum_diffusion)
        active_nodes.append(sum_active_ndoes)
        diffusion_and_actnodes[sum_diffusion]=sum_active_ndoes

    diffusion_and_actnodes=dict(sorted(diffusion_and_actnodes.items(),key = lambda x:x[1],reverse = True))
    # 排序过后
    diffusion_copy=list(diffusion_and_actnodes.keys())
    active_nodes_copy=list(diffusion_and_actnodes.values())

    avg_diffusion = math.ceil(np.mean(diffusion))
    avg_active_node = math.ceil(np.mean(active_nodes))
    print(f"一共模拟{n}次，平均扩散{avg_diffusion}次，平均每次激活{avg_active_node}个节点")

    # 返回平均扩散次数和平均激活几点数
    return avg_diffusion, avg_active_node,diffusion_copy,active_nodes_copy

