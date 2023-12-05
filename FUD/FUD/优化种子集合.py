import networkx as nx
import 数据集引用
import VIKOR
import math

seed_nodes = VIKOR.seed_nodes[:math.ceil(len(VIKOR.seed_nodes) * 0.2)]
final_nodes = []
G1 = nx.read_edgelist(数据集引用.file,nodetype=int)
G2 = nx.read_edgelist(数据集引用.file,nodetype=int)
ks = {}


# ------------------------k-shell------------------------
def kshell(G1, G2):
    min_degree = 1
    IT = 1
    shell = 1
    while G1:
        degree = dict(G1.degree).values()
        nodes = dict(G1.degree).keys()
        remove_nodes = []
        if min(degree) <= min_degree:
            for index, d in enumerate(degree):
                if d <= min_degree:
                    node = list(nodes)[index]
                    G2.nodes[node]['IT'] = IT
                    G2.nodes[node]['shell'] = shell
                    remove_nodes.append(node)
            G1.remove_nodes_from(remove_nodes)
            IT = IT + 1
        else:
            min_degree = min(degree)
            shell = shell + 1
            for index, d in enumerate(degree):
                if d == min_degree:
                    node = list(nodes)[index]
                    G2.nodes[node]['IT'] = IT
                    G2.nodes[node]['shell'] = shell
                    remove_nodes.append(node)
            G1.remove_nodes_from(remove_nodes)
            IT = IT + 1
    return G2


G = kshell(G1, G2)
for i in seed_nodes:
    ks[i] = [G.nodes[i]['shell'],G.nodes[i]['IT']]

ks = dict(sorted(ks.items(), key=lambda x: x[1], reverse=True))
seed_nodes=list(ks.keys())
for index,node in enumerate(seed_nodes):
    temp=node
    for nbr in list(G.neighbors(node)):
        if G.degree(node)<G.degree(nbr):
            temp=nbr
    final_nodes.append(temp)







