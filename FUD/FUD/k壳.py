import copy
import random
import math
import networkx as nx
import matplotlib.pyplot as plt


def k_shell(graph1, graph2, s):
    G = copy.copy(graph1)
    degrees = dict(G.degree())  # 获取图中各节点的度数信息
    # 初始化壳层数，迭代次数，当前图中节点的度
    shell = 1
    iterate = 1
    node_degree = 1

    while degrees:
        # 找到当前图中度数最小的节点
        # min_degrees = dict(G.degree())
        nodes_to_remove = []
        min_degree_key = min(dict(G.degree()), key=dict(G.degree()).get)  # 得到的是key值
        min_degree_value = dict(G.degree()).get(min_degree_key)

        for node, degree in degrees.items():
            if degree == min_degree_value:
                nodes_to_remove.append(node)
        # nodes_to_remove = [node for node, degree in degrees.items() if degree == min_degree_value]

        if min_degree_value == node_degree:
            if len(nodes_to_remove) != 0:
                for n in nodes_to_remove:
                    graph2.nodes[n]['IT'] = iterate
                    graph2.nodes[n]['shell'] = shell
                    G.remove_node(n)
                    degrees.pop(n)
                    # print(graph2.nodes[n]['IT'])
                iterate = iterate + 1
        elif min_degree_value > node_degree:
            node_degree = min_degree_value
            shell = shell + 1
            for i in nodes_to_remove:
                graph2.nodes[i]['IT'] = iterate
                graph2.nodes[i]['shell'] = shell
                G.remove_node(i)
                degrees.pop(i)
            iterate = iterate + 1

        else:
            shell = shell + 1
            for j in nodes_to_remove:
                graph2.nodes[j]['IT'] = iterate
                graph2.nodes[j]['shell'] = shell
                G.remove_node(j)
                degrees.pop(j)
            iterate = iterate + 1
        degrees = dict(G.degree())

    s=shell
    return graph2,s


def generateRandomPointFromAnnulus2(r1, r2):
    """
    在圆环内随机取点, r1<=r2
    :param r1: 内径
    :param r2: 外径
    :return:
    """
    assert r1 <= r2
    a = 1 / (r2 * r2 - r1 * r1)
    random_r = math.sqrt(random.uniform(0, 1) / a + r1 * r1)
    random_theta = random.uniform(0, 2 * math.pi)
    return random_r * math.cos(random_theta), random_r * math.sin(random_theta)


graph1 = nx.karate_club_graph()
graph2 = nx.karate_club_graph()
dict1 = {}
s=0
G1,s= k_shell(graph1, graph2, s)
# pos记录每个节点的位置信息
pos = {}
shell_node_num = {}
for s in range(s,0,-1):
    num =[]
    for n in G1.nodes:
        if G1.nodes[n]['shell'] == s:
            num.append(n)
            # pos[n] = generateRandomPointFromAnnulus2(2 * (s - 1), 2 * s - 1)
            pos[n] = generateRandomPointFromAnnulus2(2 * (5-s), 2 * (5-s) + 1)
    shell_node_num[s]=num
    circle = plt.Circle((0,0), 2*s-0.5, color='y', fill=False)
    plt.gcf().gca().add_artist(circle)
print(shell_node_num)
nx.draw_networkx(G1, pos, with_labels=True, width=0.1)
plt.axis("on")
plt.show()
# print('节点8的IT和shell分别是:', G1.nodes[8]['IT'], 'and', G1.nodes[8]['shell'])
