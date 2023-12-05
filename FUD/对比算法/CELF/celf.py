import igraph
import matplotlib.pyplot as plt
from random import uniform, seed
import numpy as np
import time
import networkx as nx
from igraph import *
import 数据集引用


def IC(g, S, p=0.5, mc=1000):
    spread = []
    for i in range(mc):

        # Simulate propagation process
        new_active, A = S[:], S[:]
        while new_active:

            # For each newly active node, find its neighbors that become activated
            new_ones = []
            for node in new_active:
                # Determine neighbors that become infected
                np.random.seed(i)
                success = np.random.uniform(0, 1, len(g.neighbors(node, mode="out"))) < p
                new_ones += list(np.extract(success, g.neighbors(node, mode="out")))

            new_active = list(set(new_ones) - set(A))

            # Add newly activated nodes to the set of activated nodes
            A += new_active

        spread.append(len(A))

    return (np.mean(spread))


def celf(g, k, p=0.1, mc=1000):
    start_time = time.time()
    marg_gain = [IC(g, [node], p, mc) for node in range(g.vcount())]

    # Create the sorted list of nodes and their marginal gain
    # vertices_list = [v.index for v in g.vs]
    # X = dict(zip(vertices_list,marg_gain))
    # a1 = sorted(X.items(), key=lambda x: x[1], reverse=True)
    # Q=list(a1)
    Q = sorted(list(zip(range(g.vcount()), marg_gain)), key=lambda x: x[1], reverse=True)


    # Select the first node and remove from candidate list
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS, timelapse = Q[1:], [g.vcount()], [time.time() - start_time]

    for _ in range(k - 1):

        check, node_lookup = False, 0

        if Q:
            while not check:
                # Count the number of times the spread is computed
                node_lookup += 1

                # Recalculate spread of top node
                current = Q[0][0]

                # Evaluate the spread function and store the marginal gain in the list
                Q[0] = (current, IC(g, S + [current], p, mc) - spread)

                # Re-sort the list
                Q = sorted(Q, key=lambda x: x[1], reverse=True)

                # Check if previous top node stayed on top after the sort
                check = (Q[0][0] == current)


        # Select the next node

            spread += Q[0][1]
            S.append(Q[0][0])
            SPREAD.append(spread)
            LOOKUPS.append(node_lookup)
            timelapse.append(time.time() - start_time)

        # Remove the selected node from the list
        Q = Q[1:]

    return (S, SPREAD, timelapse, LOOKUPS)


# source = []
# target = []
# network_file = 'D:\Programing tools\Code\Pycharm code\First-paper\论文综合代码\对比试验数据集\karate.txt'
# file = open(network_file, "r", encoding="utf-8")
# Vnumber, Enumber = file.readline().split()
# Enumber = int(Enumber)
# for i in range(0, Enumber):
#     start, end = file.readline().split()
#     source.append(int(start))
#     target.append(int(end))

# print('之前', source)
# print('之前', target)
# A = set(source + target)
# g =igraph.Graph.Read_Edgelist('D:\Programing tools\Code\Pycharm code\First-paper\论文综合代码\对比试验数据集\karate.txt')
g = 数据集引用.CELF_G
# g.add_edges(list(zip(source, target)))



# Run algorithms
celf_output = celf(g, 50, p=0.01, mc=1000)

# Print results
CELF_seeds=celf_output[0]
# print("celf output:   " + str(celf_output[0]))
