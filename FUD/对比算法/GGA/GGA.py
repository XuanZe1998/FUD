def degreeDiscountIC(G, k, p=.01):
    ''' Finds initial set of nodes to propagate in Independent Cascade model (without priority queue)
    Input: G -- networkx graph object
    k -- number of nodes needed
    p -- propagation probability
    Output:
    S -- chosen k nodes
    Note: the routine runs twice slower than using PQ. Implemented to verify results
    '''
    d = dict()
    dd = dict()  # degree discount
    t = dict()  # number of selected neighbors
    S = []  # selected set of nodes
    for u in G:
        d[u] = sum([G[u][v]['weight'] for v in G[u]])  # each edge adds degree 1
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd[u] = d[u]
        t[u] = 0
    for i in range(k):
        # dd saves tuples, max function of a tuple compares the first value in the  tuple, if it the same then compare the second,
        # we want to compare only the second, so x being a tuple with x[1] we select the second value of the tuple
        u, ddv = max(dd.items(), key=lambda x: x[1])
        #        u, ddv = max(dd.items(), key=lambda (k,v): v)
        dd.pop(u)
        S.append(u)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight']  # increase number of selected neighbors
                dd[v] = d[v] - 2 * t[v] - (d[v] - t[v]) * t[v] * p
    return S


'''
if __name__ == '__main__':
    import time
    import networkx as nx
    start = time.time()
    from YHSF import ICModel
    S=[]
    address = 'E:/新建文件夹/DPSO_demo-master/twitter.txt'
    def read_raw_data(raw_file_dir):
        g = nx.MultiDiGraph()
        for line in open(raw_file_dir):
            str_list = line.split()
            n1 = int(str_list[0])
            n2 = int(str_list[1])
            weight = float(1)
            # try:
            #     weight = float(str_list[2])
            # except:
            #     weight = float(1)
            g.add_weighted_edges_from([(n1, n2, weight)])  # G.add_edges_from([(n1, n2)])
            # g.add_edges_from([(n1, n2)])
            # g.add_edges_from([(n1, n2, {'weight': weight})])
            # g.add_edges_from([(n1, n2, {'weight': weight, 'timestamp': timestamp})])
        return g
    def read_raw_data1(raw_file_dir):
        g = nx.MultiDiGraph()
        for line in open(raw_file_dir):
            str_list = line.split()
            n1 = int(str_list[0])
            n2 = int(str_list[1])
            weight = float(1)
            # try:
            #     weight = float(str_list[2])
            # except:
            #     weight = float(1)
            # g.add_weighted_edges_from([(n1, n2, weight)])  # G.add_edges_from([(n1, n2)])
            g.add_edges_from([(n1, n2)])
            # g.add_edges_from([(n1, n2, {'weight': weight})])
            # g.add_edges_from([(n1, n2, {'weight': weight, 'timestamp': timestamp})])
        return g
    def multidir2simpledir(multidir_graph):
        # 输出所有有向边，包括权重
        # print("-" * 10)  # print(list(G.edges(data=True)))
        # for e in multidir_graph.edges.data('weight'):
        #     print(e)
        print("raw:", multidir_graph.number_of_nodes(), multidir_graph.number_of_edges(),
              nx.number_of_selfloops(multidir_graph))
        c = Counter(multidir_graph.edges())
        simpledir_graph = nx.DiGraph()
        for n1, n2, w in multidir_graph.edges.data('weight'):
            # avoid repeating edges and self-loops
            if not simpledir_graph.has_edge(n1, n2) and n1 != n2:
                simpledir_graph.add_edge(n1, n2, weight=c[n1, n2])
            if n1 == n2:  # 没有loop的节点属性为None，有loop为loop个数
                if not simpledir_graph.has_node(n1):  # 新节点要先添加
                    simpledir_graph.add_node(n1, loops=c[n1, n2])
                else:  # 已有的节点，直接设置属性
                    simpledir_graph.nodes[n1]["loops"] = c[n1, n2]  # 报错原因是n1节点尚未添加到simpledir_graph
        print("processed:", simpledir_graph.number_of_nodes(), simpledir_graph.number_of_edges(),
              nx.number_of_selfloops(simpledir_graph))
        return simpledir_graph
    # 根据有向单边图的节点loop数以及边频数，重新计算边影响力
    def edgeimpact(simpledir_graph):
        graph = nx.DiGraph()
        N1insum = dict(simpledir_graph.in_degree(weight='weight'))
        for v, u, w in simpledir_graph.edges.data('weight'):
            impactv2u = float(w) / (N1insum[u] + 0)  # simpledir_graph.nodes[u]["loops"]
            graph.add_edge(v, u, weight=impactv2u)
        flag = os.path.exists(address)
        if not flag: file = open(address, 'a')
        # print("^" * 10)  # 输出归一化边权重
        for e in graph.edges.data('weight'):
            if not flag: s = str(e[0]) + " " + str(e[1]) + " " + str(e[2]) + '\n'
            if not flag: file.write(s)
            # print(e)
        if not flag: file.close()
        print("normalized:", graph.number_of_nodes(), graph.number_of_edges(), nx.number_of_selfloops(graph))
        # print(graph.get_edge_data(6,1),graph.get_edge_data(1,2))#graph.edges[1,2]['weight'],
        # print(graph.degree[1],graph.degree)
        # print(graph[4])#,graph.neighbors(4)
        return graph
    simpledir_graph = multidir2simpledir(read_raw_data(address))
    # 归一化边权重
    graph = edgeimpact(simpledir_graph)
    print(time.time() - start)
    #for k in range(5, 55, 5):
    print(S)
    Q=[]
    W=[]
    for k in range(5, 55, 5):
        start_time = time.time()
        S = degreeDiscountIC(graph, k, p=.01)
        #print('A：%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
        activenodes = LTModel.simulate(graph, S, 0.25)
        Q.append(activenodes)
        end_time = time.time()
        runningtime1 = end_time - start_time
        W.append(runningtime1)
        print(Q)
        print("总时间：", W)
'''