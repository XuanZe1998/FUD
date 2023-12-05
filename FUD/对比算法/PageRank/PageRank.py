import networkx as nx
import 数据集引用



G=数据集引用.G
G=nx.karate_club_graph()
pg_value = nx.pagerank(G)
dc_value=nx.degree_centrality(G)
pg_seed_nodes=list(pg_value.keys())
dc_seed_nodes=list(dc_value.keys())




