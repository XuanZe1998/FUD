import networkx as nx
import igraph

# 读取facebook数据集的时候注意设置 nodetype = int
dataset_name = "power"  # name of dataset
path = "D:/Programing tools\Code/Pycharm code/First-paper/论文综合代码/datasets/" + dataset_name + ".txt"  # path to dataset
file ='D:/Programing tools/Code/Pycharm code/First-paper/论文综合代码/对比试验数据集/'+dataset_name+'.txt'
G = nx.read_edgelist(file, nodetype=int)
final_code_GRG = nx.read_edgelist(file, nodetype=int)
CIFR_GRG = nx.read_edgelist(file, nodetype=int)
ECRM_G = nx.read_edgelist(file, nodetype=int)
ECRM_GG = nx.read_edgelist(file, nodetype=int)
CELF_G =igraph.Graph.Read_Edgelist(file)