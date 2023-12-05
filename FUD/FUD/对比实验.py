import math
import time
import numpy as np
from 扩散模型 import IC_model
# import VIKOR
import matplotlib.pyplot as plt
from 论文综合代码.对比算法.ECRM import ECRM
from 论文综合代码.对比算法.CIFR import CIFR
from 论文综合代码.对比算法.CELF import celf
from 论文综合代码.对比算法.PageRank import PageRank
import 数据集引用
import 优化种子集合

'''
目的：
    同一个数据集上，不同算法
    x轴为种子集
    y轴为激活种子的数量
'''
start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
print('对比实验开始时间:', start_time)

experiments_G = 数据集引用.G
x_axis_data, y_axis_data, ECRM_x_data, ECRM_y_data, CIFR_x_data, CIFR_y_data, CELF_x_data, CELF_y_data, \
    PG_x_data, PG_y_data, DC_x_data, DC_y_data = [[] for x in range(12)]
self_f = self_a = ecrm_f = ecrm_a = cifr_f = cifr_a = celf_f = celf_a = pagerank_f = pagerank_a = dc_f = dc_a = 0
# self_seed_set = VIKOR.seed_nodes
self_seed_set = 优化种子集合.final_nodes
self_end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
print('本文算法结束时间', self_end_time)
ECRM_seed_set = ECRM.keys_list_new
ECRM_end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
print('ECRM算法结束时间', ECRM_end_time)
CIFR_seed_set = CIFR.seed_nodes
CIFR_end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
print('CIFR算法结束时间', ECRM_end_time)
CELF_seed_set = celf.CELF_seeds
CELF_end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
print('CELF算法结束时间', ECRM_end_time)
PG_seed_set = PageRank.pg_seed_nodes
PG_end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
print('PageRank算法结束时间', PG_end_time)
DC_seed_set = PageRank.dc_seed_nodes
DC_end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
print('PageRank算法结束时间', DC_end_time)

# 模型重复传播次数
# for i in np.arange(0.02, 0.055, 0.005):
#     print(f'种子节点数为前{i}%')
#     self_f, self_a, self_d, self_active_nodes = IC_model.IC_model(experiments_G,
#                                                                   self_seed_set[:math.ceil(len(self_seed_set) * i)],
#                                                                   n=200)
#     ecrm_f, ecrm_a, ecrm_d, ecrm_active_nodes = IC_model.IC_model(experiments_G,
#                                                                   ECRM_seed_set[:math.ceil(len(ECRM_seed_set) * i)],
#                                                                   n=200)
#     cifr_f, cifr_a, cifr_d, cifr_active_nodes = IC_model.IC_model(experiments_G,
#                                                                   CIFR_seed_set[:math.ceil(len(CIFR_seed_set) * i)],
#                                                                   n=200)
#     celf_f, celf_a, celf_d, celf_active_nodes = IC_model.IC_model(experiments_G,
#                                                                   CELF_seed_set[:math.ceil(len(CELF_seed_set) * i)],
#                                                                   n=200)
#     pagerank_f, pagerank_a, pagerank_d, pagerank_active_nodes = IC_model.IC_model(experiments_G, PG_seed_set[:math.ceil(
#         len(PG_seed_set) * i)], n=200)
#     dc_f, dc_a, dc_d, dc_active_nodes = IC_model.IC_model(experiments_G, DC_seed_set[:math.ceil(len(DC_seed_set) * i)],
#                                                           n=200)
#     x_axis_data.append(i)
#     y_axis_data.append(self_a)
#     ECRM_x_data.append(i)
#     ECRM_y_data.append(ecrm_a)
#     CIFR_x_data.append(i)
#     CIFR_y_data.append(cifr_a)
#     CELF_x_data.append(i)
#     CELF_y_data.append(celf_a)
#     PG_x_data.append(i)
#     PG_y_data.append(pagerank_a)
#     DC_x_data.append(i)
#     DC_y_data.append(dc_a)
for i in np.arange(5, 55, 5):
    print(f'种子节点数为前{i}')
    self_f, self_a, self_d, self_active_nodes = IC_model.IC_model(experiments_G,
                                                                  self_seed_set[:i],
                                                                  n=200)
    ecrm_f, ecrm_a, ecrm_d, ecrm_active_nodes = IC_model.IC_model(experiments_G,
                                                                  ECRM_seed_set[:i],
                                                                  n=200)
    cifr_f, cifr_a, cifr_d, cifr_active_nodes = IC_model.IC_model(experiments_G,
                                                                  CIFR_seed_set[:i],
                                                                  n=200)
    celf_f, celf_a, celf_d, celf_active_nodes = IC_model.IC_model(experiments_G,
                                                                  CELF_seed_set[:i],
                                                                  n=200)
    pagerank_f, pagerank_a, pagerank_d, pagerank_active_nodes = IC_model.IC_model(experiments_G, PG_seed_set[:i], n=200)
    dc_f, dc_a, dc_d, dc_active_nodes = IC_model.IC_model(experiments_G, DC_seed_set[:i],
                                                          n=200)
    x_axis_data.append(i)
    y_axis_data.append(self_a)
    ECRM_x_data.append(i)
    ECRM_y_data.append(ecrm_a)
    CIFR_x_data.append(i)
    CIFR_y_data.append(cifr_a)
    CELF_x_data.append(i)
    CELF_y_data.append(celf_a)
    PG_x_data.append(i)
    PG_y_data.append(pagerank_a)
    DC_x_data.append(i)
    DC_y_data.append(dc_a)
# 'b' 蓝色，'m' 洋红色，'g' 绿色，'y' 黄色，'r' 红色，'k' 黑色，'w' 白色，'c' 青绿色
# '‐' 实线，'‐‐' 破折线，'‐.' 点划线，':' 虚线。
# '.' 点标记，',' 像素标记(极小点)，'o' 实心圈标记，'v' 倒三角标记，'^' 上三角标记，'>' 右三角标记，'<' 左三角标记，'s'正方形标记，''
# 第一张图
fig1=plt.figure(1)
ax1 = fig1.add_subplot(111)
ax1.plot(x_axis_data, y_axis_data, 'ro--', alpha=0.5, linewidth=1,
         label='This article algorithm')
ax1.plot(ECRM_x_data, ECRM_y_data, 'b^--', alpha=0.5, linewidth=1,
         label='ECRM')
ax1.plot(CIFR_x_data, CIFR_y_data, 'ys--', alpha=0.5, linewidth=1,
         label='CIFR')
ax1.plot(CELF_x_data, CELF_y_data, 'co--', alpha=0.5, linewidth=1,
         label='CELF')
ax1.plot(PG_x_data, PG_y_data, 'go--', alpha=0.5, linewidth=1,
         label='PageRank')
ax1.plot(DC_x_data, DC_y_data, 'mo--', alpha=0.5, linewidth=1,
         label='DC')
ax1.set_title(数据集引用.dataset_name + '  dataset')
ax1.legend()  # 显示上面的label
plt.xlabel('seed nodes')  # x_label
plt.ylabel('active nodes')  # y_label
ax11 = fig1.gca()  # gca:get current axis得到当前轴
# 设置图片的右边框和上边框为不显示
ax11.spines['right'].set_color('none')
ax11.spines['top'].set_color('none')
fig1.savefig('C:/Users/Tian/Desktop/' + 数据集引用.dataset_name+'1' + '.png')
fig1.show()

# 第二张图
fig2=plt.figure(2)
ax2 = fig2.add_subplot(111)
# '.' 点标记，',' 像素标记(极小点)，'o' 实心圈标记，'v' 倒三角标记，'^' 上三角标记，'>' 右三角标记，'<' 左三角标记，'*'星星标记，'+'加号标记，'d'细钻标记
ax2.scatter(self_d,self_active_nodes,color='r',marker='o',label='This article algorithm')
ax2.scatter(ecrm_d,ecrm_active_nodes,color='b',marker='^',label='ECRM')
ax2.scatter(cifr_d,cifr_active_nodes,color='y',marker='s',label='CIFR')
ax2.scatter(celf_d,celf_active_nodes,color='c',marker='*',label='CELF')
ax2.scatter(pagerank_d,pagerank_active_nodes,color='g',marker='+',label='PageRank')
ax2.scatter(dc_d,dc_active_nodes,color='m',marker='d',label='DC')
ax2.set_title(数据集引用.dataset_name+'  dataset')
ax2.legend()
plt.xlabel('diffusion numbers')
plt.ylabel('active nodes')
ax22 = fig2.gca()
ax22.spines['right'].set_color('none')
ax22.spines['top'].set_color('none')
fig2.savefig('C:/Users/Tian/Desktop/' + 数据集引用.dataset_name+'2' + '.png')
fig2.show()

for l in [x_axis_data, y_axis_data, ECRM_x_data, ECRM_y_data, CIFR_x_data, CIFR_y_data, CELF_x_data, CELF_y_data,
          PG_x_data, PG_y_data, DC_x_data, DC_y_data]:
    print(l)
for i in [self_d,self_active_nodes,ecrm_d,ecrm_active_nodes,cifr_d,cifr_active_nodes,celf_d,celf_active_nodes,pagerank_d,pagerank_active_nodes,dc_d,dc_active_nodes]:
    print(i)
end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
print('对比实验结束时间:', end_time)
