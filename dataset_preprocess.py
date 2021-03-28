import os
import sys
import numpy as np
import pymetis
import metis
from data import load_data, preprocess_features, preprocess_adj
from collections import defaultdict
import networkx as nx
import pydot


def trans_dataset(dataset='cora', compressed=False, dict_of_list=False):
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)
    print('adj:', adj.shape)
    print('features:', features.shape)
    print('y:', y_train.shape, y_val.shape, y_test.shape)
    print('mask:', train_mask.shape, val_mask.shape, test_mask.shape)

    # 构建pymetis的输入数据
    edge_list = defaultdict(list)
    nnz = adj.nonzero()
    for i, j in zip(nnz[0], nnz[1]):
        edge_list[i].append(j)

    adj_list = []
    for i, edges in edge_list.items():
        adj_list.append(np.array(edges))

    # 如果后续能够确认训练收敛, 可以加上压缩的存储方式
    # 目前还没有存储为npz
    if compressed:
        pass
    else:
        pass

    if dict_of_list:
        return edge_list

    # 此处还应当返回 feature 等
    return adj_list


def shrink_point_metis(dataset='cora', n_part=2):
    """shrink_point_metis
        The input graph with N vertices contains N lines of data.
        The i-th line is a list of vertices which have an edge to the i-th vertex.
        Input: example.npz ['graph', 'feature']
    """
    # adj_matrix = np.load('processed/{}-processed.npz'.format(dataset))
    # print(adj_matrix)
    adjacency_list = trans_dataset(dataset)
    # n_cuts切割的边数, membership是一个数组, 长度为点总数, 数值为[0,n_part-1]
    n_cuts, membership = pymetis.part_graph(n_part, adjacency=adjacency_list)
    print(n_cuts)

    nodes_parts = []
    for i in range(n_part):
        nodes_parts.append(np.argwhere(np.array(membership) == i).ravel())
    print(nodes_parts, nodes_parts[0].shape, nodes_parts[1].shape)

    return node_parts


def visualized_metis(dataset='cora', n_part=2, ext='pdf'):
    """visualized_metis
        dataset: 数据集的处理暂时支持cora,pubmed和citeseer,reddit将会在四月版本支持
        n_part: 划分的块数
        ext: 输出的文件格式, 又pdf(default),svg,png
    """
    # 测试metis5.1 networkx 2.3
    # G = metis.example_networkx()
    # A =np.array([[0,0,1],[0,0,1],[1,1,0]])
    adjacency_list = trans_dataset(dataset, dict_of_list=True)

    G = nx.from_dict_of_lists(adjacency_list)
    (edge_cuts, parts) = metis.part_graph(G, n_part)
    print(edge_cuts)
    colors = ['red', 'blue', 'green', 'yellow']

    for i, p in enumerate(parts):
        G.node[i]['color'] = colors[p]
    nx.drawing.nx_pydot.write_dot(G, 'res/example.dot') # Requires pydot or pygraphviz

    # 可视化dot文件
    (graph,) = pydot.graph_from_dot_file('res/example.dot')
    if ext == 'png':
        graph.write_png('res/example.png')
    elif ext == 'pdf':
        graph.write_pdf('res/example.pdf')
    elif ext == 'svg':
        graph.write_svg('res/example.svg')
    else:
        print('[ERROR] ext option miss:', ext)

    print("Saved in res/example.{}".format(ext))


def partition_features():
    pass


if __name__ == "__main__":
    node_parts = shrink_point_metis(n_part=2)

# 早上过来试一试划分的算法
# 突然想到反向传播不会推, 去看看graphSAGE.
