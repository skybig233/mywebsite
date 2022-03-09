# -*- coding: utf-8 -*-
# @Time    : 2022/3/9 10:10
# @Author  : Jiangzhesheng
# @File    : trees.py
# @Software: PyCharm
# @Description:

from typing import List
import graphviz

class Edge:
    def __init__(self, in_id: int, out_id: int, label: str):
        self.in_node = in_id
        self.out_node = out_id
        self.label = label

    def __str__(self) -> str:
        return self.label

    def __repr__(self):
        return self.label

class TreeNode:
    _id_value = []

    def __init__(self, label=None, child=None) -> None:
        """
        特殊的树，节点可能重名所以要分配id，到子树的边有标签
        :type label:str
        :type child:Dict[str,TreeNode]
        """
        self.label = label
        self.child = child

        self.id = len(TreeNode._id_value)
        TreeNode._id_value.append(self)

    def __str__(self) -> str:
        return self.label

    def __repr__(self):
        return self.label

    @classmethod
    def id2node(cls, id: int):
        """
        :param id:
        :return:TreeNode
        """
        assert isinstance(cls._id_value[id], cls)
        return cls._id_value[id]

    @classmethod
    def init_from_dict(cls, d):
        if isinstance(d, str):
            return TreeNode(label=d, child=None)
        ans = cls()
        for node in d:
            ans.label = node
            ans.child = {}
            for edge in d[node]:
                ans.child[edge] = cls.init_from_dict(d[node][edge])
        return ans

    @property
    def treeNodes(self):
        """
        :return:List[TreeNode]
        """
        return self.visit_root_first()[0]

    @property
    def treeEdges(self):
        """
        :return:List[Edge]
        """
        return self.visit_root_first()[1]

    def visit_root_first(self):
        """
        先根遍历
        :return:节点,边
        """
        nodes = []
        edges = []

        nodes += [self]

        if self.child is None:
            return nodes, edges
        for edge in self.child:
            edge_unit = Edge(label=edge, in_id=self.id, out_id=self.child[edge].id)
            edges += [edge_unit]
            tmp_nodes, tmp_edges = self.child[edge].visit_root_first()
            nodes += tmp_nodes
            edges += tmp_edges
        return nodes, edges

    def show_tree_graphviz(self)->graphviz.Digraph:
        graph = graphviz.Digraph()
        for treenode in self.treeNodes:
            shape='oval' if treenode.child is None else 'box'
            graph.node(name='node%d' % treenode.id, label=treenode.label, fontname="Microsoft YaHei",shape=shape)
        for edge in self.treeEdges:
            graph.edge('node%d' % edge.in_node, 'node%d' % edge.out_node, label=edge.label, fontname="Microsoft YaHei")
        return graph

    def show_tree_gradual_graphviz(self)->List[graphviz.Digraph]:
        graphs=[]
        n=len(self.treeNodes)
        for i in range(n):
            graph = graphviz.Digraph()
            for treenode in self.treeNodes[:i+1]:
                shape = 'oval' if treenode.child is None else 'box'
                graph.node(name='node%d' % treenode.id, label=treenode.label, fontname="Microsoft YaHei",shape=shape)
            for edge in self.treeEdges[:i]:
                graph.edge('node%d' % edge.in_node, 'node%d' % edge.out_node, label=edge.label,
                           fontname="Microsoft YaHei")
            graphs.append(graph)
        return graphs

    # def show_tree_gradual_graphviz(self):
    #     """
    #     展示树的生成过程
    #     :return:
    #     """
    #     graphs=[]
    #     n=len(self.treeNodes)
    #     for i in range(n):
    #         #每轮展示一个树节点
    #         graph = graphviz.Digraph()
    #         show_nodes=self.treeNodes[:i+1]
    #         #出入节点在当前展示节点的边才能被展示
    #         show_nodes_id=[node.id for node in show_nodes]
    #         show_edges=[edge for edge in self.treeEdges if edge.in_node in show_nodes_id and edge.out_node in show_nodes_id]
    #
    #         for treenode in show_nodes:
    #             graph.node(name='node%d' % treenode.id, label=treenode.label, fontname="Microsoft YaHei")
    #         for edge in show_edges:
    #             graph.edge('node%d' % edge.in_node, 'node%d' % edge.out_node, label=edge.label,
    #                        fontname="Microsoft YaHei")
    #         graphs.append(graph)
    #     return graphs

if __name__ == '__main__':
    d = {'Outlook': {'overcast': {'Humidity': {'high': 'no',
                                               'normal': 'yes'}},
                     'sunny': 'yes',
                     'rainy': {'Temperature': {'cool': 'no',
                                               'hot': {'Windy': {'very': 'no', 'not': 'yes'}},
                                               'mild': 'no'}}}}
    # d={'Humidity': {'high': 'no','normal': 'yes'}}
    tmp=TreeNode.init_from_dict(d)
    graphs=tmp.show_tree_gradual_graphviz()
    # n=3
    # graphs[n].view()
    # tmp.show_tree_graphviz()
    #
    # show=[]
    # n=len(tmp.treeNodes)
    # for i in range(n-1,-1,-1):
    #     tmp.treeNodes[i].delete()
    #     show.append(tmp.treeNodes[0])
    # print(show)