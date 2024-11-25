import networkx as nx
import matplotlib.pyplot as plt

nodes1 = [1,2,3,4]
edges1 = [(1,2), (2,3), (3,4)]
G1 = nx.DiGraph()
G1.add_nodes_from(nodes1)
G1.add_edges_from(edges1)
G1.nodes[1]['test'] = 0
G1.nodes[2]['test'] = 1
G1.nodes[3]['test'] = 2
G1.nodes[4]['test'] = 3
nodes2 = [5,6,7,8]
edges2 = [(5,6), (6,7), (7,8)]
G2 = nx.DiGraph()
G2.add_nodes_from(nodes2)
G2.add_edges_from(edges2)
G2.nodes[5]['test'] = 0
G2.nodes[6]['test'] = 1
G2.nodes[7]['test'] = 2
G2.nodes[8]['test'] = 4
node_labels = dict()
node_labels[1] = 0
node_labels[2] = 1
node_labels[3] = 2
node_labels[4] = 3
node_labels1 = node_labels.copy()
node_labels[5] = 0
node_labels[6] = 1
node_labels[7] = 2
node_labels[8] = 3
nx.draw(G1)
plt.show()
nx.draw(G2)
plt.show()
print(nx.vf2pp_is_isomorphic(G1, G2))
print(nx.vf2pp_isomorphism(G1, G2, node_label='test'))