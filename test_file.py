import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()


G.add_nodes_from([1, 2, 3,4,5,6,7,8,9,10])
G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10)])
nx.draw(G, with_labels=True) 

plt.show()
