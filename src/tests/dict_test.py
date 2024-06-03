import matplotlib.pyplot as plt
import networkx as nx

test_dict = dict()
test_dict.setdefault(1, [])
test_dict.setdefault(2, [])

set1 = test_dict[1]
set1.append(1)
test_dict[1] = set1
print(test_dict)
set1 = test_dict[1]
set1.append(2)
test_dict[1] = set1
print(test_dict)
set1 = test_dict[2]
set1.append(2)
test_dict[2] = set1
print(test_dict)

G = nx.DiGraph(test_dict)
nx.draw(G)
plt.savefig('test')