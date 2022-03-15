from Utils import Graph
import torch

graph_args = {'strategy': 'spatial'}
graph = Graph(**graph_args)

A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)


print(A.shape)