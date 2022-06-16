from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_scipy_sparse_matrix
def read_combined(dataset, remove_small = None, read_edge_labels = False):
    graphs = list()
    for data in dataset:
        adj = to_scipy_sparse_matrix(data.edge_index).tocsr()
        print(adj)
        break


    graphs.append(graph)

    return graphs
if __name__=='__main__':
    dataset = TUDataset(root='model/data/TUDataset',name='MUTAG')
    print(dataset[0].node_label)
    read_combined(dataset)
