from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import numpy as np
import pandas as pd
import csv

"""
Data Import
"""
def get_cora():
    cora = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    return cora

def get_pubmed():
    pubmed = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    return pubmed

def get_citeseer():
    citeseer = Planetoid(root='data/Planetoid', name='CiteSeer', transform=NormalizeFeatures())
    return citeseer


"""
This Method creates a csv-file with all edges, composed of source and target node
"""
def create_csv2(data):
    edge_list = [data.edge_index[0].tolist(),data.edge_index[1].tolist()]
    #print(edge_list)
    edge_list_trans = tuple(zip(*edge_list))
    #print(edge_list_trans)
    with open('edges.csv', 'w') as f:
        write = csv.writer(f)
        write.writerows(edge_list_trans)
