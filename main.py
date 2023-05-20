import torch
from Datasets.CitationDatasets import *
from Models.base import build_base
from Explainers.ExplanationManager import explain_gnnexplainer_node, explain_pgmexplainer_node, explain_pgexplainer_node
from Visualizations.Visualize import plot_expl_nc, custom_to_networkx
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from Explainers.pytorch_geo import *


def start():
    """
    Dataset Import

    """

    cora = get_cora()
    print(cora.data)

    create_csv2(cora)
    raw_cora_as_Data = Data(x=cora.x, edge_index=cora.edge_index)  # ,y=cora.y
    raw_cora_as_Graph = to_networkx(raw_cora_as_Data, to_undirected=True)

    #nx.draw(raw_cora_as_Graph)
    #plt.savefig("rawgraph.png")

    """
    Model Training
    
    """

    model1, num_layers = build_base(cora)
    print(num_layers)


    """
    XAI-Methods
    1) GNNExplainer from PytorchGeometric on node 1
    2) GNNExplainer from GraphFramEx on node 1
    3) PGExplainer from GraphFramEx on node 1
    """
    '1) GNNExplainer from PytorchGeometric on node 1'
    subg = ptg_gnnexplainer(model=model1, data=cora.data)
    get_edges_and_nodes(subg)

    xai_ptg_gnnx_subgraph = Data(x=subg.node_mask, edge_index=subg.edge_mask)
    print("PYTG:")
    print(xai_ptg_gnnx_subgraph)
    print(xai_ptg_gnnx_subgraph.x[xai_ptg_gnnx_subgraph.x > 0.01])
    print(xai_ptg_gnnx_subgraph.edge_index[xai_ptg_gnnx_subgraph.edge_index > 0.01])



    '2) GNNExplainer from GraphFramEx on node 1'

    expl1 = explain_gnnexplainer_node(model= model1, data= cora, node_idx= 1,
                                      device= torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                      num_layers= num_layers )
    print(expl1)
    #print(expl1[0][0:100])

    #create binary edge mask
    mask_tensor = torch.from_numpy(expl1[0])
    mask_tensor_bool = torch.where(mask_tensor>0.9, 1 ,0)
    mask_tensor_bool = mask_tensor_bool.bool()
    #print(mask_tensor_bool)
    #print(cora.edge_index.shape)
    #print(cora.edge_index[0])
    #print(cora.edge_index[0].shape)

    #a=torch.stack((mask_tensor_bool, mask_tensor_bool))
    #print(a.shape)

    #create reduced edge mask
    cora_gnnx_0 = torch.masked_select(cora.edge_index[0], mask_tensor_bool)
    cora_gnnx_1 = torch.masked_select(cora.edge_index[1], mask_tensor_bool)
    print(cora_gnnx_0)
    print(cora_gnnx_0.shape)
    exp1_subset = torch.stack((cora_gnnx_0,cora_gnnx_1))
    print(exp1_subset)
    print(exp1_subset.shape)

    #create subgraph
    node_list = []
    for i in exp1_subset:
        for j in i:
            node_list.append(j.item())
    print(node_list)
    node_tensor = cora.x[list(dict.fromkeys(node_list))]
    print(node_tensor)
    print(node_tensor.shape)

    #This is the subgraph with the nodes and edges contained in the explanation
    xai_subgraph = Data(x=node_tensor, edge_index=exp1_subset)
    print(xai_subgraph)

    #create subgraph(2)
    expl1_as_Data_Subgraph = raw_cora_as_Data.subgraph(exp1_subset)
    expl1_as_Data_Subgraph_with_Nodes = Data(x=expl1_as_Data_Subgraph.x, edge_index=expl1_as_Data_Subgraph.edge_index, num_nodes=7)

    print("Node 1 GNN-Subgraph:")
    print(expl1_as_Data_Subgraph)
    #a = to_networkx(expl1_as_Data_Subgraph_with_Nodes, to_undirected=True)
    #g = to_networkx(expl1_as_Data_Subgraph, to_undirected=True)
    a = to_networkx(xai_subgraph)
    print(a)
    print(type(a))

    nx.draw(a)
    plt.savefig("filename.png")

    #a = custom_to_networkx(expl1_as_Data_Subgraph)

    # b = plot_expl_nc(raw_cora_as_Graph, a, None, 1, None, None)


    '3) PGExplainer from GraphFramEx on node 1'


    expl1_pg = explain_pgexplainer_node(model=model1, data=cora, node_idx=1, target=None,
                                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                        hidden_dim=2, num_layers=2, dataset_name="cora",
                                        model_save_dir=r"C:\Users\Patrick\OneDrive - student.kit.edu\07 WS 22-23 BT\XGNN\Explainers")
    print("PG-Test:")
    print(expl1_pg)
    print(expl1_pg[0])
    print(expl1_pg[0][0:100])


if __name__ == '__main__':
    start()

