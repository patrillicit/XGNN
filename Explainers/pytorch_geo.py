from torch_geometric.explain import Explainer, GNNExplainer, Explanation, PGExplainer, ThresholdConfig
import graphviz
import torch
import pandas as pd
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data


def ptg_gnnexplainer(model, data, node, thresholding, k, path):
        if thresholding is "topk":
            thresholding = ThresholdConfig("topk", k)
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='object',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',  # Model returns log probabilities.
            ),
            threshold_config= thresholding
        )

        # Generate explanation for the node at index `10`:
        explanation = explainer(data.x, data.edge_index, index=node)
        print(explanation)
        #print(explanation.edge_mask)
        #print(explanation.node_mask)
        explanation.visualize_graph(path=path + "pty.gv.pdf", backend="graphviz")

        return explainer, explanation

def get_edges_and_nodes(exp, thres):
    if thres is None:
        thres = 0.001
    edge_mask_bool = torch.where(exp.edge_mask > thres, 1, 0)
    edge_mask_bool = edge_mask_bool.bool()
    edges_source = torch.masked_select(exp.edge_index[0], edge_mask_bool)
    edges_target = torch.masked_select(exp.edge_index[1], edge_mask_bool)
    edge_score = torch.masked_select(exp.edge_mask, edge_mask_bool)
    #print(cora_gnnx_0)
    #print(cora_gnnx_0.shape)
    edges_subset = [edges_source.tolist(), edges_target.tolist()]
    print(edges_subset)

    # nodes
    node_subset = []
    for i in edges_subset:
        for j in i:
            node_subset.append(j)
    print(node_subset)
    node_subset = list(dict.fromkeys(node_subset))

    node_score = []
    for node in node_subset:
        node_score.append(exp.node_mask[node][0])
    #print(node_tensor)
    #print(node_tensor.shape)
    """
    d = [node_subset, edges_source, edges_target]
    from itertools import zip_longest
    df = pd.DataFrame(zip_longest(*d, fillvalue=''))
    print(df.head(15))
    df.to_csv(path + ".csv")
    """
    return Data(nodes=node_subset, edges=edges_subset, node_score=node_score, edge_score=edge_score)

def ptg_pgexplainer(model, data, node, thresholding, path):

        explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=30),
            explanation_type='phenomenon',
            #node_mask_type='object',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',  # Model returns log probabilities.
            ),
            threshold_config=thresholding
        )

        for epoch in range(30):
            for index in range(0,500):  # Indices to train against.
                loss = explainer.algorithm.train(epoch, model, data.x, data.edge_index,
                                                 target=data.y, index=index)

        # Generate explanation for the node at index `10`:
        explanation = explainer(data.x, data.edge_index, index=node, target=data.y)
        print(explanation)
        # print(explanation.edge_mask)
        # print(explanation.node_mask)
        explanation.visualize_graph(path=path + "pty.gv.pdf", backend="graphviz")

        return explainer, explanation
