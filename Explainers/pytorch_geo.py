from torch_geometric.explain import Explainer, GNNExplainer, Explanation, PGExplainer
import graphviz
import torch

def ptg_gnnexplainer(model, data):

        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',  # Model returns log probabilities.
            ),
            threshold_config=None
        )

        # Generate explanation for the node at index `10`:
        explanation = explainer(data.x, data.edge_index, index=1)
        print(explanation)
        print(explanation.edge_mask)
        print(explanation.node_mask)
        explanation.visualize_graph(backend="graphviz")

        return explanation

def get_edges_and_nodes(exp):
    edge_mask_bool = torch.where(exp.edge_mask > 0.5, 1, 0)
    edge_mask_bool = edge_mask_bool.bool()
    edges_source = torch.masked_select(exp.edge_index[0], edge_mask_bool)
    edges_target = torch.masked_select(exp.edge_index[1], edge_mask_bool)
    #print(cora_gnnx_0)
    #print(cora_gnnx_0.shape)
    edges_subset = torch.stack((edges_source, edges_target))
    print(edges_subset)
    print(edges_subset.shape)

    # nodes
    node_subset = []
    for i in edges_subset:
        for j in i:
            node_subset.append(j.item())
    print(node_subset)
    node_subset = list(dict.fromkeys(node_subset))
    #print(node_tensor)
    #print(node_tensor.shape)

    return edges_subset, node_subset

