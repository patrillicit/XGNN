from Explainers.GNNExplainer import TargetedGNNExplainer, GNNExplainer
from Explainers.PGMExplainer import Node_Explainer
from Explainers.PGExplainer import PGExplainer
import torch, numpy as np, os

from torch_geometric.utils import k_hop_subgraph
import random

def gpu_to_cpu(data, device):
    data.x = torch.FloatTensor(data.x.cpu().numpy().copy()).to(device)
    data.edge_index = torch.LongTensor(data.edge_index.cpu().numpy().copy()).to(device)
    if hasattr(data, 'edge_attr'):
        data.edge_attr = torch.FloatTensor(data.edge_attr.cpu().numpy().copy()).to(device)
    return data

def node_attr_to_edge(edge_index, node_mask):
    edge_mask = np.zeros(edge_index.shape[1])
    edge_mask += node_mask[edge_index[0].cpu().numpy()]
    edge_mask += node_mask[edge_index[1].cpu().numpy()]
    return edge_mask

def sample_large_graph(data):
    if len(data.edge_index[1]) > 50000:
        print("Too many edges, sampling large graph...")
        node_idx = random.randint(0, data.num_nodes - 1)
        x, edge_index, mapping, edge_mask, subset, kwargs = get_subgraph(
            node_idx, data.x, data.edge_index, num_hops=3
        )
        data = data.subgraph(subset)
        print(f"Sample size: {data.num_nodes} nodes and {data.num_edges} edges")
    return data

def get_subgraph(node_idx, x, edge_index, num_hops, **kwargs):
    num_nodes, num_edges = x.size(0), edge_index.size(1)

    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx, num_hops, edge_index, relabel_nodes=True, num_nodes=num_nodes
    )

    x = x[subset]
    for key, item in kwargs.items():
        if torch.is_tensor(item) and item.size(0) == num_nodes:
            item = item[subset]
        elif torch.is_tensor(item) and item.size(0) == num_edges:
            item = item[edge_mask]
        kwargs[key] = item

    return x, edge_index, mapping, edge_mask, subset,

def explain_gnnexplainer_node(model, data, node_idx, device, **kwargs):
    data = gpu_to_cpu(data, device)
    explainer = GNNExplainer(
        model,
        num_hops=kwargs["num_layers"],
        epochs=1000,
        #edge_ent=kwargs["edge_ent"],
        #edge_size=kwargs["edge_size"],
        allow_edge_mask=True,
        allow_node_mask=True,
        device=device,
    )
    node_feat_mask, edge_mask = explainer.explain_node(
        node_idx,
        x=data.x,
        edge_index=data.edge_index,
        #edge_attr=data.edge_attr,
    )
    edge_mask = edge_mask.cpu().detach().numpy()
    # 1 node feature mask for all the nodes.
    node_feat_mask = node_feat_mask.cpu().detach().numpy()
    return edge_mask.astype("float"), node_feat_mask.astype("float")

def explain_pgmexplainer_node(model, data, node_idx, target, device, **kwargs):
    explainer = Node_Explainer(
        model,
        data.edge_index,
        #data.edge_attr,
        X=data.x,
        num_layers = kwargs["num_layers"],
        device=device,
        print_result=0,
    )
    explanation = explainer.explain(
        node_idx,
        target,
        num_samples=100,
        top_node=None,
        p_threshold=0.05,
        pred_threshold=0.1,
    )
    node_attr = np.zeros(data.x.shape[0])
    for node, p_value in explanation.items():
        node_attr[node] = 1 - p_value
    edge_mask = node_attr_to_edge(data.edge_index, node_attr)
    return edge_mask.astype("float"), None


def explain_pgexplainer_node(model, data, node_idx, target, device, **kwargs):
    pgexplainer = PGExplainer(
        model,
        in_channels=kwargs["hidden_dim"] * 3,
        device=device,
        num_hops=kwargs["num_layers"],
        explain_graph=False,
    )
    dataset_name = kwargs["dataset_name"]
    subdir = os.path.join(kwargs["model_save_dir"], "pgexplainer")
    os.makedirs(subdir, exist_ok=True)
    pgexplainer_saving_path = os.path.join(subdir, f"pgexplainer_{dataset_name}.pth")
    if os.path.isfile(pgexplainer_saving_path):
        print("Load saved PGExplainer model...")
        state_dict = torch.load(pgexplainer_saving_path)
        pgexplainer.load_state_dict(state_dict)
    else:
        data = sample_large_graph(data)
        pgexplainer.train_explanation_network(data)
        print("Save PGExplainer model...")
        torch.save(pgexplainer.state_dict(), pgexplainer_saving_path)
        state_dict = torch.load(pgexplainer_saving_path)
        pgexplainer.load_state_dict(state_dict)
    edge_mask = pgexplainer.explain_node(
        node_idx, data.x, data.edge_index, data.edge_attr
    )
    edge_mask = edge_mask.cpu().detach().numpy()
    return edge_mask.astype("float"), None
