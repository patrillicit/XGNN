from datetime import datetime
import os
import torch
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
#import seaborn as sns
from torch_geometric.utils.convert import to_networkx
import pandas as pd

def check_dir(save_dirs):
    if save_dirs:
        if os.path.isdir(save_dirs):
            pass
        else:
            os.makedirs(save_dirs)

def custom_to_networkx(
    data, node_attrs=None, edge_attrs=None, to_undirected=False, remove_self_loops=False
):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.DiGraph` if :attr:`to_undirected` is set to :obj:`True`, or
    an undirected :obj:`networkx.Graph` otherwise.
    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool, optional): If set to :obj:`True`, will return a
            a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
            undirected graph will correspond to the upper triangle of the
            corresponding adjacency matrix. (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)
    """

    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    G.add_nodes_from(range(data.num_nodes))

    values = {}
    for key, item in data.__dict__.items():
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)
        for key in edge_attrs if edge_attrs is not None else []:
            G[u][v][key] = values[key][i]

    for key in node_attrs if node_attrs is not None else []:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    return G


def plot_expl_nc(G, G_true, role, node_idx, args, top_acc):
    G = G.to_undirected()
    if node_idx not in G.nodes():
        G.add_node(node_idx)
        G.nodes()[node_idx]["label"] = 1
    edges, weights = zip(*nx.get_edge_attributes(G, "weight").items())
    nodes, labels = zip(*nx.get_node_attributes(G, "label").items())
    nodes = np.array(nodes)
    labels = np.array(labels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    dict_color_labels = {0: "orange", 1: "green", 2: "green", 3: "green"}
    node_labels = [dict_color_labels[l] for l in labels]
    index_target_node = np.where(nodes == node_idx)[0][0]
    node_labels[index_target_node] = "tab:red"

    true_nodes = np.array(G_true.nodes())
    true_node_labels = ["green"] * len(true_nodes)
    true_node_labels[np.where(true_nodes == node_idx)[0][0]] = "tab:red"

    U = nx.compose(G_true, G)
    # pos=nx.planar_layout(U) #nx.spring_layout(U, k=5, iterations=20, seed=4321)
    weights = np.array(weights)
    weights = np.interp(weights, (weights.min(), weights.max()), (4, 7))

    nx.draw(
        G_true.to_undirected(),
        pos=nx.spring_layout(G_true),
        node_size=1200,
        with_labels=False,
        node_color=true_node_labels,
        edgecolors="black",
        edge_color="black",
        width=7,
        ax=ax1,
    )

    df = pd.DataFrame(index=G.nodes(), columns=G.nodes())
    for row, data in nx.shortest_path_length(G):
        for col, dist in data.items():
            df.loc[row, col] = dist

    df = df.fillna(df.max().max())

    layout = nx.kamada_kawai_layout(G, dist=df.to_dict())

    nx.draw(
        G,
        pos=layout,
        node_size=1200,
        with_labels=False,
        node_color=node_labels,
        edgecolors="black",
        edge_color="black",
        edgelist=edges,
        width=weights,
        ax=ax2,
    )
    date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    # dir_name = f"figures/{args.dataset}/node_{node_idx}/{args.explainer_name}/{args.strategy}_{args.param}"
    dir_name = f"figures/{args.dataset}/node_{node_idx}/{args.explainer_name}/"
    check_dir(dir_name)
    plt.savefig(
        ### os.path.join(dir_name, f"fig_expl_nc_top_{top_acc}_sparsity_{args.param}_{args.hard_mask}_{args.dataset}_{args.explainer_name}_{node_idx}_{date}.pdf")
        os.path.join(
            dir_name,
            f"fig_expl_nc_top_{top_acc}_{args.dataset}_{args.explainer_name}_{node_idx}_{date}.png",
        )
    )
