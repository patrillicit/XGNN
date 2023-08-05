#import torch.nn.quantized
import torch
#import torch.nn.quantized
from Datasets.CitationDatasets import *
import Models
from Models.two_layer_gnn import *
from Models.three_layer_gnn import *
from Models.four_layer_gnn import *
#from Models.base import build_base
#from Explainers.ExplanationManager import explain_gnnexplainer_node, explain_pgmexplainer_node, explain_pgexplainer_node
#from Visualizations.Visualize import plot_expl_nc, custom_to_networkx
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from Explainers.pytorch_geo import *
from torch_geometric.explain import ThresholdConfig
from Visualizations.subgraph_plotting import *


def start():
    """
    Dataset Import

    """
    cora = get_cora()
    print(cora.data)
    pubmed = get_pubmed()


    create_csv2(cora)
    raw_cora_as_Data = Data(x=cora.x, edge_index=cora.edge_index)  # ,y=cora.y
    raw_cora_as_Graph = to_networkx(raw_cora_as_Data, to_undirected=True)
    #nx.draw(raw_cora_as_Graph)
    #plt.savefig("rawgraph.png")

    """
    Model Training
    
    """

    """build the models"""
    #2-layer GNN
    model1, num_layers = build_base(cora)

    #3-layer GNN
    model2, num_layers = build_three_layer(cora)

    model3, num_layers = build_2_layer_no_drop(cora)
    quit()
    """load the saved models"""
    two_layer_model = GCN2(cora.num_node_features, cora.num_classes)
    two_layer_model.load_state_dict(torch.load(r'C:\Users\Patrick\OneDrive - student.kit.edu\07 WS 22-23 BT\XGNN\two_layer_model.pth'))

    three_layer_model = GCN3(cora.num_node_features, cora.num_classes)
    three_layer_model.load_state_dict(
        torch.load(r'C:\Users\Patrick\OneDrive - student.kit.edu\07 WS 22-23 BT\XGNN\three_layer_model.pth'))

    """
    XAI-Methods
    1) GNNExplainer from PytorchGeometric for experiment series
   
    """
    dataset = cora
    models = [two_layer_model, three_layer_model]
    nodes = [8, 88, 90] #[1, 8, 10, 12, 21,88, 90, 98]
    thresholding = [ [None, 0.01, None], ["topk", None, 0.5], ["topk", None, 0.75], ["topk", None, 0.9], ["topk", None, 10], ["topk", None, 25]] #[ThresholdConfig("topk_hard", 10), 0.6]
    colors = ["red", "orange", "yellow", "green", "blue", "purple", "grey"]


    def run_gnnexplainer_experiment():

        for nod in nodes:
            for mod in models:
                #print(mod)
                count = 0
                sub_nodes, sub_edges = plot_k_hop_subgraph(nod,mod.num_layers,dataset.edge_index, dataset.data.y, colors)

                #draw_graph(nod,mod.num_layers,cora.edge_index)
                for thres in thresholding:
                    """ Execute XAI Method """
                    path = r"C:\Users\Patrick\OneDrive - student.kit.edu\07 WS 22-23 BT\Experiments\_node"+str(nod)+"_layers"+str(mod.num_layers)+"thres_type"+str(count)
                    k= thres[2]
                    if thres[2] is not None:
                        if thres[2] < 1:
                            k = round(thres[2] * len(sub_edges[0]))

                    explainer, explain = ptg_gnnexplainer(model=mod, data=dataset.data, node=nod, thresholding=thres[0], k=k, path= path + ".gv.pdf")
                    current_subset = get_edges_and_nodes(explain, thres[1]) #transform result to DATA type


                    torch.set_printoptions(threshold=10_000)
                    edge_mask_bool = torch.where(explain.edge_mask < 0.001, False, True)
                    edge_mask_values = torch.masked_select(explain.edge_mask, edge_mask_bool)
                    torch.set_printoptions(threshold=10_000)
                    #print(edge_mask_values)


                    print(current_subset)
                    print(current_subset.nodes)
                    print(current_subset.edges)
                    prediction_labels = []
                    prediction_prob = []
                    for item in explain.prediction.tolist():
                        prediction_prob.append(max(item))
                        prediction_labels.append(item.index(max(item)))

                    """Metrics"""
                    #Sparsity: Doble edges in raw dataset and sometimes in the explanation
                    sparsity = 1 - ( (len(current_subset.edges[0])+len(current_subset.nodes)) / (len(dataset.data.x) + len(dataset.data.edge_index[0])))


                    #Fidelity+
                    initial_node_index = []
                    node_mask = []
                    print(type(current_subset.nodes))
                    current_subset_nodes_removed = current_subset.nodes.copy()
                    print(current_subset_nodes_removed)
                    if nod in current_subset_nodes_removed:
                        current_subset_nodes_removed.remove(nod)
                    print(current_subset_nodes_removed)
                    for pos in range(0,2708):
                        initial_node_index.append(pos)
                        if pos in current_subset_nodes_removed:
                            node_mask.append(False)
                        else:
                            node_mask.append(True)


                    nodes_as_list = dataset.data.x.tolist()

                    from itertools import compress
                    #subset with node vectors
                    nodes_subset = list(compress(nodes_as_list, node_mask))
                    #subset with node indcies
                    initial_node_index_subset = list(compress(initial_node_index, node_mask))

                    """
                    edge_mask = []
                    initial_edge_index = []
                    edges_as_list = dataset.data.edge_index.tolist()
                    edges_as_list_t = np.array(edges_as_list).T.tolist()

                    current_subset_edges_t = np.array(current_subset.edges).T.tolist()

                    for row in edges_as_list_t:#range(len(current_subset.edges[0]))
                        initial_edge_index.append(row)
                        flag = True
                        for pair in current_subset_edges_t:
                            if row[0] == pair[0] and row[1] == pair[1]:
                                edge_mask.append(False)
                                flag = False
                                break
                        if flag == True:
                            edge_mask.append(True)


                    print(len(edges_as_list_t))
                    print(len(edge_mask))
                    print(edge_mask.count(False))
                    edges_subset = list(compress(edges_as_list_t,edge_mask))
                    edges_subset = np.array(edges_subset).T.tolist()


                    edge_index = torch.Tensor(edges_subset)
                    edge_index = edge_index.type(torch.int64)
                    """

                    nodes_to_keep_tensor = torch.Tensor(initial_node_index_subset)
                    nodes_to_keep_tensor = nodes_to_keep_tensor.type(torch.int64)

                    from torch_geometric.utils import subgraph
                    new_edge_index, attr_index = subgraph(nodes_to_keep_tensor, edge_index= dataset.data.edge_index, relabel_nodes=True)
                    new_edge_index = new_edge_index.type(torch.int64)
                    #use original nodes and edges to convert to original nodes and edges
                    fp_data = Data(x=torch.Tensor(nodes_subset), edge_index=new_edge_index, initial_node_index=initial_node_index_subset )
                    print(fp_data)


                    explainer_fp, explain_fp = ptg_gnnexplainer(model=mod, data=fp_data, node=nod, thresholding=thres[0], k=k,
                                                          path=path + "graph_without_imp_nodes.gv.pdf")
                    current_subset_fp = get_edges_and_nodes(explain_fp, thres[1])
                    nodes_reconverted = []
                    for node in current_subset_fp.nodes:
                        nodes_reconverted.append(initial_node_index_subset[node])

                    edges_reconverted = [[],[]]
                    index = 0
                    for edge in current_subset_fp.edges:
                        for ele in edge:
                            edges_reconverted[index].append(initial_node_index_subset[ele])
                        index = index + 1

                    prediction_labels_fp = []
                    prediction_prob_fp = []
                    for item in explain_fp.prediction.tolist():
                        prediction_prob_fp.append(max(item))
                        prediction_labels_fp.append(item.index(max(item)))

                    draw_graph_without_imp_nodes(node_index=nodes_reconverted, edge_index=edges_reconverted, y=dataset.data.y,
                               prediction=prediction_labels, prediction_index= current_subset_fp.nodes, colors=colors, path=path + "_without_imp_nodes")


                    index_in_subset = initial_node_index_subset.index(nod)
                    print(index_in_subset)
                    prob1 = prediction_prob[nod]
                    prob2 = explain_fp.prediction.tolist()[index_in_subset][prediction_labels[nod]]
                    fidelity_p_prob = prob1 - prob2
                    print(prob1)
                    print(prob2)

                    #Accuracy Fidelity Score from Pytorch Geometric
                    from torch_geometric.explain.metric import fidelity
                    fidelity_p, fidelity_n = fidelity(explainer, explain)

                    """ Create visualized graph """
                    draw_graph(node_index=current_subset.nodes, edge_index=current_subset.edges, y=dataset.data.y, prediction=prediction_labels, colors=colors, path=path, edge_score=current_subset.edge_score)
                    count= count +1

                    """ Create Output File """
                    d = [current_subset.nodes, current_subset.edges[0], current_subset.edges[1], [sparsity] , [fidelity_p,fidelity_n, fidelity_p_prob], prediction_prob, prediction_labels, edge_mask_values]
                    from itertools import zip_longest
                    df = pd.DataFrame(zip_longest(*d, fillvalue=''), columns=['Nodes', 'Edge-source', 'Edge-target', 'Sparsity', 'Fidelity +/-', 'Prediction Prob', 'Prediction Label', 'All edge imp values'])
                    #df.rename(columns={'0':'Nodes', '1':'Edge source', '2':'Edge:target', '3':'Sparsity'}, inplace=True)
                    print(df.head(15))
                    df.to_csv(path + ".csv")

    run_gnnexplainer_experiment()


    """ CFGNNExplainer 
    import time
    from Explainers.CFGNNExplainer.utils.utils import normalize_adj, get_neighbourhood, safe_open
    from Explainers.CFGNNExplainer.cf_explanation.cf_explainer import CFExplainer
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='syn1')

    # Based on original GCN models -- do not change
    parser.add_argument('--hidden', type=int, default=20, help='Number of hidden units.')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of convolutional layers.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (between 0 and 1)')

    # For explainer
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for explainer')
    parser.add_argument('--optimizer', type=str, default="SGD", help='SGD or Adadelta')
    parser.add_argument('--n_momentum', type=float, default=0.0, help='Nesterov momentum')
    parser.add_argument('--beta', type=float, default=0.5, help='Tradeoff for dist loss')
    parser.add_argument('--num_epochs', type=int, default=500, help='Num epochs for explainer')
    parser.add_argument('--device', default='cpu', help='CPU or GPU.')
    args = parser.parse_args()

    adjMatrix = [[0 for i in range(len(cora.data.y))] for k in range(len(cora.data.y))]

    # scan the arrays edge_u and edge_v
    for i in range(len(cora.data.edge_index[0])):
        u = cora.data.edge_index[0][i]
        v = cora.data.edge_index[1][i]
        adjMatrix[u][v] = 1

    adj = torch.Tensor(adjMatrix).squeeze()  # Does not include self loops
    features = torch.Tensor(cora.data.x).squeeze()
    labels = torch.tensor(cora.data.y).squeeze()
    idx_train = torch.tensor(data["train_idx"])
    idx_test = torch.tensor(data["test_idx"])
    edge_index = dense_to_sparse(adj)


    def run_cfgnnexplainer_experiment():
        test_cf_examples = []
        start = time.time()
        for i in idx_test[:]:
            sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(int(i), edge_index, args.n_layers + 1, features,
                                                                         labels)
            new_idx = node_dict[int(i)]

            # Check that original model gives same prediction on full graph and subgraph
            with torch.no_grad():
                print("Output original model, full adj: {}".format(output[i]))
                print("Output original model, sub adj: {}".format(model(sub_feat, normalize_adj(sub_adj))[new_idx]))

            # Need to instantitate new cf model every time because size of P changes based on size of sub_adj
            explainer = CFExplainer(model=model,
                                    sub_adj=sub_adj,
                                    sub_feat=sub_feat,
                                    n_hid=args.hidden,
                                    dropout=args.dropout,
                                    sub_labels=sub_labels,
                                    y_pred_orig=y_pred_orig[i],
                                    num_classes=len(labels.unique()),
                                    beta=args.beta,
                                    device=args.device)

            if args.device == 'cuda':
                model.cuda()
                explainer.cf_model.cuda()
                adj = adj.cuda()
                norm_adj = norm_adj.cuda()
                features = features.cuda()
                labels = labels.cuda()
                idx_train = idx_train.cuda()
                idx_test = idx_test.cuda()

            cf_example = explainer.explain(node_idx=i, cf_optimizer=args.optimizer, new_idx=new_idx, lr=args.lr,
                                           n_momentum=args.n_momentum, num_epochs=args.num_epochs)
            test_cf_examples.append(cf_example)
            print("Time for {} epochs of one example: {:.4f}min".format(args.num_epochs, (time.time() - start) / 60))
        print("Total time elapsed: {:.4f}s".format((time.time() - start) / 60))
        print("Number of CF examples found: {}/{}".format(len(test_cf_examples), len(idx_test)))

        # Save CF examples in test set

        with safe_open(
                "../results/{}/{}/{}_cf_examples_lr{}_beta{}_mom{}_epochs{}_seed{}".format(args.dataset, args.optimizer,
                                                                                           args.dataset,
                                                                                           args.lr, args.beta,
                                                                                           args.n_momentum, args.num_epochs,
                                                                                           args.seed), "wb") as f:
            pickle.dump(test_cf_examples, f)

    """


    def run_pgexplainer_experiment():

        for nod in nodes:
            for mod in models:
                #print(mod)
                count = 0
                plot_k_hop_subgraph(nod,mod.num_layers,dataset.edge_index, dataset.data.y, colors)
                #draw_graph(nod,mod.num_layers,cora.edge_index)
                for thres in thresholding:
                    """ Execute XAI Method """
                    path = r"C:\Users\Patrick\OneDrive - student.kit.edu\07 WS 22-23 BT\Experiments\_node"+str(nod)+"_layers"+str(mod.num_layers)+"thres_type"+str(count)
                    explainer, explain = ptg_pgexplainer(model=mod, data=dataset.data, node=nod, thresholding=thres[0], path= path + ".gv.pdf")
                    current_subset = get_edges_and_nodes(explain, thres[1]) #transform result to DATA type


                    torch.set_printoptions(threshold=10_000)
                    edge_mask_bool = torch.where(explain.edge_mask < 0.001, False, True)
                    edge_mask_values = torch.masked_select(explain.edge_mask, edge_mask_bool)
                    torch.set_printoptions(threshold=10_000)
                    #print(edge_mask_values)


                    print(current_subset)
                    print(current_subset.nodes)
                    print(current_subset.edges)
                    prediction_labels = []
                    prediction_prob = []
                    for item in explain.prediction.tolist():
                        prediction_prob.append(max(item))
                        prediction_labels.append(item.index(max(item)))

                    """Metrics"""
                    #Sparsity: Doble edges in raw dataset and sometimes in the explanation
                    sparsity = 1 - ( (len(current_subset.edges[0])+len(current_subset.nodes)) / (len(dataset.data.x) + len(dataset.data.edge_index[0])))


                    #Fidelity+
                    initial_node_index = []
                    node_mask = []
                    print(type(current_subset.nodes))
                    current_subset_nodes_removed = current_subset.nodes.copy()
                    print(current_subset_nodes_removed)
                    if nod in current_subset_nodes_removed:
                        current_subset_nodes_removed.remove(nod)
                    print(current_subset_nodes_removed)
                    for pos in range(0,2708):
                        initial_node_index.append(pos)
                        if pos in current_subset_nodes_removed:
                            node_mask.append(False)
                        else:
                            node_mask.append(True)


                    nodes_as_list = dataset.data.x.tolist()

                    from itertools import compress
                    #subset with node vectors
                    nodes_subset = list(compress(nodes_as_list, node_mask))
                    #subset with node indcies
                    initial_node_index_subset = list(compress(initial_node_index, node_mask))

                    """
                    edge_mask = []
                    initial_edge_index = []
                    edges_as_list = dataset.data.edge_index.tolist()
                    edges_as_list_t = np.array(edges_as_list).T.tolist()

                    current_subset_edges_t = np.array(current_subset.edges).T.tolist()

                    for row in edges_as_list_t:#range(len(current_subset.edges[0]))
                        initial_edge_index.append(row)
                        flag = True
                        for pair in current_subset_edges_t:
                            if row[0] == pair[0] and row[1] == pair[1]:
                                edge_mask.append(False)
                                flag = False
                                break
                        if flag == True:
                            edge_mask.append(True)


                    print(len(edges_as_list_t))
                    print(len(edge_mask))
                    print(edge_mask.count(False))
                    edges_subset = list(compress(edges_as_list_t,edge_mask))
                    edges_subset = np.array(edges_subset).T.tolist()


                    edge_index = torch.Tensor(edges_subset)
                    edge_index = edge_index.type(torch.int64)
                    """

                    nodes_to_keep_tensor = torch.Tensor(initial_node_index_subset)
                    nodes_to_keep_tensor = nodes_to_keep_tensor.type(torch.int64)

                    from torch_geometric.utils import subgraph
                    new_edge_index, attr_index = subgraph(nodes_to_keep_tensor, edge_index= dataset.data.edge_index, relabel_nodes=True)
                    new_edge_index = new_edge_index.type(torch.int64)
                    #use original nodes and edges to convert to original nodes and edges
                    fp_data = Data(x=torch.Tensor(nodes_subset), edge_index=new_edge_index, initial_node_index=initial_node_index_subset )
                    print(fp_data)


                    explainer_fp, explain_fp = ptg_pgexplainer(model=mod, data=fp_data, node=nod, thresholding=thres[0],
                                                          path=path + "graph_without_imp_nodes.gv.pdf")
                    current_subset_fp = get_edges_and_nodes(explain_fp, thres[1])
                    nodes_reconverted = []
                    for node in current_subset_fp.nodes:
                        nodes_reconverted.append(initial_node_index_subset[node])

                    edges_reconverted = [[],[]]
                    index = 0
                    for edge in current_subset_fp.edges:
                        for ele in edge:
                            edges_reconverted[index].append(initial_node_index_subset[ele])
                        index = index + 1

                    prediction_labels_fp = []
                    prediction_prob_fp = []
                    for item in explain_fp.prediction.tolist():
                        prediction_prob_fp.append(max(item))
                        prediction_labels_fp.append(item.index(max(item)))

                    draw_graph_without_imp_nodes(node_index=nodes_reconverted, edge_index=edges_reconverted, y=dataset.data.y,
                               prediction=prediction_labels, prediction_index= current_subset_fp.nodes, colors=colors, path=path + "_without_imp_nodes")


                    index_in_subset = initial_node_index_subset.index(nod)
                    print(index_in_subset)
                    prob1 = prediction_prob[nod]
                    prob2 = explain_fp.prediction.tolist()[index_in_subset][prediction_labels[nod]]
                    fidelity_p_prob = prob1 - prob2
                    print(prob1)
                    print(prob2)

                    #Accuracy Fidelity Score from Pytorch Geometric
                    from torch_geometric.explain.metric import fidelity
                    fidelity_p, fidelity_n = fidelity(explainer, explain)

                    """ Create visualized graph """
                    draw_graph(node_index=current_subset.nodes, edge_index=current_subset.edges, y=dataset.data.y, prediction=prediction_labels, colors=colors, path=path, edge_score=current_subset.edge_score)
                    count= count +1

                    """ Create Output File """
                    d = [current_subset.nodes, current_subset.edges[0], current_subset.edges[1], [sparsity] , [fidelity_p,fidelity_n, fidelity_p_prob], prediction_prob, prediction_labels, edge_mask_values]
                    from itertools import zip_longest
                    df = pd.DataFrame(zip_longest(*d, fillvalue=''), columns=['Nodes', 'Edge-source', 'Edge-target', 'Sparsity', 'Fidelity +/-', 'Prediction Prob', 'Prediction Label', 'All edge imp values'])
                    #df.rename(columns={'0':'Nodes', '1':'Edge source', '2':'Edge:target', '3':'Sparsity'}, inplace=True)
                    print(df.head(15))
                    df.to_csv(path + ".csv")

    #run_pgexplainer_experiment()





if __name__ == '__main__':
    start()

