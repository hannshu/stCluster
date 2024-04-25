import os
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import dgl
from collections import Counter
import networkx as nx
import scanpy as sc
import squidpy as sq
from sklearn.preprocessing import LabelEncoder 
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


plt.rcParams['font.sans-serif'] = ['Times New Roman']


# set seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def build_spatial_graph(adata: sc.AnnData, 
                        seed=0,
                        radius=None,
                        knears=None,
                        k_neighbors=None,
                        method=None,
                        precluster_para=1.,
                        drop_rate=.96,
                        add_feature_graph=False,
                        alpha=None, 
                        beta=None, 
                        gama=None, 
                        delta=None,
                        need_attr=False,
                        show=False,
                        spatial_img_path=None,
                        precluster_img_save_path=None, 
                        final_graph_img_save_path=None,
    ):

    # generate spatial graph
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['row', 'col']
    if (radius):
        nbrs = NearestNeighbors(radius=radius).fit(coor)
        _, indices = nbrs.radius_neighbors(coor, return_distance=True)
    else:
        nbrs = NearestNeighbors(n_neighbors=knears+1).fit(coor)
        _, indices = nbrs.kneighbors(coor)

    edge_list = []
    for i in range(len(indices)):
        for j in range(len(indices[i])):
            # donnot add self-loop edge to the graph 
            if (i != indices[i][j]):
                edge_list.append([i, indices[i][j]])
    
    if (add_feature_graph):
        sc.pp.neighbors(adata, random_state=seed, use_rep='X', n_neighbors=k_neighbors) # generate feature graph

        print('>>> INFO: Begin to build the graph by gene expressions and spatial informations.') if (show) else None
        # add edges from feature graph
        # edge attribute:
        # - self-loop(node1-node1) -> alpha+beta+gama
        # - edge *only* from spatial graph -> alpha
        # - edge *only* from feature graph -> beta
        # - edge from both spatial graph and feature graph -> gama
        # intuitively gama > beta > alpha
        edge_index, edge_attr = add_edges_from_feature_graph(
            adata, 
            spatial_edge_index=edge_list,
            alpha=alpha, beta=beta, gama=gama
        )

        print('>>> INFO: Finish buiding the graph.') if (show) else None
    else:
        # edge attribute: 
        # - self-loop(node1-node1) -> 1-alpha
        # - other edges(node1-node2) -> alpha
        edge_index = edge_list + [[x, x] for x in range(adata.X.shape[0])]
        edge_attr = [[alpha] for _ in range(len(edge_list))] + [[1 - alpha] for _ in range(adata.X.shape[0])] if (alpha) else None
    visualize_graph(adata, edges=edge_index, save_path=spatial_img_path) if (spatial_img_path) else None

    # do pre-cluster
    if (method):
        if ('louvain' == method):
            if (False == add_feature_graph):
                sc.pp.neighbors(adata, random_state=seed, use_rep='X', n_neighbors=k_neighbors) # generate feature graph
            louvain_modify(adata, resolution=precluster_para, seed=seed)
        elif ('kmeans' == method):
            kmeans_modify(adata, k=precluster_para, seed=seed)
        
        visualize_graph(adata, edges=edge_index, save_path=precluster_img_save_path, color_label=['pre_trained_label']) if (precluster_img_save_path) else None
        print('>>> INFO: Finish pre-cluster, result image is saved at "{}", begin to prune graph.'.format(precluster_img_save_path)) if (show) else None
        
        # prune the graph by pre-cluster result:
        # divide the entire graph to separate small graphs
        # then random add some edges between those individual graphs
        # random edges' attribute should be assigned as delta
        # intuitively delta should less than alpha
        edge_index, edge_attr = prune_graph(adata, edge_index, edge_attr, delta, save_path=final_graph_img_save_path, show=show, drop_rate=drop_rate)
    else:
        edge_index = torch.LongTensor(np.array(edge_index).T)
        edge_attr = torch.FloatTensor(edge_attr) if (alpha) else None

    print('>>> INFO: Graph contains {} edges, average {:.3f} edges per node.'.format(edge_index.shape[1], edge_index.shape[1] / adata.X.shape[0])) if (show) else None
    print('>>> INFO: Build graph success!') if (show) else None
    if (need_attr):
        return edge_index, edge_attr
    else:
        return dgl.graph((edge_index[0], edge_index[1]))


def add_edges_from_feature_graph(adata: sc.AnnData, spatial_edge_index, alpha, beta, gama):
    feature_edge_index = np.array(np.nonzero(adata.obsp['connectivities'])).T.tolist()

    # edges only from spatial graph
    spatial_edges = [x for x in spatial_edge_index if (x not in feature_edge_index)]
    spatial_edges_attr = [alpha] * len(spatial_edges)
    # edges only from feature graph
    feature_edges = [x for x in feature_edge_index if (x not in spatial_edge_index)]
    feature_edges_attr = [beta] * len(feature_edges)
    # edges from both spatial graph and feature graph
    combine_edges = [
        x for x in feature_edge_index+spatial_edge_index
        if ((x in spatial_edge_index) and (x in feature_edge_index))
    ]
    combine_edges_attr = [gama] * len(combine_edges)
    # self-loop edges
    node_edges = [[x, x] for x in range(adata.X.shape[0])]
    node_edges_attr = [alpha + beta + gama] * len(node_edges)
    # edge_index = [spatial_graph_only, feature_graph_only, collective, self-loop]
    # edge_attr = [alpha, beta, gama, alpha+beta+gama]
    edge_index = np.array(spatial_edges + feature_edges + combine_edges + node_edges)
    edge_attr = np.array(spatial_edges_attr + feature_edges_attr + combine_edges_attr + node_edges_attr)

    return edge_index, edge_attr


def kmeans_modify(adata: sc.AnnData, k, seed=2022):
    kmeans_label = KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(adata.X)
    adata.obs['pre_trained_label'] = [str(x) for x in LabelEncoder().fit_transform(kmeans_label)]


def louvain_modify(adata: sc.AnnData, resolution, seed=2022):
    sc.tl.louvain(adata, resolution=resolution, key_added='pre_trained_label', random_state=seed)
    adata.obs['pre_trained_label'] = [str(x) for x in LabelEncoder().fit_transform(adata.obs['pre_trained_label'])]


def prune_graph(adata: sc.AnnData, edge_index, edge_attr, delta, save_path, show, drop_rate):
    # refine label
    adata.obs['refined_label'], original_graph = refine_label(
        labels=adata.obs['pre_trained_label'].tolist(),
        edge_index=edge_index
        )

    # prune graph: delete edges from different cluster
    label = dict(zip(range(adata.X.shape[0]), adata.obs['refined_label']))

    edges, edge_attrs = edge_sampling(
        g=edge_index, 
        dropping_rate=drop_rate, 
        edge_attr=edge_attr, 
        label=label
        )

    edge_index, edge_attr = check_lonely(
        original_graph=original_graph, 
        current_edges=edges,
        edge_attr=edge_attrs,
        delta=delta
        )   # check wether have lonely node (donnot link to any other nodes)
    visualize_graph(adata, edges=edge_index.detach().numpy().T, save_path=save_path, color_label=['refined_label']) if (save_path) else None
    print('>>> INFO: Finish pruning graph, result image is saved at "{}".'.format(save_path)) if (show) else None

    # delete useless adjacency matrix
    del adata.obs['pre_trained_label']
    del adata.obs['refined_label']

    return edge_index, edge_attr


# visualize the graph using squidpy
def visualize_graph(adata: sc.AnnData, edges, save_path, color_label=None):
    without_ring = np.array([[edge[0], edge[1]] for edge in edges if (edge[0] != edge[1])]).T
    adata.obsp['visualize'] = dgl.graph((without_ring[0], without_ring[1])).adj(scipy_fmt='coo')
    if (None == color_label):
        adata.obs['default'] = ['1'] * adata.X.shape[0]
        color_label = ['default']
    sq.pl.spatial_scatter(adata, connectivity_key="visualize", img=False, color=color_label, edges_width=0.25, title='', axis_label=['', ''], edges_color='black')
    plt.savefig(save_path, dpi=1600)
    del adata.obsp['visualize']


def refine_label(labels, edge_index):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(labels)))
    graph.add_edges_from(edge_index)

    result_labels = []
    for node in range(len(labels)):
        neighbor_labels = [labels[x] for x in list(graph.neighbors(node))]
        freq_label = Counter(neighbor_labels).most_common(1)
        if (freq_label[0][1] > len(neighbor_labels) // 2):
            result_labels.append(freq_label[0][0])
        else:
            result_labels.append(labels[node])

    return result_labels, graph


def check_lonely(original_graph: nx.Graph, current_edges, edge_attr: list, delta):
    current_graph = nx.Graph()
    current_graph.add_nodes_from(original_graph.nodes)
    current_graph.add_edges_from([(x[0], x[1]) for x in current_edges.T])
    edge_index0 = list(current_edges[0])
    edge_index1 = list(current_edges[1])

    for node in current_graph.nodes:
        neighbors = list(current_graph.neighbors(node))
        # if a node has no neighbor -> random get a edge from the original graph
        if ([node] == neighbors):
            original_neighbors = list(original_graph.neighbors(node))
            original_neighbors.remove(node) # delete self-loop
            # this node should have neighbors before
            if ([] != original_neighbors):
                edge_index0.append(node)
                edge_index1.append(int(np.random.choice(np.array(original_neighbors), size=1)))
                edge_attr.append([delta])

    edge_index = torch.LongTensor(np.array([np.array(edge_index0), np.array(edge_index1)]))
    edge_attr = torch.FloatTensor(np.array(edge_attr))
    return edge_index, edge_attr


def gen_clust_embed(data, method, embed_dim, para, seed, device, show=False):
    x = PCA(n_components=embed_dim).fit_transform(data)

    if ('kmeans' == method):
        kmeans = KMeans(n_clusters=para, random_state=seed, n_init=20)
        pred = kmeans.fit_predict(x)
    elif ('louvain' == method):
        adata = sc.AnnData(x)
        sc.pp.neighbors(adata, random_state=seed, use_rep='X', n_neighbors=15)
        sc.tl.louvain(adata, resolution=para, random_state=seed)
        pred = adata.obs['louvain'].astype(int).to_numpy()

    df = pd.DataFrame(x, index=range(x.shape[0]))
    df.insert(loc=1, column='labels', value=pred)
    cluster_embed = torch.FloatTensor(np.asarray(df.groupby("labels").mean())).to(device)

    print('>>> INFO: Finish generate precluster embedding!') if (show) else None
    return cluster_embed, torch.FloatTensor(x)


def edge_sampling(g, dropping_rate, edge_attr=None, label=None):
    if (isinstance(g, list)):
        edge_index = g
    else:
        edge_index = np.array([g.edges()[0].detach().numpy(), g.edges()[1].detach().numpy()]).T
    df = pd.DataFrame(edge_index)
    if (edge_attr):
        df.insert(loc=2, column='2', value=edge_attr)
        df.columns = ['node1', 'node2', 'attr']
    else:
        df.columns = ['node1', 'node2']
        df = df[df['node1'] != df['node2']] # delete self-loop

    if (label):
        df['node1_label'] = df['node1'].map(label)
        df['node2_label'] = df['node2'].map(label)
        mismatch_list = np.array(df[df['node1_label'] != df['node2_label']].index)
    else:
        mismatch_list = np.array(range(df.shape[0]))

    # save some edges: saving_rate=1-drop_rate
    drop_index = np.random.choice(mismatch_list, size=int(len(mismatch_list)*dropping_rate), replace=False)
    modif_df = df.drop(index=drop_index)
    edges = np.array([np.array(modif_df['node1']), np.array(modif_df['node2'])])

    if (edge_attr):
        return edges, list(modif_df['attr'])
    else:
        return edges


def cal_cluster_score(true_label, pred_label, method='ARI'):
    if ('ARI' == method):
        from sklearn.metrics import adjusted_rand_score
        return adjusted_rand_score(true_label, pred_label)
    elif ('NMI' == method):
        from sklearn.metrics import normalized_mutual_info_score
        return normalized_mutual_info_score(true_label, pred_label)
    elif ('AMI' == method):
        from sklearn.metrics import adjusted_mutual_info_score 
        return adjusted_mutual_info_score(true_label, pred_label)
    elif ('v_measure_score' == method):
        from sklearn.metrics import v_measure_score
        return v_measure_score(true_label, pred_label)
    elif ('silhouette_score' == method):
        from sklearn.metrics import silhouette_score
        # if counting silhouette_score, you have to use adata[obs_df.rows] as true_label
        return silhouette_score(true_label, pred_label)
    else:
        print(f'>>> ERROR: Method {method} is not supported!')


def draw_domain_result(adata, label_name, save_path, title, **kwargs):
    adata.obs['draw_target'] = [str(x) for x in adata.obs[label_name]]

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(f'{save_path}.pdf') as pdf:
        plt.subplots(tight_layout=True)
        sc.pl.spatial(adata, color=['draw_target'], show=False, **kwargs)
        plt.tight_layout()
        plt.title(title)
        plt.xlabel('')
        plt.ylabel('')
        pdf.savefig(bbox_inches='tight')
        plt.close()

    del adata.obs['draw_target']


def draw_umap_result(adata, embedding_name, save_path, title, **kwargs):
    sc.pp.neighbors(adata, use_rep=embedding_name, key_added='umap_key')
    sc.tl.umap(adata, neighbors_key='umap_key')

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(f'{save_path}.pdf') as pdf:
        plt.subplots(tight_layout=True)
        sc.pl.umap(adata, color=['cluster'], show=False, **kwargs)
        plt.tight_layout()
        plt.title(title)
        pdf.savefig(bbox_inches='tight')
        plt.close()

    del adata.obsp['umap_key_distances']
    del adata.obsp['umap_key_connectivities']


def gen_adata(feature: np.ndarray, coors: np.ndarray, meta_data: pd.DataFrame = None, gene_name: list = None, spot_name: list = None):

    # load gene expression matrix
    adata = sc.AnnData(csr_matrix(feature))

    # load spatial coordination
    coors = np.array(coors)
    if (2 == coors.shape[1]):
        adata.obsm['spatial'] = coors
    else:
        adata.obsm['spatial'] = coors.T

    # set misc
    if (isinstance(meta_data, pd.DataFrame)):
        adata.obs = meta_data
    if (gene_name):
        adata.var.index = gene_name
    if (spot_name):
        adata.obs.index = spot_name

    return adata


def cal_MARI(list1: list, list2: list):

    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr

    # Load aricode
    aricode = importr("aricode")

    # Call the MARI function
    mari_result = aricode.MARI(
        robjects.r['unlist'](list(list1)), robjects.r['unlist'](list(list2))
    )

    return mari_result[0]


def cal_modularity(edge_list: np.ndarray, label: list):

    g = nx.Graph()
    g.add_edges_from(edge_list)

    label_mapping = {list(set(label))[i]: i for i in range(len(set(label)))}
    partition = [[] for _ in range(len(set(label)))]
    for j in range(len(label)):
        partition[label_mapping[label[j]]-1].append(j)

    print(f'Modularity: {nx.community.modularity(g, [set(j) for j in partition])}')


def cal_partition_quality(edge_list: np.ndarray, label: list):

    g = nx.Graph()
    g.add_edges_from(edge_list)

    label_mapping = {list(set(label))[i]: i for i in range(len(set(label)))}
    partition = [[] for _ in range(len(set(label)))]
    for j in range(len(label)):
        partition[label_mapping[label[j]]-1].append(j)

    quality = nx.community.partition_quality(g, [set(j) for j in partition])
    print(f'Coverage: {quality[0]}, Performance: {quality[1]}')
