import scanpy as sc
import torch
import numpy as np
from tqdm import tqdm
import time
from scipy.sparse import issparse
import dgl
import torch.nn.functional as F
import torch.nn as nn

from .utils import build_spatial_graph, gen_clust_embed, set_seed
from .stCluster import stCluster


def train(
        data: sc.AnnData,
        radius: int = None,                                             # generate graph by limit its radius
        knears: int = None,                                             # or KNN
        spatial_img_path: str = None,                                   # save path of spatial graph
        k_neighbors: int= 15,                                           # generate feature graph with k neighbors
        drop_rate: float = .96,                                         # drop rate when deleting edges from different cluster
        build_preclust: bool = True,                                    # use precluster graph or original graph
        preclust_knears: int = 35,                                      # preclust graph generate by knears
        preclust_method: str = 'louvain',                               # pre-cluster method
        preclust_para: float = 1.,                                      # if louvain -> resolution, if kmeans -> knears
        precluster_img_save_path: str = None,                           # save path of the pre-cluster result
        final_graph_img_save_path: str = None,                          # save path of the after pruning graph result
        init_paras: str = None,                                         # load initialized model parameters
        ae_layers: list = [3000, 512, 30],                              # GNN and FC layer
        tmp: float = .5,                                                # temperature value of CL 
        cutting_prob_1: float = .05,                                    # cutting edge rate for view 1
        cutting_prob_2: float = .1,                                     # cutting edge rate for view 2
        threshold: float = .7,                                          # cutting edge rate threshold 
        multi_step: int = 50,                                           # using multi-task loss time
        dec_alpha: float = 1,                                           # define dec alpha
        ae_rate: float = .3,                                            # AE (gene expression rebuild) loss weight
        adj_rate: float = .3,                                           # adjacency matrix rebuild loss weight
        pred_rate: float = .3,                                          # domain prediction loss weight
        epochs: int = 1000,                                             # epoch times
        lr: float = 1e-3,                                               # learning rate
        early_stop_rate: float = 1e-3,                                  # early stop or not 
        dec_precluster: str = 'louvain',                                # dec pre-cluster method
        dec_precluster_para: float = 1.,                                # if louvain -> resolution, if kmeans -> knears
        save_embedding: str = None,                                     # save embedding to the appointed path
        save_model: str = None,                                         # save model parameters to the appointed path
        show: bool = True,                                              # show detail information while training
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',   # use gpu to accelerate computation or not
        embed_name: str = 'embedding',                                  # result embedding will save in adata.obsm[embed_name]
        seed: int = 3407,                                               # set seed (http://arxiv.org/abs/2109.08203)
):
    
    r"""Train stCluster.

    Args:
        data (sc.AnnData): input AnnData format data
        radius (int): generate graph by limit its radius
        knears (int): uses KNN to generate graph
        spatial_img_path (str): save path of spatial graph
        k_neighbors (int): generate feature graph with k neighbors
        drop_rate (float): drop rate when deleting edges from different cluster
        build_preclust (bool): use precluster graph or original graph
        preclust_knears (int): preclust graph generate by knears
        preclust_method (str): pre-cluster method
        preclust_para (float): if uses louvain to do preclustering, you should set
            ``resolution`` resolution, if uses kmeans to do preclustering, you 
            should set ``knears``
        precluster_img_save_path (str): save path of the pre-cluster result
        final_graph_img_save_path (str): save path of the after pruning graph result
        init_paras (str): load initialized model parameters
        ae_layers (list): GNN and FC layer
        tmp (float): temperature value of CL 
        cutting_prob_1 (float): cutting edge rate for view 1
        cutting_prob_2 (float): cutting edge rate for view 2
        threshold (float): cutting edge rate threshold 
        multi_step (int):  using multi-task loss time
        dec_alpha (float): define dec alpha
        ae_rate (float):  AE (gene expression rebuild) loss weight
        adj_rate (float): adjacency matrix rebuild loss weight
        pred_rate (float): domain prediction loss weight
        epochs (int): epoch times
        lr (float): learning rate
        early_stop_rate (float): early stop or not 
        dec_precluster (str): dec pre-cluster method
        dec_precluster_para (float): same as ``preclust_para``
        save_embedding (str): save embedding to the appointed path
        save_model (str): save model parameters to the appointed path
        show (bool): show detail information while training
        device (str): use gpu to accelerate computation or not
        embed_name (str): result embedding will save in adata.obsm[embed_name]
        seed (int): set seed (http://arxiv.org/abs/2109.08203)

    Returns:
        AnnData format result and the spatial graph.
    """

    start_time = time.time()
    adata = data.copy()

    # check input parameters
    assert (((None != radius) and (None == knears)) or ((None != knears) and (None == radius))), '>>> ERROR: You can only use one method to generate graph!'

    if ('highly_variable' in adata.var):
        adata = adata[:, adata.var['highly_variable']]
    if (issparse(adata.X)):
        x = torch.FloatTensor(adata.X.todense())
    else:
        x = torch.FloatTensor(adata.X)

    if (x.shape[1] != ae_layers[0]):
        ae_layers[0] = x.shape[1]
        print(f'>>> WARNING: Input size does NOT match the model size! Shrink/expend the AE layer to {ae_layers} (This may lower the quality of the embedding matrix)!')
    print(f'>>> INFO: Input size {x.shape}.')

    rates = {'gene_rate': ae_rate, 'adj_rate': adj_rate, 'pred_rate': pred_rate}
    set_seed(seed)

    # generate spatial graph
    spatial_graph = build_spatial_graph(
        adata, 
        radius=radius, knears=knears,
        method=None,
        spatial_img_path=spatial_img_path,
        show=show,
    )

    # count init centroid
    cluster_embed, init_embed = gen_clust_embed(
        data=x, 
        method=dec_precluster, 
        embed_dim=ae_layers[-1], 
        para=dec_precluster_para, 
        seed=seed, device=device, show=show
    )

    # build cutting edge graph
    if (build_preclust):
        preclust_graph = build_spatial_graph(
            adata, 
            knears=preclust_knears, k_neighbors=k_neighbors,
            method=preclust_method, precluster_para=preclust_para, 
            drop_rate=drop_rate, alpha=1., delta=1.,
            precluster_img_save_path=precluster_img_save_path,
            final_graph_img_save_path=final_graph_img_save_path, 
            show=show, seed=seed,
        )
    else:
        preclust_graph = spatial_graph

    # generate and train the model
    g = spatial_graph.to(device) 
    x = x.to(device) 
    model = stCluster(
        layers=ae_layers, 
        dec_alpha=dec_alpha, 
        init_centroids=cluster_embed, 
        rates=rates, 
        temp=tmp,
    ).to(device) 
    # use init model parameters
    if (None != init_paras):
        model = torch.load(init_paras, map_location=device)
        model.embed_clust = nn.Parameter(cluster_embed).to(device)
        model.gene_rate = rates['gene_rate']
        model.adj_rate = rates['adj_rate']
        model.pred_rate = rates['pred_rate']
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    embed = init_embed.to(device)
    edge_index = spatial_graph.edges()
    adj = dgl.khop_adj(preclust_graph, 1).to(device)
    spot_count = adata.X.shape[0]
    prev_pred_label = torch.Tensor([0] * spot_count).to(device)
    print('>>> INFO: Finish model preparations, begin to train model, input data size: ({}, {}).'.format(x.shape[0], x.shape[1])) if (show) else None
    
    # train the model
    model.train()
    for epoch in tqdm(range(epochs), desc='>>> INFO: Training'):
        # calculate pruning rate, noise
        with torch.no_grad():
            prob = F.cosine_similarity(embed[edge_index[0],], embed[edge_index[1],], dim=1, eps=1e-6)

        # train CL part and optimize model
        optim.zero_grad()
        embed_1 = model(build_CL_data(g, prob, cutting_prob=cutting_prob_1, threshold=threshold), x, True)
        embed_2 = model(build_CL_data(g, prob, cutting_prob=cutting_prob_2, threshold=threshold), x, True)
        loss = model.CL_loss(embed_1, embed_2)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optim.step()

        if (0 == epoch % multi_step):
            # use multitask to optimize model
            optim.zero_grad()
            embed, pred_gene, q = model(g, x)
            loss = model.multitask_loss(embed, pred_gene, x, q, adj)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optim.step()

            # early stop
            pred_label = torch.argmax(q, dim=1)                                     # generate current predict label
            changing_rate = torch.sum(pred_label != prev_pred_label) / spot_count   # count changing rate
            prev_pred_label = pred_label                                            # save current label
            if (0 < epoch and changing_rate < early_stop_rate):
                print(f">>> INFO: Early stop, changing rate: {changing_rate}, epoch: {epoch}.")
                break
        else:
            embed = model(g, x, True)

    model.eval()
    with torch.no_grad():
        embed, _, _ = model(g, x)
    adata.obsm[embed_name] = embed.cpu().detach().numpy()

    # save embedding 
    if (isinstance(save_embedding, str)):
        np.save(save_embedding, adata.obsm[embed_name], allow_pickle=True, fix_imports=True)
        print('>>> INFO: Successfully save embedding at {}.'.format(save_embedding)) if (show) else None

    # save model parameters
    if (isinstance(save_model, str)):
        torch.save(model, save_model)
        print('>>> INFO: Successfully export model at {}.'.format(save_model)) if (show) else None

    print('>>> INFO: Finish embedding process, total time: {:.3f}s.'.format(time.time() - start_time)) if (show) else None

    return adata, spatial_graph


def build_CL_data(g, prune_prob, cutting_prob=0.45, threshold=0.7):

    # pruning edges
    prune_prob = ((prune_prob.max() - prune_prob) / (prune_prob.max() - prune_prob.mean())) * cutting_prob  # count the cutting probability for each edge
    prune_prob = prune_prob.where(prune_prob < threshold, torch.ones_like(prune_prob) * threshold)          # decide cut edges
    g = dgl.remove_edges(g, torch.nonzero(torch.rand_like(prune_prob) > (1 - prune_prob)).squeeze())        # cute edge
    g = dgl.add_self_loop(dgl.remove_self_loop(g))                                                          # prevent cutting self-loop edges

    return g
