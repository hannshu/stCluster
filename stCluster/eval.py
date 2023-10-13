import scanpy as sc
import torch
import numpy as np
import time
from scipy.sparse import issparse

from .utils import build_spatial_graph
from .stCluster import stCluster


def eval(data: sc.AnnData,
          model_paras_path: str,                # model parameters path
          radius: int = None,                   # generate graph by limit its radius
          knears: int = None,                   # KNN
          ae_layers: list = [3000, 512, 30],    # GNN and FC layer
          save_embedding: str = None,           # save embedding to the appointed path
          save_rebuild_gene: str = None,        # save rebuild gene expression to the appointed path
          show: bool = True,                    # show detail information while training
          device: str = 'cuda' if torch.cuda.is_available() else 'cpu', # use gpu to accelerate computation or not
          embed_name: str = 'embedding',        # result embedding will save in adata.obsm[embed_name]
          ):

    start_time = time.time()
    adata = data.copy()

    # check input parameters
    assert ((radius and not knears) or (knears and not radius)), '>>> ERROR: You can only use one method to generate graph!'

    if ('highly_variable' in adata.var):
        adata = adata[:, adata.var['highly_variable']]
    if (issparse(adata.X)):
        x = torch.FloatTensor(adata.X.todense())
    else:
        x = torch.FloatTensor(adata.X)

    if (x.shape[1] != ae_layers[0]):
        ae_layers = [x.shape[1], 512, 30]
        print(f'>>> WARNING: Input size does NOT match the model size! Shrink/expend the AE layer to {ae_layers} (This may lower the quality of the embedding matrix)!')
    print(f'>>> INFO: Input size {x.shape}.')

    spatial_graph = build_spatial_graph(adata, radius=radius, knears=knears, show=show) # generate spatial graph

    # load model
    g = spatial_graph.to(device)
    x = x.to(device)
    model = stCluster(
        layers=ae_layers, 
        dec_alpha=None, 
        init_centroids=None, 
        rates={'gene_rate': None, 'adj_rate': None, 'pred_rate': None}, 
        temp=None,
    ).to(device)
    model = torch.load(model_paras_path, map_location=device)
    print('>>> INFO: Finish load model, begin to generate embedding and rebuild gene expression, input data size: ({}, {}).'.format(x.shape[0], x.shape[1])) if (show) else None

    # inference
    model.eval()
    with torch.no_grad():
        embed, gene_rebuild, _ = model(g, x)

    # save embedding
    adata.obsm[embed_name] = embed.cpu().detach().numpy()
    adata.layers['denoised_gene'] = gene_rebuild.cpu().detach().numpy()
    if (isinstance(save_embedding, str)):
        np.save(save_embedding, adata.obsm[embed_name], allow_pickle=True, fix_imports=True)
        print('>>> INFO: Successfully save embedding at {}.'.format(save_embedding)) if (show) else None
    if (isinstance(save_rebuild_gene, str)):
        np.save(save_rebuild_gene, gene_rebuild.cpu().detach().numpy(), allow_pickle=True, fix_imports=True)
        print('>>> INFO: Successfully save rebuild gene expression at {}.'.format(save_rebuild_gene)) if (show) else None

    print('>>> INFO: Finish embedding generation process, please use the embedding to do downstream evaluation, total time: {:.3f}s'.format(time.time() - start_time)) if (show) else None

    return adata, spatial_graph
