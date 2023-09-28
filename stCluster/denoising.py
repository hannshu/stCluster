import scanpy as sc
import torch
import numpy as np
from tqdm import tqdm
import time
from scipy.sparse import issparse
import dgl
import torch.nn.functional as F
import torch.nn as nn

from .utils import set_seed
from .stCluster import Denoising


def train(
        data: sc.AnnData,
        spatial_graph: dgl.DGLGraph,                                    # temperature value of CL 
        ae_layers: list = [30, 512, 3000],                              # GNN and FC layer
        epochs: int = 600,                                              # epoch times
        lr: float = 1e-3,                                               # learning rate
        show: bool = True,                                              # show detail information while training
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',   # use gpu to accelerate computation or not
        embed_name: str = 'embedding',                                  # result embedding will save in adata.obsm[embed_name]
        save_denoised_gene: str = None,                                 # 
        seed: int = 3407,                                               # set seed (http://arxiv.org/abs/2109.08203)
):

    start_time = time.time()
    adata = data.copy()

    x = torch.FloatTensor(adata.obsm[embed_name])
    if (issparse(adata.X)):
        gene = torch.FloatTensor(adata.X.todense())
    else:
        gene = torch.FloatTensor(adata.X)

    if (x.shape[1] != ae_layers[0]):
        ae_layers[0] = x.shape[1]
        print(f'>>> WARNING: Input size does NOT match the model size! Shrink/expend the AE layer to {ae_layers} (This may lower the quality of the embedding matrix)!')
    print(f'>>> INFO: Input size {x.shape}.')

    set_seed(seed)

    # generate and train the model
    g = spatial_graph.to(device) 
    x = x.to(device) 
    gene = gene.to(device)
    model = Denoising(layers=ae_layers).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # train the model
    model.train()
    for _ in tqdm(range(epochs), desc='>>> INFO: Training'):
        optim.zero_grad()
        rebuild_gene = model(g, x)
        loss = F.mse_loss(gene, model.activation(rebuild_gene))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optim.step()

    model.eval()
    with torch.no_grad():
        rebuild_gene = model(g, x)
    adata.layers['denoised_gene'] = rebuild_gene.cpu().detach().numpy()

    # save rebuilt gene 
    if (isinstance(save_denoised_gene, str)):
        np.save(save_denoised_gene, adata.obsm['denoised_gene'], allow_pickle=True, fix_imports=True)
        print('>>> INFO: Successfully save denoised gene at {}.'.format(save_denoised_gene)) if (show) else None

    print('>>> INFO: Finish gene denoising process, total time: {:.3f}s.'.format(time.time() - start_time)) if (show) else None

    return adata
