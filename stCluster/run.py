import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from rpy2.rinterface_lib.embedded import RRuntimeError

from .train import train 
from .eval import eval
from .utils import cal_cluster_score


def train_and_evaluate(
        adata,
        n_cluster,
        cluster_method=['mclust'],
        cluster_score_method='ARI',
        radius=None,
        knears=None,
        spatial_img_path=None,
        k_neighbors=15,
        drop_rate=.96, 
        build_preclust=True,
        preclust_knears=35,
        preclust_method='louvain',
        preclust_para=1.,
        precluster_img_save_path=None,
        final_graph_img_save_path=None,
        ae_layers=[3000, 512, 30],
        tmp=.5,
        cutting_prob_1=.05,
        cutting_prob_2=.1,
        threshold=.7,
        multi_step=50,
        dec_alpha=1,
        ae_rate=.3,
        adj_rate=.3,
        pred_rate=.3,
        epochs=1000,
        lr=1e-3,
        early_stop_rate=0.,
        dec_precluster='louvain',
        dec_precluster_para=1.,
        save_embedding=None,
        save_model=None,
        show=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        embed_name='embedding',
        seed=3407,
):
    # train
    adata, _ = train(
        data=adata,
        radius=radius,
        knears=knears,
        spatial_img_path=spatial_img_path,
        k_neighbors=k_neighbors,
        drop_rate=drop_rate,
        build_preclust=build_preclust,
        preclust_knears=preclust_knears,
        preclust_method=preclust_method,
        preclust_para=preclust_para,
        precluster_img_save_path=precluster_img_save_path,
        final_graph_img_save_path=final_graph_img_save_path,
        ae_layers=ae_layers,
        tmp=tmp,
        cutting_prob_1=cutting_prob_1,
        cutting_prob_2=cutting_prob_2,
        threshold=threshold,
        multi_step=multi_step,
        dec_alpha=dec_alpha,
        ae_rate=ae_rate,
        adj_rate=adj_rate,
        pred_rate=pred_rate,
        epochs=epochs,
        lr=lr,
        early_stop_rate=early_stop_rate,
        dec_precluster=dec_precluster,
        dec_precluster_para=dec_precluster_para,
        save_embedding=save_embedding,
        save_model=save_model,
        show=show,
        device=device,
        embed_name=embed_name,
        seed=seed,
    )

    return evaluate_embedding(
        adata=adata,
        n_cluster=n_cluster,
        cluster_method=cluster_method,
        cluster_score_method=cluster_score_method
    )


def load_and_evaluate(
        adata,
        n_cluster,
        model_paras_path,
        cluster_method=['mclust'],
        cluster_score_method='ARI',
        radius=None,
        knears=None,
        ae_layers=[3000, 512, 30],
        save_embedding=None,
        save_rebuild_gene=None,
        show=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        embed_name='embedding'
):
    adata, _ = eval(
        data=adata,
        model_paras_path=model_paras_path,
        radius=radius,
        knears=knears,
        ae_layers=ae_layers,
        save_embedding=save_embedding,
        save_rebuild_gene=save_rebuild_gene,
        show=show,
        device=device,
        embed_name=embed_name,
    )

    return evaluate_embedding(
        adata=adata,
        n_cluster=n_cluster,
        cluster_method=cluster_method,
        cluster_score_method=cluster_score_method
    )


# mclust-EEE clustering method
# source: https://github.com/QIFEIDKN/STAGATE/blob/main/STAGATE/utils.py
def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STAGATE', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def evaluate_embedding(adata, n_cluster, cluster_method=['mclust'], cluster_score_method='ARI'):
    # evaluate cluster score
    obs_df = adata.obs.dropna()
    true_label = LabelEncoder().fit_transform(obs_df['cluster'])
    score = {}

    try:
        if ('mclust' in cluster_method):
            adata = mclust_R(adata, used_obsm='embedding', num_cluster=n_cluster)
            obs_df = adata.obs.dropna()
            pred_label = LabelEncoder().fit_transform(obs_df['mclust'])
            score['mclust'] = cal_cluster_score(true_label, pred_label, cluster_score_method)
    except (TypeError or RRuntimeError):
        print('>>> WARNING: mclust report TypeError')
        score['mclust'] = -1

    if ('kmeans' in cluster_method):
        from sklearn.cluster import KMeans
        adata.obs['kmeans'] = KMeans(n_clusters=n_cluster, random_state=0).fit_predict(adata.obsm['embedding'])
        obs_df = adata.obs.dropna()
        pred_label = LabelEncoder().fit_transform(obs_df['kmeans'])
        score['kmeans'] = cal_cluster_score(true_label, pred_label, cluster_score_method)

    if ('birch' in cluster_method):
        from sklearn.cluster import Birch
        adata.obs['birch'] = Birch(n_clusters=n_cluster).fit_predict(adata.obsm['embedding'])
        obs_df = adata.obs.dropna()
        pred_label = LabelEncoder().fit_transform(obs_df['birch'])
        score['birch'] = cal_cluster_score(true_label, pred_label, cluster_score_method)

    if ('gmm' in cluster_method):
        from sklearn.mixture import GaussianMixture
        adata.obs['gmm'] = GaussianMixture(n_components=adata.obsm['embedding'].shape[1], random_state=0).fit_predict(adata.obsm['embedding'])
        obs_df = adata.obs.dropna()
        pred_label = LabelEncoder().fit_transform(obs_df['gmm'])
        score['gmm'] = cal_cluster_score(true_label, pred_label, cluster_score_method)

    if ('ahc' in cluster_method):
        from sklearn.cluster import AgglomerativeClustering   
        adata.obs['ahc'] = AgglomerativeClustering(n_clusters=n_cluster).fit_predict(adata.obsm['embedding'])
        obs_df = adata.obs.dropna()
        pred_label = LabelEncoder().fit_transform(obs_df['ahc'])
        score['ahc'] = cal_cluster_score(true_label, pred_label, cluster_score_method)

    return adata, score
