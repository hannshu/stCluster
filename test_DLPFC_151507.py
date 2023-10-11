from stCluster.train import train
from stCluster.run import evaluate_embedding
from st_datasets.dataset import get_data, get_dlpfc_data

# load dataset 
adata, n_cluster = get_data(dataset_func=get_dlpfc_data, id='151507')

# train stCluster
adata, g = train(adata, radius=150, ae_rate=0.8, adj_rate=0.2, pred_rate=0.3, init_paras='stCluster/paras/DLPFC_parameters.pkl', seed=0)

# clustering
adata, score = evaluate_embedding(adata=adata, n_cluster=n_cluster, cluster_method=['mclust'], cluster_score_method='ARI')
print(score)    # show ARI score
