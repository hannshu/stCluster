# stCluster

![stCluster overview](./framework.png) 

## Overview 
Spatial transcriptomics offers unprecedented insights into gene expression within the native tissue context, effectively bridging molecular data with spatial information to unveil intricate cellular interactions and tissue organizations. In this regard, deciphering cellular spatial domains is a crucial task that requires the effective integration of gene expression data and spatial information. We introduce stCluster, a novel method that integrates graph contrastive learning with multi-task learning to refine informative representations for spatial transcriptomic data, consequently improving spatial domain identification. stCluster first leverages graph contrastive learning to learn discriminative representations capable of recognizing spatially coherent patterns. Through jointly optimizing multiple tasks, stCluster further fine-tunes the representations to be able to capture complex relationships between gene expression and spatial organization. Experimental results reveal its proficiency in accurately identifying complex spatial domains across various datasets and platforms, spanning tissue, organ, and embryo levels, outperforming existing state-of-the-art methods. Moreover, stCluster can effectively denoise the spatial gene expression patterns and enhance the spatial trajectory inference. 


## Software dependencies
scanpy==1.9.3  
squidpy==1.3.0  
pytorch==1.13.1(cuda==11.7)   
DGL==1.1.1(cuda==11.7)  
R==4.2.0  
mclust==5.4.10


## Fast deploy
<!-- ### Deployed by DockerHub (*Recommended*):  
 -->

### Deployed by anaconda:  
[install anaconda](https://docs.anaconda.com/free/anaconda/install/)

1. Import conda environment:  
``` bash
conda env create -f environment.yml
```

2. Write a python script to run stCluster

## Example
``` python
from stCluster.train import train
from st_datasets.dataset import get_data, dataset_you_need

# load dataset 
adata, n_cluster = get_data(dataset_func=dataset_you_need, dataset_args)

# train stCluster
adata, g = train(adata, train_args)

# downstream analysis
# clustering
from stCluster.run import evaluate_embedding

adata, score = evaluate_embedding(adata=adata, n_cluster=n_cluster, cluster_method=['mclust'], cluster_score_method=['ARI'])
print(score)    # show ARI score
# ...

# denoising
from stCluster.denoising import train as denoising

adata = denoising(adata, spatial_graph=g, denoising_args)
# evaluate denoised gene expression
# ...

# other downstream tasks
# ...
```


<!-- ## Citation
If you have found stCluster useful in your work, please consider citing [our article](url):
```

``` -->