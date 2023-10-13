# stCluster
[![Documentation Status](https://readthedocs.org/projects/stcluster/badge/?version=latest)](https://stcluster.readthedocs.io/en/latest/?badge=latest)

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

To fully reproduce the results as described in the paper, it is recommended to use the container we have provided on a Nvidia RTX 3090 GPU device.

## Setup stCluster
### Setup by Docker (*Recommended*):  
1. Download the stcluster image from [DockerHub](https://hub.docker.com/repository/docker/hannshu/stcluster) and setup a container:
``` bash
docker run --gpus all --name your_container_name -idt hannshu/stcluster:latest
```

2. Access the container:
``` bash
docker start your_container_name
docker exec -it your_container_name /bin/bash
```

3. Write a python script to run stCluster

The anaconda environment for stCluster will be automatically activate in the container. The stCluster source code is located at `\root\stCluster`, please run ```git pull``` to update the codes before you use. 

- Note: Please make sure `nvidia-docker2` is properly installed on your host device. (Or follow this instruction to [setup nvidia-docker2](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)) first)

### Setup by anaconda:  
[install anaconda](https://docs.anaconda.com/free/anaconda/install/)

1. Clone this repository from Github:
``` bash
git clone https://github.com/hannshu/stCluster.git
```

2. Download dataset repository:

``` bash
git submodule init
git submodule update
```

3. Import conda environment:  
``` bash
conda env create -f environment.yml
```

4. Write a python script to run stCluster

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

## Tutorial
Read the [Documentation](https://stcluster.readthedocs.io/en/latest/) for detailed tutorials.

<!-- ## Citation
If you have found stCluster useful in your work, please consider citing [our article](url):
```

``` -->