Installation
=====

.. _installation:

Important software Dependencies
::::::

- scanpy==1.9.3
- squidpy==1.3.0
- pytorch==1.13.1 (cuda==11.7)
- DGL==1.1.1 (cuda==11.7)
- R==4.2.0
- mclust==5.4.10

Note: The evaluation device for all the results in the article is an RTX3090.

Setup by Docker (Recommended)
~~~~~~~~~~~~

1. Download the stCluster image from `DockerHub <https://hub.docker.com/repository/docker/hannshu/stcluster>`_ and set up a container:

   .. code-block:: bash

      docker run --gpus all --name your_container_name -idt hannshu/stcluster:latest

2. Access the container:

   .. code-block:: bash

      docker start your_container_name
      docker exec -it your_container_name /bin/bash

3. Write a Python script to run stCluster. The anaconda environment for stCluster will be automatically activated in the container. The stCluster source code is located at ``/root/stCluster``. Make sure to run ``git pull`` to update the code before using it.

Note: Ensure that ``nvidia-docker2`` is properly installed on your host device. If not, follow the instructions to `set up nvidia-docker2 <https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)>`_ first.

Setup by Anaconda
~~~~~~~~~~~~

1. Install `Anaconda <https://docs.anaconda.com/free/anaconda/install>`_.

2. Clone the stCluster repository from GitHub:

   .. code-block:: bash

      git clone https://github.com/hannshu/stCluster.git

3. Download the dataset repository:

   .. code-block:: bash

      git submodule init
      git submodule update

4. Import the conda environment:

   .. code-block:: bash

      conda env create -f environment.yml

5. Write a Python script to run stCluster.

