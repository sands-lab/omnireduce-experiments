# NCF
Our script relies on [this](https://github.com/mlcommons/training/tree/master/recommendation/pytorch). We use the [ML-20M](https://grouplens.org/datasets/movielens/20m/) expanded dataset, ML-20mx4x16. The folder dataset has a tiny dataset (several batches) for testing.

## Requirements
Aside from PyTorch with OmniReduce, ensure you have `numpy-indexed` and `mlperf_compliance`.

**Install Dependencies** :

    conda install -y -c conda-forge numpy-indexed
    pip install mlperf_compliance

## NCF Training
### 1. Create and edit omnireduce.cfg
- Read this [example](https://github.com/sands-lab/omnireduce/tree/master/omnireduce-RDMA/example) and create the your own `omnireduce.cfg` according to the cluster information.
- Copy the `omnireduce.cfg` to all the aggregators and workers. And `omnireduce.cfg` needs to be in the same directory as the PyTorch script (for workers) or program `aggregator` (for aggregators).
### 2. Run aggregators
Aggregator 0 and aggregator 1:

    cd ./omnireduce-RDMA/example
    ./aggregator

### 3. Run workers
Worker 0:

    CUDA_VISIBLE_DEVICES=0 GLOO_SOCKET_IFNAME=eth0 OMPI_COMM_WORLD_SIZE=2 OMPI_COMM_WORLD_RANK=0 OMPI_COMM_WORLD_LOCAL_RANK=0 ./run_and_time.sh --init tcp://IP_OF_NODE0:FREEPORT --backend gloo

Worker 1:

    CUDA_VISIBLE_DEVICES=0 GLOO_SOCKET_IFNAME=eth0 OMPI_COMM_WORLD_SIZE=2 OMPI_COMM_WORLD_RANK=1 OMPI_COMM_WORLD_LOCAL_RANK=0 ./run_and_time.sh --init tcp://IP_OF_NODE0:FREEPORT --backend gloo
