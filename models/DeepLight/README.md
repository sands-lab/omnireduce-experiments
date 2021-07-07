# DeepLight
DeepLight is a sparse DeepFwFM which is a click-through rate (CTR) prediction model. We modify [this repo](https://github.com/WayneDW/DeepLight_Deep-Lightweight-Feature-Interactions) script to support distributed data parallelism by using PyTorch DistributedDataParallel (DDP) package. The training dataset we use is [Criteo’s 1TB Click Prediction Dataset](https://docs.microsoft.com/en-us/archive/blogs/machinelearning/now-available-on-azure-ml-criteos-1tb-click-prediction-dataset). The folder dataset has a tiny dataset ((several batches)) for testing.

## Requirements
Aside from PyTorch with OmniReduce, ensure you have `sklearn`.

**Install Dependencies** :

    pip install -U scikit-learn

## DeepLight Training
### 1. Create and edit omnireduce.cfg
- Read this [example](https://github.com/sands-lab/omnireduce/tree/master/omnireduce-RDMA/example) and create the your own `omnireduce.cfg` according to the cluster information.
- Copy the `omnireduce.cfg` to all the aggregators and workers. And `omnireduce.cfg` needs to be in the same directory as the PyTorch script (for workers) or program `aggregator` (for aggregators).
### 2. Run aggregators
Aggregator 0 and aggregator 1:

    cd ./omnireduce-RDMA/example
    ./aggregator

### 3. Run workers
Worker 0:

    CUDA_VISIBLE_DEVICES=0 GLOO_SOCKET_IFNAME=eth0 OMPI_COMM_WORLD_SIZE=2 OMPI_COMM_WORLD_RANK=0 OMPI_COMM_WORLD_LOCAL_RANK=0 python main_all.py -l2 6e-7 -n_epochs 2 -warm 2 -prune 1 -sparse 0.90  -prune_deep 1 -prune_fm 1 -prune_r 1 -use_fwlw 1 -emb_r 0.444 -emb_corr 1. -backend gloo -batch_size 2048  -init tcp://IP_OF_NODE0:FREEPORT

Worker 1:

    CUDA_VISIBLE_DEVICES=0 GLOO_SOCKET_IFNAME=eth0 OMPI_COMM_WORLD_SIZE=2 OMPI_COMM_WORLD_RANK=1 OMPI_COMM_WORLD_LOCAL_RANK=0 python main_all.py -l2 6e-7 -n_epochs 2 -warm 2 -prune 1 -sparse 0.90  -prune_deep 1 -prune_fm 1 -prune_r 1 -use_fwlw 1 -emb_r 0.444 -emb_corr 1. -backend gloo -batch_size 2048  -init tcp://IP_OF_NODE0:FREEPORT
