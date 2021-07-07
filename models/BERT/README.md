# BERT
We fine-tune a pretrained BERT model on [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) dataset. Our script is based on [this nvidia repo](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT). The dataset is already in the dataset folder. The checkpoint needs to be placed in `./dataset/checkpoint`, which can be downloaded from [NGC](https://ngc.nvidia.com/catalog/models/nvidia:bert_pyt_ckpt_large_qa_squad11_amp/files).

## Requirements
Aside from PyTorch with OmniReduce, ensure you have `tqdm`, `dllogger` and `apex`.

**Install Dependencies** :

    pip install tqdm
    pip install nvidia-pyindex
    pip install nvidia-dllogger
    git clone https://github.com/NVIDIA/apex
    cd apex
    git reset --hard a651e2c2
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

**Dowload model checkpoint** :

    cd ./dataset/checkpoint
    wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/bert_pyt_ckpt_large_qa_squad11_amp/versions/19.09.0/zip -O bert_pyt_ckpt_large_qa_squad11_amp_19.09.0.zip
    unzip bert_pyt_ckpt_large_qa_squad11_amp_19.09.0.zip
    cd ../../ && mkdir -p results

## BERT Training
### 1. Create and edit omnireduce.cfg
- Read this [example](https://github.com/sands-lab/omnireduce/tree/master/omnireduce-RDMA/example) and create the your own `omnireduce.cfg` according to the cluster information.(Note: `buffer_size` needs to be set 2048 for BERT.)
- Copy the `omnireduce.cfg` to all the aggregators and workers. And `omnireduce.cfg` needs to be in the same directory as the PyTorch script (for workers) or program `aggregator` (for aggregators).
### 2. Run aggregators
Aggregator 0 and aggregator 1:

    cd ./omnireduce-RDMA/example
    ./aggregator

### 3. Run workers
Worker 0:

    CUDA_VISIBLE_DEVICES=0 GLOO_SOCKET_IFNAME=eth0 OMPI_COMM_WORLD_SIZE=2 OMPI_COMM_WORLD_RANK=0 OMPI_COMM_WORLD_LOCAL_RANK=0 ./run.sh --init tcp://IP_OF_NODE0:FREEPORT --backend gloo

Worker 1:

    CUDA_VISIBLE_DEVICES=0 GLOO_SOCKET_IFNAME=eth0 OMPI_COMM_WORLD_SIZE=2 OMPI_COMM_WORLD_RANK=1 OMPI_COMM_WORLD_LOCAL_RANK=0 ./run.sh --init tcp://IP_OF_NODE0:FREEPORT --backend gloo
