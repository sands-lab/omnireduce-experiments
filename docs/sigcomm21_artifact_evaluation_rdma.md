# SIGCOMM 2021 Artifact Evaluation Getting Start Guide for OmniReduce (RDMA)
This document introduces how to reproduce the evaluation of OmniReduce of RDMA and GDR version in our SIGCOMM'21 paper. We use the docker image here to ensure that you don't encounter problems with the system environment.
## Overview
* Hardware
* Run experiments
* Validate results
* Produce paper's plots
## Hardware
Our experiments require 8 CPU servers to work as `aggregators` and 8 GPU servers to work as `workers`. Each GPU server has one GPU. To reproduce the evaluation in our SIGCOMM'21 paper,  GPUs need to support GPUDirect. The network bandwidth between each aggregator and each worker is 100Gbps.
## Run experiments
Our experiments include two parts. The first is the **micro-benchmark** experiment, which tests allreduce latency on 100MB tensors with different parameters (sparsity, worker number...). The second is the **end-to-end** experiment, which tests the training time of six deep learning models including DeepLight, LSTM, NCF, BERT, ResNet152 and VGG19. We have written several distributed scripts for running OmniReduce experiments with multiple workers and aggregators. You only need to update the configuration file `omnireduce.cfg` and run these scripts on the **worker-0** node, and then the programs of other nodes including workers and aggregators will be launched automatically.

For ease of introduction, assume that the network interface to use is `ens1f1` and IB interface to use is `mlx5_1:1` for workers and aggregators. The index of GPU to use is 1 and the IP addresses of 8 workers and 8 aggregators are as follows:
| Worker number | IP address | Aggregator number | IP address |
|--|--|--|--|
| 0 | 10.0.0.10 | 0 | 10.0.0.20 |
| 1 | 10.0.0.11 | 1 | 10.0.0.21 |
| 2 | 10.0.0.12 | 2 | 10.0.0.22 |
| 3 | 10.0.0.13 | 3 | 10.0.0.23 |
| 4 | 10.0.0.14 | 4 | 10.0.0.24 |
| 5 | 10.0.0.15 | 5 | 10.0.0.25 |
| 6 | 10.0.0.16 | 6 | 10.0.0.26 |
| 7 | 10.0.0.17 | 7 | 10.0.0.27 |

Below, we will introduce how to run experiments step by step.

Firstly, you need to pull our docker image:

```bash
docker pull phlix/omnireduce-rdma:latest
```

Next, we set up a shared ssh key pair so that the containers can communicate with one another through ssh:

```bash
# on any host machine
SHARED_FOLDER=~/.ssh/omnireduce_shared_ssh_key
mkdir -p $SHARED_FOLDER
ssh-keygen -q -t rsa -N '' -f $SHARED_FOLDER/id_rsa
cat $SHARED_FOLDER/id_rsa.pub > $SHARED_FOLDER/authorized_keys
chmod -R 700 $SHARED_FOLDER
sudo chown -R root:root $SHARED_FOLDER
```

Put `SHARED_FOLDER` in a file system accessible to all servers or manually copy it to the same path on all servers.

Finally, launch containers:

- For all the aggregators:
    
	```bash
	docker run -it --net=host --cap-add=IPC_LOCK --device=/dev/infiniband/uverbs1 -v $SHARED_FOLDER:/root/.ssh phlix/omnireduce-rdma:latest /bin/bash
	```
- For all the workers:

	```bash
	docker run -it --gpus all --net=host --cap-add=IPC_LOCK --device=/dev/infiniband/uverbs1 -v $SHARED_FOLDER:/root/.ssh phlix/omnireduce-rdma:latest /bin/bash
	```

After launching containers, we can run experiments as follows.

### 1. Micro-benchmark (~48 minutes)
* **Update configuration file**
Every time, before you run the benchmark scripts, you need to update the `omnireduce.cfg` file in `/usr/local/omnireduce/example` according to your cluster information. You only need to do this on the `worker-0` node as our script will copy this file to all the other machines.
Below is a `omnireduce.cfg` for 2 workers and 2 aggregators. The parameter `direct_memory = 1` means OmniReduce will  try to use GDR. To get the results without GDR, you need to set this to 0.
Note that the `worker_cores` and `aggregator_cores` refer to the core bound to the thread and value -1 means no CPU affinity setting. To get the best performance, you need to bind threads to cores.
	<details>
	<summary>omnireduce.cfg for 2 workers and 2 aggregators</summary>

	>[omnireduce] <br>
	>num_workers = 2 <br>
	>num_aggregators = 2 <br>
	>num_threads = 8 <br>
	>worker_cores = -1,-1,-1,-1,-1,-1,-1,-1 <br>
	>aggregator_cores = -1,-1,-1,-1,-1,-1,-1,-1 <br>
	>threshold = 0.0 <br>
	>buffer_size = 1024 <br>
	>chunk_size = 1048576 <br>
	>bitmap_chunk_size = 16777216 <br>
	>message_size = 256 <br>
	>block_size = 256 <br>
	>ib_hca = mlx5_1 <br>
	>gid_idx = 2 <br>
	>gpu_devId = 0 <br>
	>direct_memory = 1 <br>
	>adaptive_blocksize = 0 <br>
	>worker_ips = 10.0.0.10,10.0.0.11 <br>
	>aggregator_ips = 10.0.0.20,10.0.0.21 <br>
	</details>	

* **Run scripts** 
We provide three scripts in `/home/exps/benchmark` to reproduce our results: `nccl-benchmark.sh`, `omni-benchmark.sh` and `omni-bf-benchmark.sh`. `nccl-benchmark.sh` and `omni-benchmark.sh` are used for comparing OmniReduce with NCCL and `omni-bf-benchmark.sh` is used to evaluate our Block Fusion method.
All these scripts will read `omnireduce.cfg` to know which workers and aggregators will be used and whether GDR is used.
These scripts only need to be run on the `worker-0`, the commands are as follows:

	- For `nccl-benchmark.sh` (~1 minutes)

	      # now you are in docker environment
	      cd /home/exps/benchmark
	      CUDA_VISIBLE_DEVICES=1 NCCL_SOCKET_IFNAME=ens1f1 ./nccl-benchmark.sh
    
	    After running this script, the result files will be saved in `/home/exps/benchmark/100G-results/2/NCCL-GDR/`. `2` in the path string refers to the number of workers. `NCCL-GDR` refers to that GDR is used in this run and it will become `NCCL-RDMA` if you set `direct_memory` to 0 in `omnireduce.cfg`. The result files are saved in this format, `${density}.log`, where `density` means the proportion of non-zero data (e.g. `0.8.log`).

	- For `omni-benchmark.sh` (~3 minutes)

	      # now you are in docker environment
	      cd /home/exps/benchmark
	      CUDA_VISIBLE_DEVICES=1 GLOO_SOCKET_IFNAME=ens1f1 ./omni-benchmark.sh

		After running this script, the result files will be saved in `/home/exps/benchmark/100G-results/2/omnireduce-GDR/`. `2` in the path string refers to the number of workers. `omnireduce-GDR` refers to that GDR is used in this run and it will become `omnireduce-RDMA` if you set `direct_memory` to 0 in `omnireduce.cfg`.
		**NOTE: to reproduce our results, you need to run the above scripts with 2, 4 and 8 workers and aggregators for NCCL and omnireduce. And both GDR and RDMA (w/o GDR) need to be tested.**
		So the total time to run is about **24 minutes** ((1 minutes + 3 minutes)\*|{2,4,8}|\*|{GDR, RDMA}|).
		
    
	***Block Fusion evaluation**
	To evaluate the performance of Block Fusion method, we also run experiments with different `message_size` and `block_size` in `omnireduce.cfg`. Currently, Block Fusion method only works without GDR, so in the `omnireduce.cfg`, the `direct_memory` needs to be 0.
	Below table show the values of `block_size` and `message_size` that need to be tested.
		
	| Block Fusion (Block_size/Message_size) | Non Block Fusion (Block_size/Message_size) |
	|--|--|
	| 32/1024 | 32/32 |
	| 64/1024 | 64/64 |
	| 128/1024 | 128/128 |
 	| 256/1024 | 256/256 |
 	
	 We use `omni-bf-benchmark.sh` to run this experiment. And we still only need to run the following commands on `worker-0` node.
	 For `omni-bf-benchmark.sh` (~3 minutes)

	  # now you are in docker environment 
	  cd /home/exps/benchmark
	  CUDA_VISIBLE_DEVICES=1 GLOO_SOCKET_IFNAME=ens1f1 ./omni-bf-benchmark.sh

	After running this script, the result files will be saved in `/home/exps/benchmark/100G-results/2/omnireduce-RDMA/`.The result files of this experiment  are saved in this format, `${density}-${block_size}-${message_size}.log`, where `density` means the proportion of non-zero data and `block_size` and `message_size` are the values specified in `omnireduce.cfg` (e.g. `0.8-256-1024.log`). The total time to run this experiment is about **24 minutes** (3 minutes * 8).

### 2. End-to-end (~80 minutes)
In the end-to-end experiments, we use 8 workers and 8 aggregators to train 6 deep learning models including DeepLight, LSTM, NCF, BERT, ResNet152 and VGG19. We provide a tiny dataset for these model training in this docker image, so you do not need to download the dataset. Below, we introduce how to run these models one by one.
* **Update configuration file**
Similar to the benchmark, you need to update the `omnireduce.cfg` before running end-to-end experiments. The `omnireduce.cfg` files for all the models are almost the same. (**Note that the only difference is `buffer_size`, which needs to be set 2048 for BERT training and 1024 for other models.**)
Below is a `omnireduce.cfg` for end-to-end experiments.
	<details>
	<summary>omnireduce.cfg for 8 workers and 8 aggregators</summary>

	>[omnireduce] <br>
	>num_workers = 8 <br>
	>num_aggregators = 8 <br>
	>num_threads = 8 <br>
	>worker_cores = -1,-1,-1,-1,-1,-1,-1,-1 <br>
	>aggregator_cores = -1,-1,-1,-1,-1,-1,-1,-1 <br>
	>threshold = 0.0 <br>
	>buffer_size = 1024 <br>
	>chunk_size = 1048576 <br>
	>bitmap_chunk_size = 16777216 <br>
	>message_size = 256 <br>
	>block_size = 256 <br>
	>ib_hca = mlx5_1 <br>
	>gid_idx = 2 <br>
	>gpu_devId = 0 <br>
	>direct_memory = 1 <br>
	>adaptive_blocksize = 0 <br>
	>worker_ips 10.0.0.10,10.0.0.11,10.0.0.12,10.0.0.13,10.0.0.14,10.0.0.15,10.0.0.16,10.0.0.1<br>
	>aggregator_ips 10.0.0.20,10.0.0.21,10.0.0.22,10.0.0.23,10.0.0.24,10.0.0.25,10.0.0.26,10.0.0.2<br>
	</details>

* **Run end-to-end scripts** 
	- (1). DeepLight **(~5 minutes + ~3 minutes)**
		We provide two scripts including `nccl-deeplight.sh` and `omni-deeplight.sh` in `/home/exps/models/DeepLight` for comparing OmniReduce and NCCL. These scripts only need to be run on the `worker-0`, the commands are as follows:
		For `nccl-deeplight.sh`: (~5 minutes)

	      # now you are in docker environment
	      cd /home/exps/models/DeepLight
	      CUDA_VISIBLE_DEVICES=1 NCCL_SOCKET_IFNAME=ens1f1 ./nccl-deeplight.sh

		For `omni-deeplight.sh`: (~3 minutes)

	      # now you are in docker environment
	      cd /home/exps/models/DeepLight
	      CUDA_VISIBLE_DEVICES=1 GLOO_SOCKET_IFNAME=ens1f1 ./omni-deeplight.sh
	
		After running the above commands,  the output files are `/home/exps/models/DeepLight/100G-results/NCCL/log.txt` and `/home/exps/models/DeepLight/100G-results/omnireduce/log.txt`.
	- (2). LSTM **(~3 minutes + ~3 minutes)**
		We provide two scripts including `nccl-lstm.sh` and `omni-lstm.sh` in `/home/exps/models/LSTM` for comparing OmniReduce and NCCL. These scripts only need to be run on the `worker-0`, the commands are as follows:
		For `nccl-lstm.sh`: (~3 minutes)

	      # now you are in docker environment
	      cd /home/exps/models/LSTM
	      CUDA_VISIBLE_DEVICES=1 NCCL_SOCKET_IFNAME=ens1f1 ./nccl-lstm.sh

		For `omni-lstm.sh`: (~3 minutes)

	      # now you are in docker environment
	      cd /home/exps/models/LSTM
	      CUDA_VISIBLE_DEVICES=1 GLOO_SOCKET_IFNAME=ens1f1 ./omni-lstm.sh
	
		After running the above commands,  the output files are `/home/exps/models/LSTM/100G-results/NCCL/log.txt` and `/home/exps/models/LSTM/100G-results/omnireduce/log.txt`.
	- (3). NCF **(~3 minutes + ~3 minutes)**
		We provide two scripts including `nccl-ncf.sh` and `omni-ncf.sh` in `/home/exps/models/NCF` for comparing OmniReduce and NCCL. These scripts only need to be run on the `worker-0`, the commands are as follows:
		For `nccl-ncf.sh`: (~3 minutes)

	      # now you are in docker environment
	      cd /home/exps/models/NCF
	      CUDA_VISIBLE_DEVICES=1 NCCL_SOCKET_IFNAME=ens1f1 ./nccl-ncf.sh

		For `omni-ncf.sh`: (~3 minutes)

	      # now you are in docker environment
	      cd /home/exps/models/NCF
	      CUDA_VISIBLE_DEVICES=1 GLOO_SOCKET_IFNAME=ens1f1 ./omni-ncf.sh
	
		After running the above commands,  the output files are `/home/exps/models/NCF/100G-results/NCCL/log.txt` and `/home/exps/models/NCF/100G-results/omnireduce/log.txt`.

	- (4). BERT **(~20 minutes + ~20 minutes)**
		We provide two scripts including `nccl-bert.sh` and `omni-bert.sh` in `/home/exps/models/BERT` for comparing OmniReduce and NCCL. For the `omnireduce.cfg`, the `buffer_size` needs to be set 2048. These scripts only need to be run on the `worker-0`, the commands are as follows:
		For `nccl-bert.sh`: (~20 minutes)

	      # now you are in docker environment
	      cd /home/exps/models/BERT
	      CUDA_VISIBLE_DEVICES=1 NCCL_SOCKET_IFNAME=ens1f1 ./nccl-bert.sh

		For `omni-bert.sh`: (~20 minutes)

	      # now you are in docker environment
	      cd /home/exps/models/BERT
	      CUDA_VISIBLE_DEVICES=1 GLOO_SOCKET_IFNAME=ens1f1 ./omni-bert.sh
	
		After running the above commands,  the output files are `/home/exps/models/BERT/100G-results/NCCL/log.txt` and `/home/exps/models/BERT/100G-results/omnireduce/log.txt`.
	- (5). ResNet152 **(~5 minutes + ~5 minutes)**
		We provide two scripts including `nccl-resnet152.sh` and `omni-resnet152.sh` in `/home/exps/models/CNN` for comparing OmniReduce and NCCL. These scripts only need to be run on the `worker-0`, the commands are as follows:
		For `nccl-resnet152.sh`: (~5 minutes)

	      # now you are in docker environment
	      cd /home/exps/models/CNN
	      CUDA_VISIBLE_DEVICES=1 NCCL_SOCKET_IFNAME=ens1f1 ./nccl-resnet152.sh

		For `omni-resnet152.sh`: (~5 minutes)

	      # now you are in docker environment
	      cd /home/exps/models/CNN
	      CUDA_VISIBLE_DEVICES=1 GLOO_SOCKET_IFNAME=ens1f1 ./omni-resnet152.sh
	
		After running the above commands,  the output files are `/home/exps/models/CNN/100G-results/NCCL/ResNet152_log.txt` and `/home/exps/models/CNN/100G-results/omnireduce/ResNet152_log.txt`.
	- (6). VGG19 **(~5 minutes + ~5 minutes)**
		We provide two scripts including `nccl-vgg19.sh` and `omni-vgg19.sh` in `/home/exps/models/CNN` for comparing OmniReduce and NCCL. These scripts only need to be run on the `worker-0`, the commands are as follows:
		For `nccl-vgg19.sh`: (~5 minutes)

	      # now you are in docker environment
	      cd /home/exps/models/CNN
	      CUDA_VISIBLE_DEVICES=1 NCCL_SOCKET_IFNAME=ens1f1 ./nccl-vgg19.sh

		For `omni-vgg19.sh`: (~5 minutes)

	      # now you are in docker environment
	      cd /home/exps/models/CNN
	      CUDA_VISIBLE_DEVICES=1 GLOO_SOCKET_IFNAME=ens1f1 ./omni-vgg19.sh
	
		After running the above commands,  the output files are `/home/exps/models/CNN/100G-results/NCCL/VGG19_log.txt` and `/home/exps/models/CNN/100G-results/omnireduce/VGG19_log.txt`.
## Validate results

The output of the experiments will validate the following claims:

- Figure 4, Figure 5 and Figure 13: `/usr/local/omnireduce/example/100G-results/` reproduces Figure 4, Figure 5 and Figure 13 on Page 8, 9 and 12.
- Figure 10: `/home/exps/models/DeepLight/results`, `/home/exps/models/LSTM/results`, `/home/exps/models/NCF/results`, `/home/exps/models/BERT/results` and `/home/exps/models/CNN/results`  reproduce Figure 10 on Page 10. (Note that the results of DeepLight, NCF, ResNet152 and VGG19 will be better than the paper's as we just use a tiny dataset for these model training in this docker image. The entire dataset is too large to be placed in the image.)

## Produce paper's plots
To produce paper's plots, we provide `benckmark-rdma.ipynb` and `e2e-rdma.ipynb` in `/home/exps/notebook`. To start the notebook server, run the following commands on `worker-0`:

    # now you are in docker environment
    cd /home/exps/notebook
    jupyter notebook --allow-root --ip 10.0.0.10 --port 8888 

After running the notebook server, you can copy/paste the *URL* into your browser and produce paper's plots.
