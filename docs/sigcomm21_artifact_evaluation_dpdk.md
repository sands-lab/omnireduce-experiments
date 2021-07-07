# SIGCOMM 2021 Artifact Evaluation Getting Start Guide for OmniReduce (DPDK)
This document introduces how to reproduce the evaluation of OmniReduce DPDK version in our SIGCOMM'21 paper. We use Docker to minimize the impact of environment difference.
## Overview
* Hardware
* Run experiments
* Validate results
* Produce paper's plots
## Hardware
The DPDK evaluations in the paper are run with 8 physical CPU (dual 8-core Intel Xeon Silver 4108 at 1.80 GHz) servers working as `aggregators`, and 8 physical GPU servers (1 NVIDIA P100 GPU per server, dual 10-core CPUIntel Xeon E5-2630 v4 at 2.20 GHz) working as `workers`. They are interconnected with 10Gbps bandwidth links.
## Prerequisites
- root priveledge
- Docker (for workers, NVIDIA container toolkit is required to run end-to-end experiments, See [NVIDIA Container Toolkit User Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html))
- DPDK:
While we try to minimize dependency setups for compilation through docker, to run DPDK applications, you are required to configure your systems following the [DPDK User Guide](https://doc.dpdk.org/guides-19.11/linux_gsg/sys_reqs.html#Running-DPDK-Applications). Depending on your NIC hardware (in our environment, Intel NICs requires binding to the igb_uio driver while Mellanox NICs don't), you may need to change PCI driver bindings using the `dpdk-devbind.py` script (available inside the docker image). See [DPDK Network Interface Controller Drivers](https://doc.dpdk.org/guides/nics/) for more information. (Reference: https://github.com/usnistgov/ndn-dpdk/blob/main/docs/Docker.md#prepare-the-host-machine)
## Run experiments
Our experiments include two parts. The first is the **micro-benchmark** experiment, which tests allreduce latency on 100MB tensors with different parameters (sparsity, worker number). The second is the **end-to-end** experiment, which tests the training time of six deep learning models including DeepLight, LSTM, NCF, BERT, ResNet152 and VGG19.

For ease of introduction, we assume the network interface cards are named `ens1f0` and the IP addresses of the 8 workers and 8 aggregators are:
| Worker number | IP address | Aggregator number | IP address |
|--|--|--|--|
| 0 | 11.0.0.201 | 0 | 11.0.0.209 |
| 1 | 11.0.0.202 | 1 | 11.0.0.210 |
| 2 | 11.0.0.203 | 2 | 11.0.0.211 |
| 3 | 11.0.0.204 | 3 | 11.0.0.212 |
| 4 | 11.0.0.205 | 4 | 11.0.0.213 |
| 5 | 11.0.0.206 | 5 | 11.0.0.214 |
| 6 | 11.0.0.207 | 6 | 11.0.0.215 |
| 7 | 11.0.0.208 | 7 | 11.0.0.216 |

### Prepare Docker container environment
We prebuilt docker images which you can fetch by:
```bash
docker pull chenyuho/omnireduce-dpdk:latest
docker tag chenyuho/omnireduce-dpdk:latest omnireduce-dpdk:latest 
docker pull chenyuho/omnireduce-dpdk-aggregator:latest
docker tag chenyuho/omnireduce-dpdk-aggregator:latest omnireduce-dpdk-aggregator:latest
```
The building process of Omnireduce-DPDK is kernel version dependent. Although the above prebuilt image might just work fine, we recommend that you build the docker image yourself if you run a different kernel version from our servers (4.15.0). You can do this by:
```bash
mkdir -p /tmp/omnireduce
cd /tmp/omnireduce
wget https://raw.githubusercontent.com/ChenYuHo/omnireduce/docker/docker/Dockerfile
docker build -t omnireduce-dpdk -f ./Dockerfile .
```
The `omnireduce-dpdk` Docker image can serve as both workers and aggregators. If you prefer a slim image without CUDA libraries or you run a different linux kervel version for your aggregator machines, you can do:
```bash
mkdir -p /tmp/omnireduce
cd /tmp/omnireduce
wget https://raw.githubusercontent.com/ChenYuHo/omnireduce/docker/docker/aggregator_Dockerfile
docker build -t omnireduce-dpdk-aggregator -f ./aggregator_Dockerfile .
```
With Docker images ready, ensure that you properly setup hugepages and bind your NIC (if needed) according to the [Prerequisites](#Prerequisites). You can extract DPDK setup helper scripts from the docker image:
```
CTID=$(docker container create omnireduce-dpdk)
docker cp $CTID:/root/omnireduce/daiet/lib/dpdk/usertools - | sudo tar -x -C .
docker rm $CTID
# check usage of the scripts
./usertools/dpdk-devbind.py --help
```
### Edit config files
The worker processes will first look for `/etc/daiet.cfg`, then `daiet-$(hostname).cfg` in the working directory, and finally, `daiet.cfg` in the working directory.
Similarly, the aggregator processes look for `/etc/ps.cfg`, `ps-$(hostname).cfg`, and `ps.cfg`.
We put sample config files in `exps/dpdk-exp-utils`, you should edit the config files according to your system settings. We list some of the config entries that require your attention:
- `worker_ip`: This field only takes one IP, so if your working directory is accessible to all worker machines you either need to use the `daiet-$(hostname).cfg`format or change to another directory.
- `ps_ips`: This field takes all aggregator NIC IPs (separated by a comma and a space ", "). Note that, if your NIC requires binding to the DPDK driver, it does not have an IP anymore. Simply put unique but valid IP strings for each worker and aggregator and the system will work. Actually, in our implementation, IPs are not checked during packet processing.
- `ps_macs`: This field takes all aggregator NIC MACs (separated by a comma and a space ", "). You will need to record MACs before binding them to the DPDK driver (if applicable).
- `worker_port`and `ps_port`: a unused port
- `extra_eal_options`: we put `-w <[domain:]bus:devid.func>` which adds a PCI device in white list to ensure DPDK uses the corect device. You can get a list of NIC PCI addresses by
	```bash
	$ lspci -nn | grep Ether
	18:00.0 Ethernet controller [0200]: Intel Corporation 82599ES 10-Gigabit SFI/SFP+ Network Connection [8086:10fb] (rev 01)
	86:00.0 Ethernet controller [0200]: Intel Corporation I350 Gigabit Network Connection [8086:1521] (rev 01)
	
	# or, by DPDK devbind script
	$ python usertools/dpdk-devbind.py --status
	0000:18:00.0 '82599ES 10-Gigabit SFI/SFP+ Network Connection 10fb' if=ens1f0 drv=ixgbe unused=
	0000:86:00.0 'I350 Gigabit Network Connection 1521' if=enp134s0f0 drv=igb unused= *Active*
	
	``` 
	Refer to https://doc.dpdk.org/guides-19.11/linux_gsg/linux_eal_parameters.html for more options
- `cores`: we always use 4 cores for workers and aggregators in our experiments. To ensure best performance you need to bind to cores that are on the same socket with the NIC. You can check this by:
	```bash
	# assume you want to bind NIC 18:00.0
	$ cat /sys/bus/pci/devices/0000\:18\:00.0/numa_node
	1
	```
	Next, you can check NUMA information with
	```bash
	$ lscpu | grep NUMA
	NUMA node(s):        2
	NUMA node0 CPU(s):   0-7,16-23
	NUMA node1 CPU(s):   8-15,24-31
	```
	We should choose NUMA node 1 CPU cores, thus the `10-13` in the sample config file.

### Launching Aggregators
All experiments with omnireduce-DPDK share the same way of launching aggregators. You need to launch aggregators **before** launching workers. On each aggregator host mchine:
```bash
# replace omnireduce-dpdk with omnireduce-dpdk-aggregator if you use separate images
# change directory to where you put ps.cfg, it will be mounted to the docker container
docker run --privileged --network=host --rm -v `pwd`:/workspace -w /workspace --mount type=bind,source=/dev/hugepages,target=/dev/hugepages --mount type=bind,source=/sys,target=/sys $(find /dev -name 'uio*' -type c -printf ' --device %p') --device /dev/vfio omnireduce-dpdk-aggregator /root/omnireduce/daiet/ps/build/ps
```
To stop the aggregator, press "Ctrl+C". You only need to stop and restart the aggregator if you change `ps.cfg` (e.g., running experiments with a different number of workers), or after worker(s) crashed.

### Launching Workers
For all experiments below, run command on each worker host machine. You need to have proper `NW` (number of workers) and `RANK` (worker rank) environment variable set up, and modify `daiet.cfg` accordingly before launching workers. 

### 1. Micro-benchmark
We run the following commands on all worker machines:
```bash
# create logging directories
for system in NCCL-TCPIP omnireduce-DPDK; do for nw in 2 4 8; do for rank in `seq 0 $((nw-1))`; do mkdir -p ./10G-results/${nw}-${rank}/${system}; done; done; done
```
- NCCL-TCPIP
```bash
# modify NCCL_SOCKET_IFNAME, --ip to fit your environment
# outputs are be saved to ./10G-results/${NUM_WORKER}-${RANK}/NCCL-TCPIP
# e.g., on worker rank 0 when running 2 workers, 
NW=2
RANK=0
export NCCL_SOCKET_IFNAME=ens1f0
WORKER0_IP=11.0.0.201
nvidia-docker run --env CUDA_VISIBLE_DEVICES=0 --env NCCL_SOCKET_IFNAME --env NCCL_IB_DISABLE=1 --env NCCL_DEBUG=INFO --network=host --rm omnireduce-dpdk python /root/exps/benchmark/benchmark.py -d 1.0 --backend nccl -t 26214400 --ip ${WORKER0_IP} -s $NW -r $RANK | tee ./10G-results/${NW}-${RANK}/NCCL-TCPIP/1.0.log
```

- omnireduce-DPDK
**Note**: launch aggregators as described above first
```bash
# modify GLOO_SOCKET_IFNAME, --ip to fit your environment
# $(pwd) should contain daiet.cfg or daiet-`hostname`.cfg
# e.g., on worker rank 0 when running 2 workers, 
NW=2
RANK=0
export GLOO_SOCKET_IFNAME=ens1f0
WORKER0_IP=11.0.0.201
for density in 1.0 0.4 0.1 0.01; do nvidia-docker run --env CUDA_VISIBLE_DEVICES=0 --env GLOO_SOCKET_IFNAME  --privileged --network=host --rm -v `pwd`:/workspace -w /workspace --mount type=bind,source=/dev/hugepages,target=/dev/hugepages $(find /dev -name 'uio*' -type c -printf ' --device %p') --device /dev/vfio omnireduce-dpdk python /root/exps/benchmark/benchmark.py -d ${density} --backend gloo -t 26214400 --ip ${WORKER0_IP} -s $NW -r $RANK | tee ./10G-results/${NW}-${RANK}/omnireduce-DPDK/${density}.log; done
```

### 2. End-to-end
In the end-to-end experiments, we use 8 workers and 8 aggregators to train 6 deep learning models including DeepLight, LSTM, NCF, BERT, ResNet152 and VGG19. We provide a tiny dataset for these model training in this docker image, so you do not need to download the dataset. We provide `omnireduce-dpdk-e2e.sh` and `nccl-tcpip-e2e.sh`for ease of execution.
You need to:
1. Launch aggregators as described above if you are using omnireduce-DPDK
2. Ensure every worker have exported `RANK` environment variable, and then run the two scripts on every worker machine., e.g.,
```bash
# Note: This may take more than an hour depending on your hardware. Please refer to the RDMA guide for more detailed timing estimates.
# on worker 0
export RANK=0
./omnireduce-dpdk-e2e.sh
./nccl-tcpip-e2e.sh

# on worker 1
export RANK=1
./omnireduce-dpdk-e2e.sh
./nccl-tcpip-e2e.sh

# vice versa for wokers 2-7
# You need to change NW in the scripts if you want to use a different number of workers
```
## Validate results

The output of the experiments will validate the following figures:
- Micro-benchmark experiments: Figure 4
- End-to-end experiments: Figure 10

## Produce paper's plots
To produce paper's plots, we provide `benckmark-dpdk.ipynb` and `e2e-dpdk.ipynb` in `exps/notebook`. If you don't have jupyter environment, you can run the following commands inside worker 0 omnireduce-dpdk container:
```bash
    cd exps # where "10G-results" should reside
    jupyter notebook --ip 11.0.0.201 --port 8888 --allow-root
```
After running the notebook server, you can copy/paste the *URL* into your browser and produce paper's plots.
