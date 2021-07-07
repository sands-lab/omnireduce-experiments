#!/bin/bash
NW=8
WORKER0_ADDR="11.0.0.201"
export GLOO_SOCKET_IFNAME=ens1f0
SYSTEM=omnireduce-DPDK
for model in DeepLight LSTM NCF BERT ResNet152 VGG19; do for nw in 8; do for rank in `seq 0 $((nw-1))`; do mkdir -p ./10G-results/${nw}-${rank}/${SYSTEM}/${model}; done; done; done

MODEL=DeepLight
echo "Running $MODEL"
nvidia-docker run --env OMPI_COMM_WORLD_SIZE=${NW} --env OMPI_COMM_WORLD_RANK=$RANK --env OMPI_COMM_WORLD_LOCAL_RANK=0 --env CUDA_VISIBLE_DEVICES=0 --env GLOO_SOCKET_IFNAME --privileged --network=host --rm -v `pwd`:/workspace -w /root/exps/models/${MODEL} --mount type=bind,source=/dev/hugepages,target=/dev/hugepages $(find /dev -name 'uio*' -type c -printf ' --device %p') --device /dev/vfio omnireduce-dpdk bash -c "cp /workspace/daiet-`hostname`.cfg . && python main_all.py -l2 6e-7 -n_epochs 2 -warm 2 -prune 1 -sparse 0.90 -prune_deep 1 -prune_fm 1 -prune_r 1 -use_fwlw 1 -emb_r 0.444 -emb_corr 1. -backend gloo -batch_size 2048 -init tcp://${WORKER0_ADDR}:4040" |& tee ./10G-results/${NW}-${RANK}/${SYSTEM}/${MODEL}/log.txt

MODEL=LSTM
echo "Running $MODEL"
nvidia-docker run --env OMPI_COMM_WORLD_SIZE=${NW} --env OMPI_COMM_WORLD_RANK=$RANK --env OMPI_COMM_WORLD_LOCAL_RANK=0 --env CUDA_VISIBLE_DEVICES=0 --env GLOO_SOCKET_IFNAME --privileged --network=host --rm -v `pwd`:/workspace -w /root/exps/models/${MODEL} --mount type=bind,source=/dev/hugepages,target=/dev/hugepages $(find /dev -name 'uio*' -type c -printf ' --device %p') --device /dev/vfio omnireduce-dpdk bash -c "cp /workspace/daiet-`hostname`.cfg . && ./run.sh --init tcp://${WORKER0_ADDR}:4040 --backend gloo" |& tee ./10G-results/${NW}-${RANK}/${SYSTEM}/${MODEL}/log.txt

MODEL=NCF
echo "Running $MODEL"
nvidia-docker run --env OMPI_COMM_WORLD_SIZE=${NW} --env OMPI_COMM_WORLD_RANK=$RANK --env OMPI_COMM_WORLD_LOCAL_RANK=0 --env CUDA_VISIBLE_DEVICES=0 --env GLOO_SOCKET_IFNAME --privileged --network=host --rm -v `pwd`:/workspace -w /root/exps/models/${MODEL} --mount type=bind,source=/dev/hugepages,target=/dev/hugepages $(find /dev -name 'uio*' -type c -printf ' --device %p') --device /dev/vfio omnireduce-dpdk bash -c "sed -i 's#tcp://10.200.0.31:44444#{args.init}#g' /root/exps/models/NCF/ncf.py && cp /workspace/daiet-`hostname`.cfg . && ./run_and_time.sh --init tcp://${WORKER0_ADDR}:4040 --backend gloo" |& tee ./10G-results/${NW}-${RANK}/${SYSTEM}/${MODEL}/log.txt

MODEL=BERT
echo "Running $MODEL"
nvidia-docker run --env OMPI_COMM_WORLD_SIZE=${NW} --env OMPI_COMM_WORLD_RANK=$RANK --env OMPI_COMM_WORLD_LOCAL_RANK=0 --env CUDA_VISIBLE_DEVICES=0 --env GLOO_SOCKET_IFNAME --privileged --network=host --rm -v `pwd`:/workspace -w /root/exps/models/${MODEL} --mount type=bind,source=/dev/hugepages,target=/dev/hugepages $(find /dev -name 'uio*' -type c -printf ' --device %p') --device /dev/vfio omnireduce-dpdk bash -c "cp /workspace/daiet-`hostname`.cfg . && ./run.sh --init tcp://${WORKER0_ADDR}:4040 --backend gloo" |& tee ./10G-results/${NW}-${RANK}/${SYSTEM}/${MODEL}/log.txt

MODEL=ResNet152
echo "Running $MODEL"
nvidia-docker run --env CUDA_VISIBLE_DEVICES=0 --env GLOO_SOCKET_IFNAME --privileged --network=host --rm -v `pwd`:/workspace -w /root/exps/models/CNN --mount type=bind,source=/dev/hugepages,target=/dev/hugepages $(find /dev -name 'uio*' -type c -printf ' --device %p') --device /dev/vfio omnireduce-dpdk bash -c "cp /workspace/daiet-`hostname`.cfg . && python main.py -a resnet152 --lr 0.1 --world-size ${NW} --rank ${RANK} --dist-url tcp://${WORKER0_ADDR}:4040 --dist-backend gloo ./dataset/" |& tee ./10G-results/${NW}-${RANK}/${SYSTEM}/${MODEL}/log.txt

MODEL=VGG19
echo "Running $MODEL"
nvidia-docker run --env CUDA_VISIBLE_DEVICES=0 --env GLOO_SOCKET_IFNAME --privileged --network=host --rm -v `pwd`:/workspace -w /root/exps/models/CNN --mount type=bind,source=/dev/hugepages,target=/dev/hugepages $(find /dev -name 'uio*' -type c -printf ' --device %p') --device /dev/vfio omnireduce-dpdk bash -c "cp /workspace/daiet-`hostname`.cfg . && python main.py -a vgg19 --lr 0.01 --world-size ${NW} --rank ${RANK} --dist-url tcp://${WORKER0_ADDR}:4040 --dist-backend gloo ./dataset/" |& tee ./10G-results/${NW}-${RANK}/${SYSTEM}/${MODEL}/log.txt
