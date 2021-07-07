# OmniReduce Experiments
This repo contains several OmniReduce experiments, including micro-benchmark, CV/NLP/Recommandation models implemented in PyTorch. You can use them to reproduce the evaluation results in our SIGCOMM'21 paper.

## Usage
Before running experiments, you should install PyTorch with OmniReduce according to [this](https://github.com/sands-lab/omnireduce/tree/master/omnireduce-RDMA/frameworks_integration/pytorch_patch). All the scripts including benchmark and model training are based on the PyTorch distributed package, so you can get the NCCL results for performance comparison by changing the `--backend` flag to `nccl` for performance comparison.

## Artifact Evaluation
SIGCOMM 2021 Artifact Evaluation Getting Start Guide for OmniReduce is in the folder `docs`.
