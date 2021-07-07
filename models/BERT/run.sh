#!/usr/bin/env bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#OUT_DIR=/results/SQuAD

echo "Container nvidia build = " $NVIDIA_BUILD_ID
nccl_ib_disable=0
BACKEND="gloo"
INIT="tcp://127.0.0.1:4444"

while (( "$#" )); do
    case "$1" in
        --backend)
	    BACKEND=$2
	    shift 2
	    ;;
	--init)
	    INIT=$2
	    shift 2
	    ;;
	--)
	    shift
	    break
	    ;;
    esac
done


epochs="1.0"
init_checkpoint="./dataset/checkpoint/bert_large_qa.pt"
learning_rate="3e-5"
precision="fp32"
num_gpu="1"
seed="1"
squad_dir="./dataset/squad/v1.1"
vocab_file="./dataset/vocab.txt"
OUT_DIR="."
mode="train"
CONFIG_FILE="./dataset/checkpoint/bert_config.json"
max_steps="200"
batch_size="4"


echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16=" --fp16 "
fi


CMD="python run_squad.py "
CMD+="--init_checkpoint=$init_checkpoint "
if [ "$mode" = "train" ] ; then
  CMD+="--do_train "
  CMD+="--train_file=$squad_dir/train-v1.1.json "
  CMD+="--train_batch_size=$batch_size "
elif [ "$mode" = "eval" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
elif [ "$mode" = "prediction" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
else
  CMD+=" --do_train "
  CMD+=" --train_file=$squad_dir/train-v1.1.json "
  CMD+=" --train_batch_size=$batch_size "
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
fi
CMD+=" --do_lower_case "
# CMD+=" --old "
# CMD+=" --loss_scale=128 "
BUCKET_SIZE_MB=100
if [ ! -z "$BUCKET_SIZE_MB" ] ; then
    CMD+="--bucket_size_mb=$BUCKET_SIZE_MB "
fi
CMD+=" --bert_model=bert-large-uncased "
CMD+=" --learning_rate=$learning_rate "
CMD+=" --seed=$seed "
CMD+=" --num_train_epochs=$epochs "
CMD+=" --max_seq_length=384 "
CMD+=" --doc_stride=128 "
CMD+=" --output_dir=$OUT_DIR "
CMD+=" --vocab_file=$vocab_file "
CMD+=" --config_file=$CONFIG_FILE "
CMD+=" --max_steps=$max_steps "
CMD+=" --dist-backend=$BACKEND "
CMD+=" --init=$INIT "
CMD+=" $use_fp16"

echo "$CMD"
time $CMD
