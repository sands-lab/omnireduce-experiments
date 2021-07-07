#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODEL=lstm_luong_wmt_en_de
BATCH_SIZE=64
shared_path="$DIR"
PARAMS=""
BACKEND=""
INIT=""
rm -f $shared_path/python_init_process_group
while (( "$#" )); do
  case "$1" in
    -b|--batch-size)
      BATCH_SIZE=$2
      shift 2
      ;;
    --backend)
      BACKEND=$2
      shift 2
      ;;
    --init)
      INIT=$2
      shift 2
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done
# set positional arguments in their proper place
eval set -- "$PARAMS"

python lm/main.py --data ./Google-Billion-Words/PyTorch_GBW_LM --backend $BACKEND  --shared_path ${shared_path} --batch_size $BATCH_SIZE --lr 1e-3 --init $INIT $PARAMS 

