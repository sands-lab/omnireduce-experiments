#!/bin/bash
set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

THRESHOLD=1.0
BACKEND=""
INIT=""

# Get command line seed
seed=1
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
	--seed)
	    seed=$2
	    shift 2
	    ;;
	--)
	    shift
	    break
	    ;;
    esac
done
            

# Get the multipliers for expanding the dataset
USER_MUL=${USER_MUL:-4}
ITEM_MUL=${ITEM_MUL:-16}


# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"
python ncf.py \
    -l 0.0002 \
    -b 1048576 \
    --layers 256 256 128 64 \
    -f 64 \
	--seed $seed \
    --threshold $THRESHOLD \
    --user_scaling ${USER_MUL} \
    --item_scaling ${ITEM_MUL} \
    --cpu_dataloader \
    --random_negatives \
    --backend ${BACKEND} \
    --init ${INIT}

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( $end - $start ))
result_name="recommendation"
echo "RESULT,$result_name,$seed,$result,$USER,$start_fmt"





