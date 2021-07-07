#read omnireduce.cfg
i=1
wnum=0
anum=0
mode="RDMA"
while read line; do

if [[ $line =~ "num_workers"  ]]
then
    wnum=$((${line: 14}))
    echo "Worker number: $wnum"
fi
if [[ $line =~ "num_aggregators"  ]]
then
    anum=$((${line: 18}))
    echo "Aggregator number: $anum"
fi
if [[ $line =~ "direct_memory"  ]]
then
    mode_num=$((${line: 16}))
    if [[ "$mode_num" -eq "1" ]]
    then
        mode="GDR"
    fi
    echo "mode: $mode"
fi
if [[ $line =~ "worker_ips"  ]]
then
    #echo "$line"
    line=${line: 13}
    worker_arr=(${line//,/ })
    j=0
    while [ $j -lt $wnum ]
    do
        echo "worker $j IP : ${worker_arr[$j]}"
	j=$((j+1))
    done
fi
if [[ $line =~ "aggregator_ips"  ]]
then
    #echo "$line"
    line=${line: 17}
    aggregator_arr=(${line//,/ })
    j=0
    while [ $j -lt $anum ]
    do
        echo "aggregator $j IP : ${aggregator_arr[$j]}"
	j=$((j+1))
    done
fi
i=$((i+1))
done < omnireduce.cfg

#copy omnireduce.cfg
i=0
while [ $i -lt $wnum ]
do
    scp -P 2222 ./omnireduce.cfg ${worker_arr[$i]}:/usr/local/omnireduce/example/
    i=$((i+1))
done
i=0
while [ $i -lt $anum ]
do
    scp -P 2222 ./omnireduce.cfg ${aggregator_arr[$i]}:/usr/local/omnireduce/example/
    i=$((i+1))
done
densities="1.0 0.8 0.4 0.2 0.1 0.08 0.04 0.02 0.01"
for density in $densities; do
    echo "Tensor density : $density"
    # start aggregators
    i=0
    while [ $i -lt $anum ]
    do
        ssh -p 2222 ${aggregator_arr[$i]} "pkill -9 aggregator"
        ssh -p 2222 ${aggregator_arr[$i]} "cd /usr/local/omnireduce/example; nohup ./aggregator > aggregator.log 2>&1 &"
        i=$((i+1))
    done
    # start workers
    i=0
    while [ $i -lt $wnum ]
    do
        ssh -p 2222 ${worker_arr[$i]} "cd /home/exps/benchmark; mkdir -p ./100G-results/${wnum}/omnireduce-${mode}/ ; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}; nohup /usr/local/conda/bin/python benchmark.py -d ${density} --backend gloo -t 26214400 -r $i -s ${wnum} --ip ${worker_arr[0]} > ./100G-results/${wnum}/omnireduce-${mode}/${density}.log 2>&1 &"
        i=$((i+1))
    done
    # check completed
    while [[ 1 ]]
    do
        count=`ps -ef |grep python |grep -v "grep" |wc -l`
	if [ 0 == $count ];then
	    break
	fi
    done
    i=0
    while [ $i -lt $anum ]
    do
        ssh -p 2222 ${aggregator_arr[$i]} "pkill -9 aggregator"
        i=$((i+1))
    done
done
