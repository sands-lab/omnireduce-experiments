import torch
import torch.distributed as dist
import argparse
from time import sleep
from random import randint
import time
import random
import numpy
import sys
import os
initvalue = 0.01

def gen_data(rank, tensorsize, blocksize, density):
    blocknum = int(tensorsize/blocksize)
    if rank==-1:
        nonzero_bnum=0
    else:
        nonzero_bnum = int(blocknum*density)
    random.seed(rank)
    nonzero_blocks = random.sample(range(blocknum), nonzero_bnum)
    data = [0.0]*tensorsize
    for bid in nonzero_blocks:
        idx = bid*blocksize
        while idx<(bid+1)*blocksize:
            data[idx] = initvalue
            idx += 1
    return data

def gen_data_nonoverlap(rank, worldsize, tensorsize, blocksize, density):
    blocknum = int(tensorsize/blocksize)
    if rank==-1:
        nonzero_bnum=0
    else:
        nonzero_bnum = int(blocknum*density)
    start_index = int(rank*(blocknum-nonzero_bnum)/(worldsize-1))
    nonzero_blocks = range(start_index, min(blocknum, start_index+nonzero_bnum))
    data = [0.0]*tensorsize
    for bid in nonzero_blocks:
        idx = bid*blocksize
        while idx<(bid+1)*blocksize:
            data[idx] = initvalue
            idx += 1
    return data

def check_density(tensor, blocksize, tensorsize):
    i=0
    nonzeronum = 0
    while i<tensorsize:
        if tensor[i]!=0.0:
            nonzeronum+=1
        i += blocksize
    return nonzeronum

def get_expected_result(worldsize, tensorsize, blocksize, density, allreduce_times):
    data = [0.0 for i in range(tensorsize)]
    for rank in range(worldsize):
        tmp = gen_data(rank, tensorsize, blocksize, density)
        for i in range(tensorsize):
            data[i] += tmp[i]
    return data

def benchmark(rank, world_size, tensorsize, blocksize, density, check):
    local_rank = 0
    torch.cuda.set_device(local_rank)
    mydevice = torch.device("cuda", local_rank)
    begin = time.time()
    data = gen_data(rank, tensorsize, blocksize, density) # random overlap
    #data = gen_data(0, tensorsize, blocksize, density) # all overlap
    #data = gen_data_nonoverlap(rank, world_size, tensorsize, blocksize, density) # non-overlap
    tensor_data = torch.FloatTensor(data).cuda(device=mydevice)
    tensor = tensor_data.clone()
    # group all ranks
    ranks = list(range(world_size))
    group = dist.new_group(ranks=ranks)
    allreduce_time = []
    extra_time = time.time()-begin
    localtime = numpy.zeros(1)
    globaltime = numpy.zeros(1)
    #Warm up
    for step in range(10):
        sys.stdout.flush()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        tensor = tensor_data.clone()
        torch.cuda.synchronize()
    print("Warm up over")
    sys.stdout.flush()
    allreduce_times = 0
    for step in range(100):
        localtime = numpy.zeros(1)
        globaltime = numpy.zeros(1)
        if step%1==0:
            allreduce_times = 0
            tensor = tensor_data.clone()
        torch.cuda.synchronize()
        allreduce_times += 1
        begin = time.time()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        torch.cuda.synchronize()
        localtime[0] = int((time.time()-begin)*1000000)
        globaltime[0]=localtime[0]
        if rank>=0:
            print("time:"+str(globaltime[0])+";")
            sys.stdout.flush()

    if rank==0 and check==1:
        print("final check:")
        print("gen expected result...")
        expected = get_expected_result(world_size, tensorsize, blocksize, density, allreduce_times)
        tensor = tensor.cpu().data.numpy()
        torch.cuda.synchronize() 
        result_value = initvalue*pow(2,allreduce_times)
        for i in range(tensorsize):
            expected_value = expected[i]*pow(world_size, allreduce_times-1)
            if i % 1000000==0:
                print("check: ", i, expected_value, tensor[i])
            if abs(tensor[i]-expected_value)>0.1 :
                print("allreduce error: ", expected_value, tensor[i])
                break

def initialize(backend, rank, world_size, ip, port, tensorsize, blocksize, density):
    if rank==0:
        print(density)
    torch.cuda.set_device(0)
    dist.init_process_group(backend=backend,init_method='tcp://{}:{}'.format(ip, port),rank=rank,world_size=world_size)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='10.200.0.32')
    parser.add_argument('--port', type=str, default='40000')
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--tensor-size', '-t', type=int, default=26214400)
    parser.add_argument('--block-size', '-b', type=int, default=256)
    parser.add_argument('--density', '-d', type=float, default=1.0)
    parser.add_argument('--rank', '-r', type=int)
    parser.add_argument('--size', '-s', type=int)
    parser.add_argument('--check', '-c', type=int, default=0)
    args = parser.parse_args()
    initialize(args.backend, args.rank, args.size, args.ip, args.port, args.tensor_size, args.block_size, args.density)
    benchmark(args.rank, args.size, args.tensor_size, args.block_size, args.density, args.check)

if __name__ == '__main__':
    main()
