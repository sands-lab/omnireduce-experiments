import torch.jit
import os
import math
import time
import timeit
from datetime import datetime
from collections import OrderedDict
from argparse import ArgumentParser
from alias_generator import AliasSample
import pickle
from convert import generate_negatives
from convert import generate_negatives_flat
from convert import CACHE_FN

#import tqdm
import numpy as np
import torch
import torch.nn as nn

import utils
from neumf import NeuMF
from torch.nn.parallel import DistributedDataParallel as DDP
from mlperf_compliance import mlperf_log

def init_distributed(args):
    args.world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
    args.distributed = args.world_size >= 1

    if args.distributed:
        print('distributed')
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

        '''
        Set cuda device so everything is done on the right GPU.
        THIS MUST BE DONE AS SOON AS POSSIBLE.
        '''
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend=args.backend, init_method=args.init, world_size=args.world_size, rank=args.rank)
    else:
        args.rank=0
        args.local_rank = 0

def parse_args():
    parser = ArgumentParser(description="Train a Nerual Collaborative"
                                        " Filtering model")
    parser.add_argument('-e', '--epochs', type=int, default=2,
                        help='number of epochs for training')
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help='number of examples for each iteration')
    parser.add_argument('--valid-batch-size', type=int, default=2**20,
                        help='number of examples in each validation chunk')
    parser.add_argument('-f', '--factors', type=int, default=8,
                        help='number of predictive factors')
    parser.add_argument('--layers', nargs='+', type=int,
                        default=[64, 32, 16, 8],
                        help='size of hidden layers for MLP')
    parser.add_argument('-n', '--negative-samples', type=int, default=4,
                        help='number of negative examples per interaction')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.001,
                        help='learning rate for optimizer')
    parser.add_argument('-k', '--topk', type=int, default=10,
                        help='rank for test examples to be considered a hit')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int,
                        help='manually set random seed for torch')
    parser.add_argument('--threshold', '-t', type=float,
                        help='stop training early at threshold')
    parser.add_argument('--valid-negative', type=int, default=999,
                        help='Number of negative samples for each positive test example')
    parser.add_argument('--processes', '-p', type=int, default=1,
                        help='Number of processes for evaluating model')
    parser.add_argument('--workers', '-w', type=int, default=8,
                        help='Number of workers for training DataLoader')
    parser.add_argument('--beta1', '-b1', type=float, default=0.9,
                        help='beta1 for Adam')
    parser.add_argument('--beta2', '-b2', type=float, default=0.999,
                        help='beta1 for Adam')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='eps for Adam')
    parser.add_argument('--user_scaling', default=1, type=int)
    parser.add_argument('--item_scaling', default=1, type=int)
    parser.add_argument('--cpu_dataloader', action='store_true',
                        help='pre-process data on cpu to save memory')
    parser.add_argument('--random_negatives', action='store_true',
                        help='do not check train negatives for existence in dataset')
    parser.add_argument('--backend', default='nccl', type=str,
                        help='backend for distributed processing')
    parser.add_argument('--init', default='tcp://127.0.0.1:4000', type=str,
                        help='init method for distributed')
    return parser.parse_args()


# TODO: val_epoch is not currently supported on cpu
def val_epoch(model, x, y, dup_mask, real_indices, K, samples_per_user, num_user, output=None,
              epoch=None, loss=None):

    start = datetime.now()
    log_2 = math.log(2)

    model.eval()
    hits = torch.tensor(0., device='cuda')
    ndcg = torch.tensor(0., device='cuda')

    with torch.no_grad():
        for i, (u,n) in enumerate(zip(x,y)):
            res = model(u.cuda().view(-1), n.cuda().view(-1), sigmoid=True).detach().view(-1,samples_per_user)
            # set duplicate results for the same item to -1 before topk
            res[dup_mask[i]] = -1
            out = torch.topk(res,K)[1]
            # topk in pytorch is stable(if not sort)
            # key(item):value(predicetion) pairs are ordered as original key(item) order
            # so we need the first position of real item(stored in real_indices) to check if it is in topk
            ifzero = (out == real_indices[i].cuda().view(-1,1))
            hits += ifzero.sum()
            ndcg += (log_2 / (torch.nonzero(ifzero)[:,1].view(-1).to(torch.float)+2).log_()).sum()

    mlperf_log.ncf_print(key=mlperf_log.EVAL_SIZE, value={"epoch": epoch, "value": num_user * samples_per_user})
    mlperf_log.ncf_print(key=mlperf_log.EVAL_HP_NUM_USERS, value=num_user)
    mlperf_log.ncf_print(key=mlperf_log.EVAL_HP_NUM_NEG, value=samples_per_user - 1)

    end = datetime.now()

    hits = hits.item()
    ndcg = ndcg.item()

    if output is not None:
        result = OrderedDict()
        result['timestamp'] = datetime.now()
        result['duration'] = end - start
        result['epoch'] = epoch
        result['K'] = K
        result['hit_rate'] = hits/num_user
        result['NDCG'] = ndcg/num_user
        result['loss'] = loss
        utils.save_result(result, output)

    return hits/num_user, ndcg/num_user


def main():
    args = parse_args()
    init_distributed(args)
    args.seed = args.rank
    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)

    # Save configuration to file
    config = {k: v for k, v in args.__dict__.items()}
    config['timestamp'] = "{:.0f}".format(datetime.utcnow().timestamp())
    config['local_timestamp'] = str(datetime.now())
    run_dir = "./run/neumf/{}".format(config['timestamp'])

    # Check that GPUs are actually available
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # Check where to put data loader
    if use_cuda:
        dataloader_device = 'cpu' if args.cpu_dataloader else 'cuda'
    else:
        dataloader_device = 'cpu'

    # more like load trigger timmer now
    mlperf_log.ncf_print(key=mlperf_log.PREPROC_HP_NUM_EVAL, value=args.valid_negative)
    # The default of np.random.choice is replace=True, so does pytorch random_()
    mlperf_log.ncf_print(key=mlperf_log.PREPROC_HP_SAMPLE_EVAL_REPLACEMENT, value=True)
    mlperf_log.ncf_print(key=mlperf_log.INPUT_HP_SAMPLE_TRAIN_REPLACEMENT, value=True)
    mlperf_log.ncf_print(key=mlperf_log.INPUT_STEP_EVAL_NEG_GEN)

    # sync worker before timing.
    torch.cuda.synchronize()

    #===========================================================================
    #== The clock starts on loading the preprocessed data. =====================
    #===========================================================================
    mlperf_log.ncf_print(key=mlperf_log.RUN_START)
    run_start_time = time.time()
    nb_users=498975
    nb_items=427888
    # Create model
    model = NeuMF(nb_users, nb_items,
                  mf_dim=args.factors, mf_reg=0.,
                  mlp_layer_sizes=args.layers,
                  mlp_layer_regs=[0. for i in args.layers])
    print(model)
    print("{} parameters".format(utils.count_parameters(model)))

    # Save model text description
    #with open(os.path.join(run_dir, 'model.txt'), 'w') as file:
    #    file.write(str(model))

    # Add optimizer and loss to graph
    params = model.parameters()

    optimizer = torch.optim.Adam(params, lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.eps)
    criterion = nn.BCEWithLogitsLoss(reduction = 'none') # use torch.mean() with dim later to avoid copy to host
    mlperf_log.ncf_print(key=mlperf_log.OPT_LR, value=args.learning_rate)
    mlperf_log.ncf_print(key=mlperf_log.OPT_NAME, value="Adam")
    mlperf_log.ncf_print(key=mlperf_log.OPT_HP_ADAM_BETA1, value=args.beta1)
    mlperf_log.ncf_print(key=mlperf_log.OPT_HP_ADAM_BETA2, value=args.beta2)
    mlperf_log.ncf_print(key=mlperf_log.OPT_HP_ADAM_EPSILON, value=args.eps)
    mlperf_log.ncf_print(key=mlperf_log.MODEL_HP_LOSS_FN, value=mlperf_log.BCE)

    if use_cuda:
        # Move model and loss to GPU
        model = model.cuda()
        criterion = criterion.cuda()
    if args.distributed:
        model=DDP(model)
    local_batch = args.batch_size
    traced_criterion = torch.jit.trace(criterion.forward, (torch.rand(local_batch,1),torch.rand(local_batch,1)))
    success = False
    mlperf_log.ncf_print(key=mlperf_log.TRAIN_LOOP)
    for epoch in range(args.epochs):

        mlperf_log.ncf_print(key=mlperf_log.TRAIN_EPOCH, value=epoch)
        mlperf_log.ncf_print(key=mlperf_log.INPUT_HP_NUM_NEG, value=args.negative_samples)
        mlperf_log.ncf_print(key=mlperf_log.INPUT_STEP_TRAIN_NEG_GEN)
        begin = time.time()
        
        st = timeit.default_timer()
        epoch_users = torch.load("./dataset/epoch_users.pt")
        epoch_items = torch.load("./dataset/epoch_items.pt")
        epoch_label = torch.load("./dataset/epoch_label.pt")
        epoch_users_list = epoch_users.split(local_batch)
        epoch_items_list = epoch_items.split(local_batch)
        epoch_label_list = epoch_label.split(local_batch)
        num_batches=200
        start_train = time.time()
        print("number batches: ", num_batches)
        for i in range(num_batches):
            # selecting input from prepared data
            user = epoch_users_list[i%5].cuda()
            item = epoch_items_list[i%5].cuda()
            label = epoch_label_list[i%5].view(-1,1).cuda()

            for p in model.parameters():
                p.grad = None

            outputs = model(user, item)
            loss = traced_criterion(outputs, label).float()
            loss = torch.mean(loss.view(-1), 0)

            loss.backward()
            optimizer.step()
            if i%10==0:
                print("["+str(i)+"/"+str(num_batches)+"]loss: ", float(loss))
            if (i+1)/200==1:
                break
       
        del epoch_users, epoch_items, epoch_label, epoch_users_list, epoch_items_list, epoch_label_list, user, item, label
        train_time = time.time() - begin
        print("train time:", time.time()-start_train)

    run_stop_time = time.time()
    mlperf_log.ncf_print(key=mlperf_log.RUN_FINAL)

    # easy way of tracking mlperf score
    if success:
        print("mlperf_score", run_stop_time - run_start_time)

if __name__ == '__main__':
    main()
