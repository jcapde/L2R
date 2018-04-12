#!/usr/bin/env python3

import pfa
from utils import number_partitions
import json
import csv
from multiprocessing import Pool
import sys
from scipy.io import loadmat
from utils import load_data_set, doc_ids_partitions


if __name__ == "__main__":
    dataset = sys.argv[1]
    N = int(sys.argv[2])
    W = int(sys.argv[3])
    K = int(sys.argv[4])
    Ntest = int(sys.argv[5])
    max_num_partitions = int(float(sys.argv[6]))
    num_threads = int(sys.argv[7])

    num_samples = 1e4
    num_partials = 100

    opts = "{\"num_threads\" :" + str(num_threads) + ", \"num_partials\":" + str(num_partials) + ", \"num_samples\":" + str(num_samples) + "}"

    y_D, p, r, Phi = load_data_set(dataset, N, W, K)

    if max_num_partitions > 0:
        docs = doc_ids_partitions(max_num_partitions, Ntest, y_D, K)
        counter = len(docs)
    else:
        docs = list(range(Ntest))
        counter = Ntest

    print("Direct Sampling")
    pr = pfa.inference_ds("DS", opts, y_D[docs,:], Phi, r, p)
    
    with open('python/output/DS_'+dataset+"_"+str(counter)+"Ntest_"+str(W)+'W_'+str(K)+'K.json', 'w') as outfile:
        json.dump(pr, outfile)
