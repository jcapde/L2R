import numpy as np
from scipy.special import binom
from csv import reader
from math import floor
import pfa

MAX_PARTITIONS = 1e9

def load_data_set(dataset, N, W, K):
    filename = "data/" + dataset + "_GaP_"+str(N)+"N_"+str(W)+"W_"+str(K)+"T.mat"
    print("Loading data/" + dataset + "_GaP_"+str(N)+"N_"+str(W)+"W_"+str(K)+"T.mat")
    y_D, p, r, Phi = pfa.load_model(filename, N, W, K)
    r = r.reshape([len(r)])
    p = p.reshape([len(p)])
    return y_D, p, r, Phi

def number_partitions(doc, K):
    return np.prod([binom(word + K - 1, K - 1) for word in doc])

def doc_ids_partitions(max_num_partitions, Ntest, y_D, K):
    docs = []
    doc_id = 0
    N, W = y_D.shape
    counter = 0
    while (counter < Ntest) and doc_id < N:
        num_part = number_partitions(y_D[doc_id,], K)
        if num_part < max_num_partitions and sum(y_D[doc_id,]>0) > 1:
            docs.append(doc_id)
            counter += 1
        doc_id += 1

    return docs;

