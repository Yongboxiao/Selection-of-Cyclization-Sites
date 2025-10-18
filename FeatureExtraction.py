import pandas as pd
import numpy as np


AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

_LOOKUP = np.full(128, -1, dtype=np.int16)
for num, ch in enumerate(AMINO_ACIDS):
    _LOOKUP[ord(ch)] = num


def seq_to_idx(seq):
    ascii_codes = np.frombuffer(seq.encode('ascii', 'ignore'), dtype=np.uint8)
    return _LOOKUP[ascii_codes]


# Amino acid composition (AAC)
def aac_numpy(sequences) -> np.ndarray:
    n = len(sequences)
    feats = np.zeros((n, 20), dtype=float)
    for i, seq in enumerate(sequences): 
        idx = seq_to_idx(seq)
        counts = np.bincount(idx, minlength=20)
        
        feats[i] = counts / idx.size
    return feats


# Dipeptide composition (DPC)
def dpc_numpy(sequences, max_k: int = 0) -> np.ndarray:
    n = len(sequences)
    feats = np.zeros((n, (max_k + 1) * 400), dtype=float)
    for r, seq in enumerate(sequences):
        idx = seq_to_idx(seq)
        if idx.size < 2:
            continue
        for k in range(max_k + 1):
            if idx.size <= k + 1:         
                break
            left  = idx[: idx.size - k - 1]
            right = idx[k + 1 :]
            flat = left * 20 + right
            counts = np.bincount(flat, minlength=400)  
            feats[r, k * 400 : (k + 1) * 400] = counts / counts.sum()     
    return feats


def one_hot_encode_sequences(sequences, max_len=14):
   
    N = len(sequences)
    idx_matrix = np.full((N, max_len), -1, dtype=int)

    for i, seq in enumerate(sequences):
        L = min(len(seq), max_len)
        for j in range(L):
            aa = seq[j]
            if aa in AMINO_ACIDS:
                idx_matrix[i, j] = seq_to_idx(aa)

    one_hot = np.zeros((N, max_len, 20), dtype=np.float32)

    rows, cols = np.where(idx_matrix >= 0)
    aas = idx_matrix[rows, cols]
    one_hot[rows, cols, aas] = 1.0

    return one_hot.reshape(N, max_len * 20)


def summarized_featuresPC(N1):
    
    if N1 == 'P': # If the last amino acid is Pro
        return [1,0]
    elif N1 == 'C':
        return [0,1]
    else:
        return [0,0]


def summarized_featuresF(N1, C1, C3):   
    
    if N1 == 'F':
        if C3 == 'CRG':
            return [1,0,1,1]
        elif C3 == 'YRG':
            return [0,1,1,1]
        elif C1 == 'G':
            return [0,0,1,1]
        else:
            return [0,0,0,1] 
    else:
        return [0,0,0,0]


def summarized_featuresM(N1, C1, C6):  

    if N1 == 'M':
        if C6 == 'PNSFEG':
            return [1,1,1]
        elif C1 == 'G':
            return [1,0,1]
        else:
            return [0,0,1]
    else:
        return [0,0,0]


def summarized_featuresT(N1, N2, C1, C2):
    
    if N1 == 'T':
        if N2 == 'TD':
            if C2 == 'GG':
                return [1,1,0,0,1,1]
            elif C1 == 'G':
                return [1,0,0,0,1,1]
            else:
                return [0,0,0,0,1,1]
        elif C2 == 'DG':
            return [0,0,1,0,0,1]
        elif C2 == 'FG':
            return [0,0,0,1,0,1]
        elif C1 == 'G':
            return [1,0,0,0,0,1]
        else:
            return [0,0,0,0,0,1]
    else:
        return [0,0,0,0,0,0]


def summarized_featuresG(N1):

    if N1 == 'G':
        return [1]
    else:
        return [0]


def summarized_features(seq):

    N1 = seq[0]
    N2 = seq[:2]
    C1 = seq[-1]
    C2 = seq[-2:]
    C3 = seq[-3:]
    C6 = seq[-6:]
    
    PC = summarized_featuresPC(N1)
    F = summarized_featuresF(N1, C1, C3)
    M = summarized_featuresM(N1, C1, C6)
    T = summarized_featuresT(N1, N2, C1, C2)
    G = summarized_featuresG(N1)

    return PC + F + M + T + G


def extract_features(X):
    
    hc_AAC_test = aac_numpy(X)
    DPC = dpc_numpy(X, max_k=0)
    one_hot_encoded = one_hot_encode_sequences(X)
    Generated_features = np.array([summarized_features(seq) for seq in X], dtype=int)

    X_train = np.c_[Generated_features, hc_AAC_test, DPC, one_hot_encoded]
    print(hc_AAC_test.shape, DPC.shape, one_hot_encoded.shape, Generated_features.shape)
    return X_train