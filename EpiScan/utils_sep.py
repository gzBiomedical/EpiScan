from __future__ import print_function, division

import torch
import torch.utils.data

import numpy as np
import h5py
import multiprocessing as mp

from tqdm import tqdm
from functools import partial
from datetime import datetime


def log(m, file=None, timestamped=True, print_also=False):
    curr_time = f"[{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}] "
    log_string = f"{curr_time if timestamped else ''}{m}"
    if file is None:
        print(log_string)
    else:
        print(log_string, file=file)
        if print_also:
            print(log_string)
        file.flush()


def RBF(D, sigma=None):
    """
    Convert distance matrix into similarity matrix using Radial Basis Function (RBF) Kernel.
    """
    sigma = sigma or np.sqrt(np.max(D))
    return np.exp(-1 * (np.square(D) / (2 * sigma ** 2)))


def _hdf5_load_partial_func(k, file_path):
    """
    Helper function for load_hdf5_parallel
    """

    with h5py.File(file_path, "r") as fi:
        emb = torch.from_numpy(fi[k][:])
    return emb


def load_hdf5_parallel(file_path, keys, n_jobs=-1):
    """
    Load keys from hdf5 file into memory
    """
    torch.multiprocessing.set_sharing_strategy("file_system")

    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    with mp.Pool(processes=n_jobs) as pool:
        all_embs = list(
            tqdm(
                pool.imap(
                    partial(_hdf5_load_partial_func, file_path=file_path), keys
                ),
                total=len(keys),
            )
        )

    embeddings = {k: v for k, v in zip(keys, all_embs)}
    return embeddings


class PairedDataset(torch.utils.data.Dataset):
    """
    Dataset to be used by the PyTorch data loader for pairs of sequences and their labels.
    """

    def __init__(self, X0, X1, Y,catsite):
        self.X0 = X0
        self.X1 = X1
        self.Y = Y
        self.catsite = catsite
        assert len(X0) == len(X1), (
            "X0: "
            + str(len(X0))
            + " X1: "
            + str(len(X1))
            + " Y: "
            + str(len(Y))
            + " catsite: "
            + str(len(catsite))
        )
        # assert len(X0) == len(Y), (
        #     "X0: "
        #     + str(len(X0))
        #     + " X1: "
        #     + str(len(X1))
        #     + " Y: "
        #     + str(len(Y))
        # )

    def __len__(self):
        return len(self.X0)

    def __getitem__(self, i):
        return self.X0[i], self.X1[i], self.Y[i], self.catsite[i]


def collate_paired_sequences(args):
    """
    Collate function for PyTorch data loader.
    """
    x0 = [a[0] for a in args]
    x1 = [a[1] for a in args]
    y = [a[2] for a in args]
    catsite = [a[3] for a in args]             
    return x0, x1, torch.stack(y, 0),torch.stack(catsite, 0)
