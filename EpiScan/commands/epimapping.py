import argparse
import datetime
import os
import pickle
import sys
import h5py
import pandas as pd
import torch
import h5py
from tqdm import tqdm
from EpiScan.utils_sep import log, load_hdf5_parallel
import numpy as np


def main(args):
    output = args.outfile
    if output is None:
        output = sys.stdout
    else:
        output = open(output, "w")
    # Set the device
    device = args.device
    use_cuda = (device > -1) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        log(
            f"Using CUDA device {device} - {torch.cuda.get_device_name(device)}",
            file=output,
            print_also=True,
        )
    else:
        log("Using CPU", file=output, print_also=True)
        device = "cpu"
    train_model(args, output)
    output.close()


def add_args(parser):
    data_grp = parser.add_argument_group("Data")
    # Data
    data_grp.add_argument(
        "--test", required=True
    )
    data_grp.add_argument(
        "--embedding",
        required=True,
    )
    return parser


def train_model(args, output):
    # Create data sets

    test_data = args.test
    embedding_h5 = args.embedding

    # Set Device
    device = 0
    use_cuda = (device >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        log(
            f"Using CUDA device {device} - {torch.cuda.get_device_name(device)}"
        )
    else:
        log("Using CPU")


    modelCon_path = './trained_model/Seq_final.sav'
    if use_cuda:
        modelCon = torch.load(modelCon_path).cuda()
        modelCon.use_cuda = True
    else:
        modelCon = torch.load(modelCon_path, map_location=torch.device("cpu")).cpu()
        modelCon.use_cuda = False
    embPathCon = embedding_h5

    #####loading Agfeatures data
    path = "../../dataProcess/publicPairs/con_pdb_dict_AgAb.pickle"  
    with open(path , "rb") as fh:
        encoding_dict = pickle.load(fh)

    #####loading cdr data 
    pathh = "../../dataProcess/publicPairs/con_cdr_dict.pickle" 
    with open(pathh , "rb") as fhh:
        con_cdr_dict = pickle.load(fhh)

    # Load Pairs
    test_fiCon = test_data
    test_dfCon = pd.read_csv(test_fiCon, sep="\t", header=None)

    h5fiCon = h5py.File(embPathCon, "r")
    embeddingsCon = {}
    allProteinsCon = set(test_dfCon[0]).union(test_dfCon[1])
    for prot_name in tqdm(allProteinsCon):
            embeddingsCon[prot_name] = torch.from_numpy(h5fiCon[prot_name][:, :])

    n0Con = test_dfCon[0]
    n1Con = test_dfCon[1]
    p0Con = embeddingsCon[n0Con]
    p1Con = embeddingsCon[n1Con]
    if use_cuda:
        p0Con = p0Con.cuda()
        p1Con = p1Con.cuda()

    meta_acon = torch.tensor(encoding_dict[n0Con]).unsqueeze(0)
    meta_acon = meta_acon.to(torch.float)
    p0Con = torch.cat([p0Con[:,:,:], meta_acon], 2)      #aminoacid properties
    index_cdrlist = [a for a, b in enumerate(con_cdr_dict[n1Con]) if b == 1]
    cmCon,_ = modelCon.map_predict(p0Con, p1Con,test_dfCon[3],index_cdrlist) 
    probCon_map = torch.mean(cmCon,3).squeeze()
    probCon=probCon_map.cpu().detach().numpy()
    probCon.to_csv('mappingResults.csv',index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())