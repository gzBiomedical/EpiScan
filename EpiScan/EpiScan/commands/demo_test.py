import argparse
import datetime
import os
import pandas as pd
import os
import pickle
import sys
####
# os.chdir(r"PATH")
# sys.path.append(os.getcwd())


import torch
import h5py
from tqdm import tqdm
from EpiScan.commands.utils import log, load_hdf5_parallel
import numpy as np
from EpiScan.models.embedding import FullyConnectedEmbed
from EpiScan.models.contact_sep import ContactCNN
from EpiScan.models.interaction_sep import ModelInteraction


test_data = '../dataProcess/public/public_sep_testAg.tsv'
embedding_h5 = '../dataProcess/public/DB1.h5'

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


embedding_model = FullyConnectedEmbed(
    6165, 46, 0.5
)
embedding_modelAg = FullyConnectedEmbed(
    46, 46, 0.5
)
contact_model = ContactCNN(46, 23, 7)
modelCon = ModelInteraction(
    embedding_model,
    embedding_modelAg,
    contact_model,
    use_cuda
)



modelCon_path = '../trained_model/Seq_final.pth'
modelCon.load_state_dict(torch.load(modelCon_path))
if use_cuda:
    modelCon.cuda()


#####loading Agfeatures data
path = "../dataProcess/publicPairs/con_pdb_dict_AgAb.pickle"  
with open(path , "rb") as fh:
    encoding_dict = pickle.load(fh)

#####loading cdr data 
pathh = "../dataProcess/publicPairs/con_cdr_dict.pickle" 
with open(pathh , "rb") as fhh:
    con_cdr_dict = pickle.load(fhh)

# Load Pairs
test_fiCon = test_data
test_dfCon = pd.read_csv(test_fiCon, sep="\t", header=None)

embPathCon = embedding_h5
h5fiCon = h5py.File(embPathCon, "r")
embeddingsCon = {}
allProteinsCon = set(test_dfCon[0]).union(test_dfCon[1])
for prot_name in tqdm(allProteinsCon):
        embeddingsCon[prot_name] = torch.from_numpy(h5fiCon[prot_name][:, :])

## Let's compute the epitope of the first Ag-Ab sample
indedx = 0
n0Con = test_dfCon[0][indedx]  
n1Con = test_dfCon[1][indedx]
p0Con = embeddingsCon[n0Con]
p1Con = embeddingsCon[n1Con]
if use_cuda:
    p0Con = p0Con.cuda()
    p1Con = p1Con.cuda()

meta_acon = torch.tensor(encoding_dict[n0Con]).unsqueeze(0)
meta_acon = meta_acon.to(torch.float).cuda()
p0Con = torch.cat([p0Con[:,:,:], meta_acon], 2)
index_cdrlist = [a for a, b in enumerate(con_cdr_dict[n1Con]) if b == 1]
cmCon,_ = modelCon.map_predict(p0Con, p1Con,test_dfCon[3][indedx],index_cdrlist) 
probCon_map = torch.mean(cmCon,3).squeeze()
probCon=probCon_map.cpu().detach().numpy()
df = pd.DataFrame(probCon)
df.to_csv('mappingResults.csv', index=False, header=None)


