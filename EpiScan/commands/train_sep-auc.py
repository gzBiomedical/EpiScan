from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import average_precision_score as average_precision
from tqdm import tqdm
from typing import Callable, NamedTuple, Optional
import pickle

import sys
import argparse
import h5py
import subprocess as sp
import numpy as np
import pandas as pd
import gzip as gz
from sklearn.metrics import roc_auc_score

import sys,os
sys.path.append(os.getcwd())

from EpiScan.__init__ import __version__
from EpiScan.utils_sep import (
    PairedDataset,
    collate_paired_sequences,
    log,
    load_hdf5_parallel,
)
from EpiScan.models.embedding import FullyConnectedEmbed
from EpiScan.models.contact_sep import ContactCNN
from EpiScan.models.interaction_sep import ModelInteraction

from EpiScan.selfLoss.BdiceLoss import BinaryDiceLoss,dice_coeff
from EpiScan.selfLoss.Bfocalloss import BinaryFocalLoss
from libauc.losses import CompositionalAUCLoss,AUCMLoss
from libauc.optimizers import PDSCA
from EpiScan.models.deep_ppi import DeepPPI
from typing import Callable, NamedTuple, Optional

def corr2d(matrix, kernel):
    h, w = kernel.shape
    output = torch.zeros(matrix.shape[0]-h+1, matrix.shape[1]-w+1)
    
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):      
            output[i,j] = (matrix[i:i+h, j:j+w]*kernel).sum()
            
    return output


class TrainArguments(NamedTuple):
    cmd: str
    device: int
    train: str
    test: str
    embedding: str
    no_augment: bool
    input_dim: int
    projection_dim: int
    dropout: float
    hidden_dim: int
    kernel_width: int
    no_w: bool
    no_sigmoid: bool
    do_pool: bool
    pool_width: int
    num_epochs: int
    batch_size: int
    weight_decay: float
    lr: float
    interaction_weight: float
    run_tt: bool
    glider_weight: float
    glider_thresh: float
    outfile: Optional[str]
    save_prefix: Optional[str]
    checkpoint: Optional[str]
    func: Callable[[TrainArguments], None]


def add_args(parser):

    data_grp = parser.add_argument_group("Data")
    proj_grp = parser.add_argument_group("Embedding Block")
    contact_grp = parser.add_argument_group("Contact Block")
    inter_grp = parser.add_argument_group("Interaction Block")
    train_grp = parser.add_argument_group("Training")
    misc_grp = parser.add_argument_group("Output and Device")

    # Data
    data_grp.add_argument(
        "--train", required=True
    )
    data_grp.add_argument(
        "--test", required=True
    )
    data_grp.add_argument(
        "--embedding",
        required=True,
    )
    data_grp.add_argument(
        "--no-augment",
        action="store_true",
    )

    # Embedding Block
    proj_grp.add_argument(
        "--input-dim",
        type=int,
        default=6165,   
    )
    proj_grp.add_argument(
        "--projection-dim",
        type=int,
        default=46,  
    )
    proj_grp.add_argument(
        "--dropout-p",
        type=float,
        default=0.5,      
    )

    # Contact Block
    contact_grp.add_argument(
        "--hidden-dim",
        type=int,
        default=23,   #50
    )
    contact_grp.add_argument(
        "--kernel-width",
        type=int,
        default=7,
    )

    # Interaction Block
    inter_grp.add_argument(
        "--no-w",
        action="store_true",
    )
    inter_grp.add_argument(
        "--no-sigmoid",
        action="store_true",
    )
    inter_grp.add_argument(
        "--do-pool",
        action="store_true",
    )
    inter_grp.add_argument(
        "--pool-width",
        type=int,
        default=9,
    )

    # Training
    train_grp.add_argument(
        "--num-epochs",
        type=int,
        default=1,     #10
    )

    train_grp.add_argument(
        "--batch-size",
        type=int,
        default=2,    #25
    )
    train_grp.add_argument(
        "--weight-decay",
        type=float,
        default=0,
    )
    train_grp.add_argument(
        "--lr",
        type=float,
        default=0.0001,
    )
    train_grp.add_argument(
        "--lambda",
        dest="interaction_weight",
        type=float,
        default=0.35,
    )

    # Topsy-Turvy
    train_grp.add_argument(
        "--topsy-turvy",
        dest="run_tt",
        action="store_true",
    )
    train_grp.add_argument(
        "--glider-weight",
        dest="glider_weight",
        type=float,
        default=0.2,
    )
    train_grp.add_argument(
        "--glider-thresh",
        dest="glider_thresh",
        type=float,
        default=0.925,
    )

    # Output
    misc_grp.add_argument(
        "-o", "--outfile"
    )
    misc_grp.add_argument(
        "--save-prefix"
    )
    misc_grp.add_argument(
        "-d", "--device", type=int, default=-1
    )
    misc_grp.add_argument(
        "--checkpoint"
    )

    return parser


def predict_cmap_interaction(model,modelSeq, n0, n1, tensors, encoding_dict,con_cdr_dict, catsite, use_cuda):  

    b = len(n0)

    p_hat = []
    c_map_mag = []
    prob_map = []
    for i in range(b):
        z_a = tensors[n0[i]]
        z_b = tensors[n1[i]]

        index_cdrlist = [a for a, b in enumerate(con_cdr_dict[n1[i]]) if b == 1]
        meta_a = torch.tensor(encoding_dict[n0[i]]).unsqueeze(0)
        meta_a = meta_a.to(torch.float)
        z_a = torch.cat([z_a[:,:,:], meta_a], 2)  


        if use_cuda:
            z_a = z_a.cuda()
            z_b = z_b.cuda()
        cm, ph = model.map_predict(z_a, z_b, catsite[i],index_cdrlist)   
        p_hat.append(ph)
        c_map_mag.append(torch.mean(cm))
        prob_tmp = torch.mean(cm[:,:,:,:],3).squeeze()
        padnum = 50 - (z_a.shape[1] % 50)
        p0Conseq = torch.cat((z_a, z_a[:,-padnum:,:]), 1)
        phatnew,_ = modelSeq(p0Conseq,prob_tmp)
        prob_map.append(prob_tmp)     
    
    p_hat = torch.stack(p_hat, 0)   
    c_map_mag = torch.stack(c_map_mag, 0)

    return c_map_mag,prob_map  


def predict_interaction(model,modelSeq, n0, n1, tensors, encoding_dict,con_cdr_dict, catsite, use_cuda):

    _, p_hat = predict_cmap_interaction(model,modelSeq, n0, n1, tensors, encoding_dict,con_cdr_dict, catsite, use_cuda)

    return p_hat


def interaction_grad(
    model,
    modelSeq,
    n0,
    n1,
    y,
    catsite,                 
    tensors,
    encoding_dict,
    con_cdr_dict,
    accuracy_weight=0.95,   
    run_tt=False,
    glider_weight=0,
    glider_map=None,
    glider_mat=None,
    use_cuda=True,
):


    c_map_mag, p_hat = predict_cmap_interaction(
        model,modelSeq, n0, n1, tensors, encoding_dict, con_cdr_dict, catsite, use_cuda     
    )


    if use_cuda:
        y = y.cuda()
    y = Variable(y)

    # p_hat = p_hat.float()

    diceloss = BinaryDiceLoss()
    focalloss = BinaryFocalLoss()
    aucloss = CompositionalAUCLoss()
    klLoss =  nn.KLDivLoss()
    # aucloss = AUCMLoss()

    bce_loss_list = []
    dice_loss_list = []
    focal_loss_list = []
    auc_loss_list = []
    for ii in range(len(y)):

        bce_loss_temp = F.binary_cross_entropy(p_hat[ii].float(), y[ii][:len(p_hat[ii])].float()) 
        bce_loss_list.append(bce_loss_temp)
        bce_tensor=torch.stack(bce_loss_list,0)

        dice_loss_temp = diceloss(p_hat[ii].float(), y[ii][:len(p_hat[ii])].float()) 
        dice_loss_list.append(dice_loss_temp)
        dice_tensor=torch.stack(dice_loss_list,0)

        focal_loss_temp = focalloss(p_hat[ii].float(), y[ii][:len(p_hat[ii])].float()) 
        # print(focal_loss_temp)
        focal_loss_list.append(focal_loss_temp)
        focal_tensor=torch.stack(focal_loss_list,0)

        auc_loss_temp = aucloss(p_hat[ii].float(), y[ii][:len(p_hat[ii])].float()) 
        # print(focal_loss_temp)
        auc_loss_list.append(auc_loss_temp.flatten())
        auc_tensor=torch.stack(auc_loss_list,0)

    # bce_loss = bce_tensor.mean()
    dice_loss = dice_tensor.mean()
    focal_loss = focal_tensor.mean()
    auc_loss = auc_tensor.mean()
    bce_loss = bce_tensor.mean()


    # representation_loss = torch.mean(c_map_mag)
    loss = (0.95 * dice_loss) + (
        (1-0.95) * bce_loss     #representation_loss
    )
    # loss = bce_loss

    b = len(p_hat)

    p_hat[0].retain_grad()
    # Backprop Loss

    loss.backward()


    if use_cuda:
        y = y.cpu()
        # p_hat = p_hat.cpu()
        p_hat = [x.cpu() for x in p_hat]
        if run_tt:
            g_score = g_score.cpu()

    with torch.no_grad():
        correct_list = []
        mse_list = []
        b_ii_list = []
        auc_list = []
        guess_cutoff = 0.5     #####0.5
        for ii in range(len(p_hat)):
            b_ii = len(p_hat[ii])
            p_hat[ii] = p_hat[ii].float()
            p_guess = (guess_cutoff * torch.ones(b_ii) < p_hat[ii]).float()
            y[ii] = y[ii].float()
            correct_temp = torch.sum(p_guess == y[ii][:b_ii]).item()
            mse_temp = torch.mean((y[ii][:b_ii].float() - p_hat[ii]) ** 2).item()
            correct_list.append(correct_temp)
            mse_list.append(mse_temp)
            b_ii_list.append(b_ii)

            try:
                auc_temp = roc_auc_score(y[ii][:b_ii], p_hat[ii].data.cpu().numpy())
            except ValueError:
                auc_temp = 0.5
            auc_list.append(auc_temp)

        correct = np.mean(correct_list)
        mse = np.mean(mse_list)
        bacc = np.mean(b_ii_list)
        aucc = np.mean(auc_list)

    del b_ii_list,correct_list,mse_list
    del auc_loss_list,auc_tensor,auc_loss_temp,dice_loss_list,dice_tensor,dice_loss_temp,b_ii,p_hat,y,focal_loss_list,focal_tensor,focal_loss_temp  #,bce_loss_list,bce_tensor,bce_loss_temp
    torch.cuda.empty_cache()

    return loss, correct, mse, b, bacc, aucc


def interaction_eval(model,modelSeq, test_iterator, tensors, encoding_dict,con_cdr_dict, use_cuda):

    p_hat = []
    true_y = []

    for n0, n1, y, catsite in test_iterator:
        p_hat.append(predict_interaction(model,modelSeq, n0, n1, tensors, encoding_dict,con_cdr_dict, catsite, use_cuda))

        true_y.append(y)
    y = torch.stack(true_y)
    y = y.squeeze()


    if use_cuda:
        y = [x.cuda() for x in y]
        y = torch.stack(y)
        y.cuda()
        y = torch.reshape(y,(-1,y.shape[2]))
        p_hat = sum(p_hat, [])


    diceloss = BinaryDiceLoss()
    focalloss = BinaryFocalLoss()
    klloss = nn.KLDivLoss(reduction="batchmean")
    aucloss = CompositionalAUCLoss()

    bce_loss_list = []
    dice_loss_list = []
    klloss_loss_list = []
    auc_loss_list = []
    for ii in range(len(y)):

        bce_loss_temp = F.binary_cross_entropy(p_hat[ii].float(), y[ii][:len(p_hat[ii])].float()) 
        bce_loss_list.append(bce_loss_temp)
        bce_tensor=torch.stack(bce_loss_list,0)

        dice_loss_temp = diceloss(p_hat[ii].float(), y[ii][:len(p_hat[ii])].float()) 
        dice_loss_list.append(dice_loss_temp)
        dice_tensor=torch.stack(dice_loss_list,0)

        klloss_loss_temp = klloss(p_hat[ii].float(), y[ii][:len(p_hat[ii])].float()) 
        klloss_loss_list.append(klloss_loss_temp)
        klloss_tensor=torch.stack(klloss_loss_list,0)

        auc_loss_temp = aucloss(p_hat[ii].float(), y[ii][:len(p_hat[ii])].float()) 
        auc_loss_list.append(auc_loss_temp.flatten())
        auc_tensor=torch.stack(auc_loss_list,0)

    dice_loss = dice_tensor.mean()
    klloss_loss = klloss_tensor.mean()
    auc_loss = auc_tensor.mean()
    bce_loss = bce_tensor.mean()

    loss = (0.9 * dice_loss) + (
        (0.05) * bce_loss    
        + 
        (0.05) * klloss_loss
    )

    # loss = bce_loss

    with torch.no_grad():
        esp = 1e-6
        guess_cutoff = torch.Tensor([0.5]).float()   
        y = y.float()
        correct_list = []
        mse_list = []
        pr_list = []
        re_list = []
        f1_list = []
        dice_list = []
        guess_cutoff = 0.5    
        for ii in range(len(p_hat)):
            b_ii = len(p_hat[ii])
            p_hat[ii] = p_hat[ii].float()
            p_guess = (guess_cutoff * torch.ones(b_ii).cuda() < p_hat[ii]).float()
            y[ii] = y[ii].float()
            correct_temp = torch.sum(p_guess == y[ii][:b_ii]).item()
            mse_temp = torch.mean((y[ii][:b_ii].float() - p_hat[ii]) ** 2).item()
            correct_list.append(correct_temp)
            mse_list.append(mse_temp)

            tp_temp = torch.sum(y[ii][:b_ii] * p_guess).item()   #p_hat[ii]
            pr_temp = tp_temp / (torch.sum(p_guess).item() + esp)     #p_hat[ii]
            re_temp = tp_temp / (torch.sum(y[ii][:b_ii]).item() + esp)
            f1_temp = 2 * pr_temp * re_temp / (pr_temp + re_temp + esp)
            dice_temp = dice_coeff(p_guess.data.cpu(),y[ii][:b_ii].data.cpu())
            pr_list.append(pr_temp)
            re_list.append(re_temp)
            f1_list.append(f1_temp)
            dice_list.append(dice_temp)

        correct = np.mean(correct_list)
        mse = np.mean(mse_list)
        pr = np.mean(pr_list)
        re = np.mean(re_list)
        f1 = np.mean(f1_list)
        dice_score = np.mean(dice_list)


    aupr_list = []  
    auc_list = []
    b_ii_list = []
    y = y.data.cpu().numpy()  
    for ii in range(len(p_hat)):
        b_ii = len(p_hat[ii])
        b_ii_list.append(b_ii)
        aupr_temp = average_precision(y[ii][:b_ii], p_hat[ii].data.cpu().numpy())
        try:
            auc_temp = roc_auc_score(y[ii][:b_ii], p_hat[ii].data.cpu().numpy())
        except ValueError:
            auc_temp = 0.5
        aupr_list.append(aupr_temp)
        auc_list.append(auc_temp)
    aupr = np.mean(aupr_list)
    aucc = np.mean(auc_list)
    bacc = np.mean(b_ii_list)

    del aupr_list,b_ii_list,correct_list,mse_list,pr_list,re_list,f1_list,auc_list,auc_temp
    del dice_loss_list,dice_tensor,dice_loss_temp,b_ii,p_hat,y,dice_list,dice_temp  
    torch.cuda.empty_cache()


    return loss, correct, mse, pr, re, f1, aupr, bacc, aucc, dice_score


def train_model(args, output):
    # Create data sets

    batch_size = args.batch_size
    use_cuda = (args.device > -1) and torch.cuda.is_available()
    train_fi = args.train
    test_fi = args.test
    no_augment = args.no_augment

    embedding_h5 = args.embedding
    h5fi = h5py.File(embedding_h5, "r")

    train_df = pd.read_csv(train_fi, sep="\t", header=None)
    train_df.columns = ["prot1", "prot2", "label", "catnum"]    

    if no_augment:
        train_p1 = train_df["prot1"]
        train_p2 = train_df["prot2"]
        trainx_str = train_df["label"].values
        trainx_strcat = train_df["catnum"].values
        trainx_trans = []
        for i in range(len(trainx_str)):
            as_temp = np.array(list(map(int, list(str(trainx_str[i])))))
            as_temp = np.pad(as_temp,(0,2000- len(as_temp)))
            trainx_trans.append(as_temp)
        train_y = torch.from_numpy(np.array(trainx_trans))   

        trainx_transcat = []
        for i in range(len(trainx_strcat)):
            as_temp = int(str(trainx_strcat[i]))
            trainx_transcat.append(as_temp)
        train_ycatsite = torch.from_numpy(np.array(trainx_transcat))   

    else:
        train_p1 = pd.concat(
            (train_df["prot1"], train_df["prot2"]), axis=0
        ).reset_index(drop=True)
        train_p2 = pd.concat(
            (train_df["prot2"], train_df["prot1"]), axis=0
        ).reset_index(drop=True)
        trainx_str = pd.concat((train_df["label"], train_df["label"])).values
        train_y = torch.from_numpy(trainx_str)


    train_dataset = PairedDataset(train_p1, train_p2, train_y, train_ycatsite)    
    train_iterator = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_paired_sequences,
        shuffle=True,
        # num_workers=0,
    )
    log(f"Loaded {len(train_p1)} training pairs", file=output)
    output.flush()

    test_df = pd.read_csv(test_fi, sep="\t", header=None)
    test_df.columns = ["prot1", "prot2", "label", "catnum"]       
    test_p1 = test_df["prot1"]
    test_p2 = test_df["prot2"]
    testx_str = test_df["label"].values
    testx_strcat = test_df["catnum"].values
    testx_trans = []
    for i in range(len(testx_str)):
        as_temp = np.array(list(map(int, list(str(testx_str[i])))))
        as_temp = np.pad(as_temp,(0,2000- len(as_temp)))
        testx_trans.append(as_temp)
    test_y = torch.from_numpy(np.array(testx_trans)) 
    
    testx_transcat = []
    for i in range(len(testx_strcat)):
        as_temp = int(str(testx_strcat[i]))
        testx_transcat.append(as_temp)
    test_ycatsite = torch.from_numpy(np.array(testx_transcat)) 

    test_dataset = PairedDataset(test_p1, test_p2, test_y,test_ycatsite) 
    test_iterator = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_paired_sequences,
        shuffle=False,
        # num_workers=0,
    )

    log(f"Loaded {len(test_p1)} test pairs", file=output)
    log("Loading embeddings...", file=output)
    output.flush()

    embeddings = {}
    all_proteins = set(train_p1).union(train_p2).union(test_p1).union(test_p2)
    for prot_name in tqdm(all_proteins):
        embeddings[prot_name] = torch.from_numpy(h5fi[prot_name][:, :])
    # embeddings = load_hdf5_parallel(embedding_h5, all_proteins)

    #####loading aaMeta data
    path = "./dataProcess/publicPairs/con_pdb_dict_AgAb.pickle"   
    with open(path , "rb") as fh:
        encoding_dict = pickle.load(fh)

    #####loading con_cdr_dict data
    pathh = "./dataProcess/publicPairs/con_cdr_dict.pickle"    
    with open(pathh , "rb") as fhh:
        con_cdr_dict = pickle.load(fhh)

    if args.checkpoint is None:

        # Create embedding model
        input_dim = args.input_dim
        projection_dim = args.projection_dim
        dropout_p = args.dropout_p
        embedding_model = FullyConnectedEmbed(
            input_dim, projection_dim, dropout=dropout_p
        )
        embedding_modelAg = FullyConnectedEmbed(
            46, projection_dim, dropout=dropout_p
        )
        log("Initializing embedding model with:", file=output)
        log(f"\tprojection_dim: {projection_dim}", file=output)
        log(f"\tdropout_p: {dropout_p}", file=output)

        # Create contact model
        hidden_dim = args.hidden_dim
        kernel_width = args.kernel_width
        log("Initializing contact model with:", file=output)
        log(f"\thidden_dim: {hidden_dim}", file=output)
        log(f"\tkernel_width: {kernel_width}", file=output)

        contact_model = ContactCNN(projection_dim, hidden_dim, kernel_width)
        # Create the full model
        do_w = not args.no_w
        do_pool = args.do_pool
        pool_width = args.pool_width
        do_sigmoid = not args.no_sigmoid
        log("Initializing interaction model with:", file=output)
        log(f"\tdo_poool: {do_pool}", file=output)
        log(f"\tpool_width: {pool_width}", file=output)
        log(f"\tdo_w: {do_w}", file=output)
        log(f"\tdo_sigmoid: {do_sigmoid}", file=output)
        model = ModelInteraction(
            embedding_model,
            embedding_modelAg,
            contact_model,
            use_cuda,
            do_w=do_w,
            pool_size=pool_width,
            do_pool=do_pool,
            do_sigmoid=do_sigmoid,
        )

        log(model, file=output)

    else:
        log(
            "Loading model from checkpoint {}".format(args.checkpoint),
            file=output,
        )
        model = torch.load(args.checkpoint)
        model.use_cuda = use_cuda


    modelSeq = DeepPPI(50,5)    #DeepPPI(50,5) 
    log(modelSeq, file=output)
        

    if use_cuda:
        model.cuda()
        modelSeq = modelSeq.cuda()

    # Train the model
    lr = args.lr
    wd = args.weight_decay
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    inter_weight = args.interaction_weight
    cmap_weight = 1 - inter_weight
    digits = int(np.floor(np.log10(num_epochs))) + 1
    save_prefix = args.save_prefix

    params = [p for p in model.parameters() if p.requires_grad]
    paramsSeq = [p for p in modelSeq.parameters() if p.requires_grad]
    optim = torch.optim.NAdam(params, lr=lr)
    optimSeq = torch.optim.NAdam(paramsSeq, lr=1e-4)

    log(f'Using save prefix "{save_prefix}"', file=output)
    log(f"Training with Adam: lr={lr}, weight_decay={wd}", file=output)
    log(f"\tnum_epochs: {num_epochs}", file=output)
    log(f"\tbatch_size: {batch_size}", file=output)
    log(f"\tinteraction weight: {inter_weight}", file=output)
    log(f"\tcontact map weight: {cmap_weight}", file=output)
    output.flush()

    batch_report_fmt = (
        "[{}/{}] training {:.1%}: Loss={:.6}, Accuracy={:.3%}, MSE={:.6}, AUC={:.6}"
    )
    epoch_report_fmt = "Finished Epoch {}/{}: Loss={:.6}, Accuracy={:.3%}, MSE={:.6}, Precision={:.6}, Recall={:.6}, F1={:.6}, AUPR={:.6}, AUC={:.6}, DICE={:.6}"

    N = len(train_iterator) * batch_size

    for epoch in range(num_epochs):

        model.train()
        modelSeq.train()

        n = 0
        nacc = 0
        loss_accum = 0
        acc_accum = 0
        mse_accum = 0

        # Train batches
        for (z0, z1, y, catsite) in train_iterator:    #for (z0, z1, y) in train_iterator:

            loss, correct, mse, b, bacc, aucc = interaction_grad(
                model,
                modelSeq,
                z0,
                z1,
                y,
                catsite,
                embeddings,
                encoding_dict,
                con_cdr_dict,
                accuracy_weight=inter_weight,
                use_cuda=use_cuda,
            )

            n += b
            delta = b * (loss - loss_accum)
            loss_accum += delta / n

            nacc += bacc 
            delta = correct - bacc * acc_accum     ## b
            acc_accum += delta / nacc

            delta = b * (mse - mse_accum)
            mse_accum += delta / n

            report = (n - b) // 100 < n // 100

            optim.step()
            optim.zero_grad()
            model.clip()
            
            optimSeq.step()
            optimSeq.zero_grad()
            # modelSeq.clip()

            if report:
                tokens = [
                    epoch + 1,
                    num_epochs,
                    n / N,
                    loss_accum,
                    acc_accum,
                    mse_accum,
                    aucc,
                ]
                log(batch_report_fmt.format(*tokens), file=output)
                output.flush()

        model.eval()
        modelSeq.eval()



        with torch.no_grad():

            (
                inter_loss,
                inter_correct,
                inter_mse,
                inter_pr,
                inter_re,
                inter_f1,
                inter_aupr,
                bacc,
                inter_auc,
                inter_dice,
            ) = interaction_eval(model,modelSeq, test_iterator, embeddings,encoding_dict,con_cdr_dict, use_cuda)
            tokens = [
                epoch + 1,
                num_epochs,
                inter_loss,
                inter_correct / (bacc),    # len(test_iterator) * batch_size
                inter_mse,
                inter_pr,
                inter_re,
                inter_f1,
                inter_aupr,
                inter_auc,
                inter_dice,
            ]
            savetokens = [
                inter_pr,
                inter_re,
                inter_f1,
                inter_aupr,
                inter_auc,
                inter_dice,
            ]
            log(epoch_report_fmt.format(*tokens), file=output)

            if save_prefix is not None:
                save_path = save_prefix +str(epoch + 1)+ "_final.sav"
                save_pathSeq = save_prefix +str(epoch + 1)+ "Seq_final.sav"
                log(f"Saving final model to {save_path}", file=output)
                model.cpu()
                modelSeq.cpu()
                torch.save(model, save_path)
                torch.save(modelSeq, save_pathSeq)
                if use_cuda:
                    model.cuda()
                    modelSeq.cuda()
                    output.flush()

            if epoch % 200 == 0:
                    save_path = save_prefix +str(epoch)+ "_final.sav"
                    save_pathSeq = save_prefix +str(epoch)+ "Seq_final.sav"
                    log(f"Saving final model to {save_path}", file=output)
                    model.cpu()
                    modelSeq.cpu()
                    torch.save(model, save_path)
                    torch.save(modelSeq, save_pathSeq)
                    if use_cuda:
                        model.cuda()
                        modelSeq.cuda()
                        output.flush()


        output.flush()


def main(args):
 
    output = args.outfile
    if output is None:
        output = sys.stdout
    else:
        output = open(output, "w")

    log(f"EpiScan Version {__version__}", file=output, print_also=True)
    log(f'Called as: {" ".join(sys.argv)}', file=output, print_also=True)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
