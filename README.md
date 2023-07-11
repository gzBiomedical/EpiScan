# EpiScan
Attention-Based Model for Antibody-Specific Epitope Mapping with Integrating Biological Priors 

## Dependencies

+ cuda >= 9.0
+ cudnn >= 7.0
+ python=3.7
+ pytorch=1.11
+ numpy >= 1.19.1
+ scikit-learn >= 0.23.2
+ biopython
+ h5py
+ matplotlib
+ tqdm

## Dataset

```
#***********************
DB1: 162 antibody/antigen complexes 
DB2: 30 antibody/antigen complexes 
Antigen Samples (.pickle). can be refered in the './dataProcess/'.
Antibody Samples (.h5). Can be downloaded from zenodo here: Published later.

The Antigen Sample data format is the following:
- list of dictionaries with complex_code(str)
- - Each dictionary has the following keys:
- - - - Indices [0,20] represent a conservation profile for that position across a set of homologous proteins obtained from PSI-BLAST (The initial column with a bias value of zero).
- - - - Indices [21,40] represent a local amino acid profile that indicates the frequency of each amino acid type within 8 Å of the residue.
- - - - Indices [41:42] represent the absolute and relative solvent accessible surface area of the residue, calculated by STRIDE.
- - - - Indices [43:45] represent the 3D coordinates of the amino acids of the antigenic molecules. 
#***********************
```

## Models

The trained models can be refered in the `./trained_model/Seq_final.sav`.

Training script : 

```
#***********************
python ./EpiScan/commands/train_sep-auc.py --train ./dataProcess/public/public_sep_trainAg.tsv --test ./dataProcess/public/public_sep_valAg.tsv --embedding ./dataProcess/public/fasta/DB1.h5 --lr 1e-4 --save-prefix ./save_model/2023 --no-augment  --device 0 --num-epochs 250 --batch-size 15
#***********************
```

Predicting script : 

```
#***********************
python ./EpiScan/commands/epimapping.py --test ./dataProcess/public/public_sep_testAg.tsv --embedding ./dataProcess/public/fasta/DB1.h5
#***********************
```



## Webserver

[coming soon]()

