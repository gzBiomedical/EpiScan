import os, sys, subprocess as sp, random, torch, h5py
from tqdm import tqdm
from EpiScan.embLLM.fasta import parse, parse_directory, write
from EpiScan.embLLM.pretrained import get_pretrained
from EpiScan.embLLM.alphabets import Uniprot21
from EpiScan.embLLM.utils import log
from datetime import datetime

def embed_from_fasta(fastaPath, outputPath, device=0, verbose=False):
    def setup_device():
        if (device >= 0) and torch.cuda.is_available():
            torch.cuda.set_device(device)
            return f"CUDA device {device} - {torch.cuda.get_device_name(device)}"
        return "CPU"

    def load_model():
        model = get_pretrained("lm_v1")
        model.proj.weight.data.normal_()
        model.proj.bias = torch.nn.Parameter(torch.zeros(100))
        return model.cuda() if torch.cuda.is_available() and device >= 0 else model

    def encode_sequences(seqs):
        alphabet = Uniprot21()
        return [torch.from_numpy(alphabet.encode(s.encode("utf-8"))).cuda() if torch.cuda.is_available() and device >= 0 else torch.from_numpy(alphabet.encode(s.encode("utf-8"))) for s in seqs]

    def log_info(message):
        if verbose:
            log(message)

    log_info(f"# Using {setup_device()}")
    log_info("# Loading Model...")
    model = load_model()
    model.eval()

    log_info("# Loading Sequences...")
    names, seqs = parse(fastaPath)
    encoded_seqs = encode_sequences(seqs)

    log_info(f"# {len(encoded_seqs)} Sequences Loaded")
    log_info(f"# Approximate Storage Required: ~{len(encoded_seqs) * (1 / 125)}GB")

    with h5py.File(outputPath, "w") as h5fi:
        log_info(f"# Storing to {outputPath}...")
        with torch.no_grad():
            try:
                for name, x in tqdm(zip(names, encoded_seqs), total=len(names)):
                    if name not in h5fi:
                        z = model.transform(x.long().unsqueeze(0))
                        h5fi.create_dataset(name, data=z.cpu().numpy(), compression="lzf")
            except KeyboardInterrupt:
                sys.exit(1)

if __name__ == "__main__":
    pass
