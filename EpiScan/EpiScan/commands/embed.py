from __future__ import annotations
import argparse
import sys
import os
from typing import Callable, NamedTuple
from pathlib import Path

# 添加当前工作目录到系统路径
sys.path.append(str(Path.cwd()))

# 导入所需模块
from EpiScan.embLLM.language_model import embed_from_fasta

class EmbeddingConfig(NamedTuple):
    command: str
    compute_device: int
    output_file: str
    input_sequences: str
    process_function: Callable[[EmbeddingConfig], None]

def configure_argument_parser(arg_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    arg_parser.add_argument("--seqs", help="Input sequences for embedding", required=True)
    arg_parser.add_argument("-o", "--outfile", help="Output file path (H5 format)", required=True)
    arg_parser.add_argument("-d", "--device", type=int, default=-1, help="Computation device ID")
    return arg_parser

def process_embedding(config: EmbeddingConfig) -> None:
    print(f"Processing embedding with the following configuration:")
    print(f"Input: {config.input_sequences}")
    print(f"Output: {config.output_file}")
    print(f"Device: {config.compute_device}")
    
    embed_from_fasta(
        fastaPath=config.input_sequences,
        outputPath=config.output_file,
        device=config.compute_device,
        verbose=True
    )

def main(args: argparse.Namespace) -> None:
    config = EmbeddingConfig(
        command="embed",
        compute_device=args.device,
        output_file=args.outfile,
        input_sequences=args.seqs,
        process_function=process_embedding
    )
    config.process_function(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequence Embedding Tool")
    parser = configure_argument_parser(parser)
    args = parser.parse_args()
    main(args)

# 为了保持与其他文件的兼容性，保留原有的函数名
def add_args(parser):
    return configure_argument_parser(parser)
