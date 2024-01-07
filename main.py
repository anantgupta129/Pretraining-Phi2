import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import lightning as L
import torch
from torch.utils.data import DataLoader, IterableDataset

from lit_gpt import Tokenizer
from lit_gpt.model import GPT, Block, Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.utils import chunked_cross_entropy, estimate_flops, get_default_supported_precision, num_parameters


# Data proportions from https://arxiv.org/pdf/2302.13971.pdf Table 1
data_config = [
    ("arxiv", 200.5),
    ("book", 400.5),
    ("c4", 1500.0),
    ("cc", 6700.0),
    ("github", 400.5),
    ("stackexchange", 200.0),
    ("wikipedia", 400.5),
]

class PhiAPIDataset(IterableDataset):
    def __init__(self) -> None:
        self.url = ""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer = tokenizer
        
        self.prompts = [1, 2, 3, 4]
    
    def __iter__(self):
        for prompt in self.prompts:
            print("using phi")
            encoded = self.toeknizer.encode(str(prompt) * 4097)
            num_pad = 4097 - len(encoded)
            encoded = torch.cat([encoded, torch.Tensor([2] * num_pad)])
            yield encoded
            

def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric: L.Fabric, shuffle: bool = True, seed: int = 12345
) -> DataLoader:
    datasets = []
    for prefix, _ in data_config:
        filenames = list(data_dir.glob(f"{prefix}*"))
        if not filenames:
            raise FileNotFoundError(
                f"No files found at {str(data_dir)} with prefix {prefix}. Did you forget to run `prepare_redpajama.py`?"
            )
        dataset = PackedDataset(
            filenames,
            n_chunks=4,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)
    
    # api_data = PhiAPIDataset()
    # datasets.append(api_data)
    
    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config] #+ [0.1]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]
    print(sum_weights, weights)

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric: L.Fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader

data_path = Path(r"C:\Users\anant\Downloads\data\redpajama-shots")
fabric = L.Fabric()
train_dataloader, _ = create_dataloaders(train_data_dir=data_path, batch_size=2, block_size=4096, fabric=fabric)


cnt = 0
for data in train_dataloader:
    # print(data)
    input_ids = data[:, 0 : 123].contiguous()
    targets = data[:, 1 : 123 + 1].contiguous()
    cnt += 1
    
    # print(input_ids)
    # print("--")
    # print(targets)
    
    # break
print(cnt)
