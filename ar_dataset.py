"""
AR Dataset: loads cached JSONL latents, builds tag vocab from CSV,
unpacks binary latents, and batches by bucket shape.

Simplified for BitDance-style training (no SoundStorm delay pattern).
"""
import os
import json
import glob
import csv
import random
from typing import List, Tuple, Dict, Any
from functools import partial
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader

# ── Constants ────────────────────────────────────────────────────────────────
NUM_CODE_GROUPS = 4       # number of RVQ groups stored in dataset
CHANNELS_PER_GROUP = 14   # bits per group
LATENT_CHANNELS = NUM_CODE_GROUPS * CHANNELS_PER_GROUP  # 56


# ── Tag Vocabulary ───────────────────────────────────────────────────────────

class TagVocab:
    """
    Loads tag vocabulary from a CSV file and remaps to sequential IDs.
    
    CSV format: tag_id,name,category,count,selected_count,train_count,...
    The raw tag_id values can be very large (up to 9999999), so we remap
    to sequential 0-indexed IDs for efficient embedding lookup.
    """
    def __init__(self, csv_path: str):
        self.name_to_id: Dict[str, int] = {}
        self.id_to_name: Dict[int, str] = {}
        
        # Load CSV and remap to sequential IDs
        sequential_id = 0
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Support different CSV header formats
                name = row.get('name') or row.get('character') or row.get('tag')
                if name is None:
                    continue
                name = name.strip()
                if name and name not in self.name_to_id:
                    self.name_to_id[name] = sequential_id
                    self.id_to_name[sequential_id] = name
                    sequential_id += 1

        self._num_tags = sequential_id
        
        # Define special tokens after regular tags
        self._sep_id = sequential_id
        self._line_sep_id = sequential_id + 1
        self._eos_id = sequential_id + 2
        self._pad_id = sequential_id + 3
        
        self.name_to_id['[SEP]'] = self._sep_id
        self.name_to_id['[LINE_SEP]'] = self._line_sep_id
        self.name_to_id['[EOS]'] = self._eos_id
        self.name_to_id['[PAD]'] = self._pad_id
        
        self.id_to_name[self._sep_id] = '[SEP]'
        self.id_to_name[self._line_sep_id] = '[LINE_SEP]'
        self.id_to_name[self._eos_id] = '[EOS]'
        self.id_to_name[self._pad_id] = '[PAD]'
        
        print(f"TagVocab: {self._num_tags} tags + 4 special = {self.vocab_size} total")

    def encode(self, prompt: str, max_tags: int = 64) -> List[int]:
        """Encode a prompt string into a list of tag token IDs."""
        if not prompt or not prompt.strip():
            return []
        
        # Tags with spaces like "Hatsune Miku" need comma splitting.
        # If there are no commas, fallback to space splitting just in case.
        if ',' in prompt:
            tags = prompt.strip().split(',')
        else:
            tags = prompt.strip().split()
            
        tag_ids = []
        for tag in tags:
            tag = tag.strip()
            if tag in self.name_to_id:
                tag_ids.append(self.name_to_id[tag])
        return tag_ids[:max_tags]

    @property
    def sep_id(self) -> int:
        return self._sep_id

    @property
    def line_sep_id(self) -> int:
        return self._line_sep_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return self._num_tags + 4


# ── Dataset ──────────────────────────────────────────────────────────────────

class StreamingLatentDataset(IterableDataset):
    """
    Streams JSONL latent files to prevent out-of-memory errors.
    Groups samples by bucket shape and yields completed batches.
    Handles multi-GPU (DDP) and multi-worker sharding automatically.
    """
    def __init__(self, data_dir: str, tag_vocab: TagVocab, batch_size: int, max_tags: int = 64, shuffle: bool = True):
        self.data_dir = data_dir
        self.tag_vocab = tag_vocab
        self.batch_size = batch_size
        self.max_tags = max_tags
        self.shuffle = shuffle
        self.jsonl_files = sorted(glob.glob(os.path.join(data_dir, '**', '*.jsonl'), recursive=True))
        print(f"Streaming dataset initialized with {len(self.jsonl_files)} JSONL files.")

    def __iter__(self):
        files = list(self.jsonl_files)
        
        # 1. Shard files for DDP (Multi-GPU)
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            files = files[rank::world_size]
            
        # 2. Shard files for DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files = files[worker_id::num_workers]
            
        if self.shuffle:
            random.shuffle(files)
        buffers = defaultdict(list)
        
        for filepath in files:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if self.shuffle:
                    random.shuffle(lines)
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                        
                    prompt = data.get('prompt', '')
                    tag_ids = self.tag_vocab.encode(prompt, self.max_tags)
                    bucket = tuple(data['bucket'])  # (H_pixels, W_pixels)
                    latent = np.array(data['latent'], dtype=np.int16)  # [4, Hq, Wq]
                    
                    sample = {
                        'tag_ids': tag_ids,
                        'bucket': bucket,
                        'latent': latent,
                        'Hq': bucket[0] // 16,
                        'Wq': bucket[1] // 16,
                    }
                    
                    buffers[bucket].append(sample)
                    
                    # If we have enough samples in this bucket for a batch, yield it
                    if len(buffers[bucket]) == self.batch_size:
                        yield buffers.pop(bucket)
                        
        # Yield any remaining partial batches
        for bucket, batch in buffers.items():
            if len(batch) > 0:
                yield batch


# ── Collation ────────────────────────────────────────────────────────────────

def collate_bitdance_fn(
    batch: List[Dict[str, Any]],
    tag_vocab: TagVocab,
    num_code_groups: int = NUM_CODE_GROUPS,
    channels_per_group: int = CHANNELS_PER_GROUP,
) -> Dict[str, torch.Tensor]:
    """
    Collate batch for BitDance-style training.
    
    Sequence layout per sample (handled by the model):
        Prefix: [PAD_h, PAD_w, tag_1, ..., tag_N, SEP]
        Image:  patchified binary latents (handled internally by model)
    
    Returns:
        tag_tokens:   [B, max_prefix_len] — padded prefix tokens
        latents:      [B, C, Hq, Wq] — binary {-1, +1}
        shape_h_ids:  [B]
        shape_w_ids:  [B]
    """
    B = len(batch)

    # ── Build tag prefix per sample ──
    prefixes = []
    for sample in batch:
        # [PAD(shape_h), PAD(shape_w), tag_1, ..., tag_N, SEP]
        prefix = [tag_vocab.pad_id, tag_vocab.pad_id] + sample['tag_ids'] + [tag_vocab.sep_id]
        prefixes.append(prefix)

    max_prefix_len = max(len(p) for p in prefixes)

    tag_tokens = torch.full((B, max_prefix_len), tag_vocab.pad_id, dtype=torch.long)
    for i, prefix in enumerate(prefixes):
        tag_tokens[i, :len(prefix)] = torch.tensor(prefix, dtype=torch.long)

    # ── Unpack binary latents ──
    Hq = batch[0]['Hq']
    Wq = batch[0]['Wq']
    C = num_code_groups * channels_per_group  # 56

    latents = torch.zeros(B, C, Hq, Wq)
    for i, sample in enumerate(batch):
        lat = torch.from_numpy(sample['latent'].astype(np.int32))  # [4, Hq, Wq]
        for g in range(num_code_groups):
            for bit in range(channels_per_group):
                latents[i, g * channels_per_group + bit] = \
                    ((lat[g] >> bit) & 1).float() * 2.0 - 1.0

    return {
        'tag_tokens': tag_tokens,
        'latents': latents,
        'shape_h_ids': torch.tensor([s['Hq'] for s in batch], dtype=torch.long),
        'shape_w_ids': torch.tensor([s['Wq'] for s in batch], dtype=torch.long),
    }


# ── Dataloader Factory ───────────────────────────────────────────────────────

def make_ar_dataloader(data_dir: str, tag_csv_path: str, batch_size: int,
                       num_workers: int = 0, max_tags: int = 64) -> Tuple[DataLoader, TagVocab]:
    """Create a DataLoader for streaming AR dataset with bucket-based batching."""
    vocab = TagVocab(tag_csv_path)
    dataset = StreamingLatentDataset(data_dir, vocab, batch_size, max_tags)

    loader = DataLoader(
        dataset,
        batch_size=None,  # Dataset natively yields batches
        collate_fn=partial(collate_bitdance_fn, tag_vocab=vocab),
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, vocab
