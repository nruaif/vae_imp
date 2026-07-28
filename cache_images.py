import argparse
import io
import json
import math
import os
import time
import glob
import uuid

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF

from model import QwenRVQAutoencoder


# ── Bucket calculation (Matches dataset.py) ─────────────────────────────────

def make_buckets(image_size):
    target_area = image_size ** 2
    aspect_ratios = [1.0, 0.75, 1.33, 0.56, 1.78]
    buckets = []
    for ar in aspect_ratios:
        h = int(math.sqrt(target_area / ar))
        w = int(h * ar)
        # Round to nearest multiple of 32 for QwenRVQ compatibility
        h = (h // 32) * 32
        w = (w // 32) * 32
        if h > 0 and w > 0:
            buckets.append((h, w))
    return buckets

def find_best_bucket(w, h, buckets):
    img_ar = w / h
    bucket_ars = [bw / bh for bh, bw in buckets]
    best_idx = min(range(len(bucket_ars)), key=lambda i: abs(bucket_ars[i] - img_ar))
    return buckets[best_idx]


# ── Image preprocessing ─────────────────────────────────────────────────────

def preprocess_image(image, target_h, target_w):
    w, h = image.size
    img_ar = w / h
    target_ar = target_w / target_h

    if img_ar > target_ar:
        new_h = target_h
        new_w = int(target_h * img_ar)
    else:
        new_w = target_w
        new_h = int(target_w / img_ar)

    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Center crop
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    image = image.crop((left, top, left + target_w, top + target_h))

    tensor = TF.to_tensor(image)
    tensor = (tensor - 0.5) * 2.0
    return tensor


# ── Extract prompt ──────────────────────────────────────────────────────────

def extract_prompt(sample):
    if "json" in sample:
        data = sample["json"]
        if isinstance(data, bytes):
            data = json.loads(data.decode("utf-8"))
        if isinstance(data, dict):
            parts = []
            if "tags" in data and isinstance(data["tags"], list):
                for tag_entry in data["tags"]:
                    if "tags" in tag_entry and isinstance(tag_entry["tags"], dict):
                        t = tag_entry["tags"]
                        for category in ["rating", "character", "general"]:
                            tag_list = t.get(category, [])
                            if isinstance(tag_list, list):
                                for item in tag_list:
                                    if isinstance(item, dict) and "name" in item:
                                        parts.append(str(item["name"]))
            else:
                for key in ["rating", "character_tags", "general_tags"]:
                    tags = data.get(key, [])
                    if isinstance(tags, list):
                        parts.extend(str(t) for t in tags)
                    elif tags:
                        parts.append(str(tags))
            if parts:
                return " ".join(parts)[:512]
            return str(data)[:512]
        return str(data)[:512]
    
    if "txt" in sample:
        txt = sample["txt"]
        if isinstance(txt, bytes):
            txt = txt.decode("utf-8")
        return txt[:512]
    
    if "caption" in sample:
        cap = sample["caption"]
        if isinstance(cap, bytes):
            cap = cap.decode("utf-8")
        return cap[:512]
    
    return ""


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cache images into JSONL with packed INT16 discrete latents")
    parser.add_argument("--input", type=str, required=True, help="WebDataset URL/glob")
    parser.add_argument("--output_dir", type=str, default="./cached_jsonl")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--bucket_size", type=int, default=32)
    parser.add_argument("--encode_batch_size", type=int, default=32)
    parser.add_argument("--checkpoint", type=str, default="", help="Path to QwenRVQAutoencoder checkpoint")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_per_file", type=int, default=10000, help="Images per JSONL file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Model
    print("Loading QwenRVQAutoencoder...")
    # FIX 1: Explicitly enable use_quant=True if you are bit-packing discrete latents
    model = QwenRVQAutoencoder(
        f=16, 
        d_enc=128, 
        d_dec=128, 
        num_groups=16, 
        channels_per_group=14,
        use_quant=True  
    )
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading weights from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model_state_dict"], strict = False)
    
    # Apply torch memory format to the model
    model = model.to(device, memory_format=torch.channels_last).eval()

    buckets = make_buckets(args.image_size)
    print(f"Buckets: {buckets}")

    bucket_data = {b: {"keys": [], "tensors": [], "prompts": []} for b in buckets}

    os.makedirs(args.output_dir, exist_ok=True)
    
    current_file_idx = 0
    current_count = 0
    f_out = None
    
    def open_next_file():
        nonlocal f_out, current_file_idx, current_count
        if f_out is not None:
            f_out.close()
        path = os.path.join(args.output_dir, f"latents-{current_file_idx:05d}.jsonl")
        f_out = open(path, "w", encoding="utf-8")
        current_file_idx += 1
        current_count = 0
        print(f"\nWriting to {path} ...")

    open_next_file()
    total_samples = 0

    def flush_bucket(bucket_key):
        nonlocal total_samples, current_count

        data = bucket_data[bucket_key]
        keys = data["keys"]
        tensors = data["tensors"]
        prompts = data["prompts"]

        if not tensors:
            return

        bh, bw = bucket_key
        n = len(tensors)

        all_packed = []
        
        for i in range(0, n, args.encode_batch_size):
            # Send batch to device with channels_last memory format
            batch = torch.stack(tensors[i:i + args.encode_batch_size]).to(
                device, memory_format=torch.channels_last
            )
            
            with torch.no_grad():
                z = model.encoder(batch) 
                
                # FIX 2: Apply the exact quantization the model uses during training
                if model.use_quant:
                    groups = z.chunk(model.num_groups, dim=1)
                    groups = [model.quant(g) for g in groups]
                    z = torch.cat(groups, dim=1)
                
                # Keep first 4 groups only (4 groups * 14 channels = 56)
                z = z[:, :56, :, :] 
                
                # Map the {-1.0, 1.0} latents to {0, 1} for INT16 packing
                z_bin = (z > 0).to(torch.int16)
                
                B, _, H, W = z_bin.shape
                # The packed tensor defaults to torch.contiguous_format (NCHW memory layout)
                # which is ideal for saving/converting to NumPy since we're no longer doing convolutions.
                z_packed = torch.zeros((B, 4, H, W), dtype=torch.int16, device=device)
                
                # Pack 14 bits into one int16 per group
                for g in range(4):
                    for bit_idx in range(14):
                        z_packed[:, g, :, :] |= (z_bin[:, g*14 + bit_idx, :, :] << bit_idx)
                        
                all_packed.append(z_packed.cpu().numpy())

        stacked_packed = np.concatenate(all_packed, axis=0)

        for idx in range(n):
            item = {
                "key": keys[idx],
                "prompt": prompts[idx],
                "bucket": [bh, bw],
                "latent": stacked_packed[idx].tolist() # Logically remains [4, H, W]
            }
            f_out.write(json.dumps(item) + "\n")
            current_count += 1
            total_samples += 1
            
            if current_count >= args.max_per_file:
                open_next_file()

        # Reset bucket
        data["keys"].clear()
        data["tensors"].clear()
        data["prompts"].clear()

    def process_sample(sample):
        image = None
        for ext in ["jpg", "png", "webp", "jpeg"]:
            if ext in sample:
                image = sample[ext]
                break

        if image is None or not isinstance(image, Image.Image):
            return None

        image = image.convert("RGB")
        w, h = image.size
        bucket_key = find_best_bucket(w, h, buckets)
        target_h, target_w = bucket_key

        try:
            tensor = preprocess_image(image, target_h, target_w)
        except Exception:
            return None

        prompt = extract_prompt(sample)
        key = sample.get("__key__", uuid.uuid4().hex)
        
        return {
            "key": key,
            "bucket_key": bucket_key,
            "tensor": tensor,
            "prompt": prompt
        }

    dataset = (
        wds.WebDataset(
            args.input,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            handler=wds.warn_and_continue
        )
        .decode("pil", handler=wds.warn_and_continue)
        .map(process_sample, handler=wds.warn_and_continue)
        .select(lambda x: x is not None)
    )

    if args.num_workers > 0:
        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            num_workers=args.num_workers,
            prefetch_factor=4
        )
    else:
        loader = dataset

    t0 = time.time()
    print("\nProcessing images...\n")

    for sample in tqdm(loader, desc="Reading", unit="img"):
        bucket_key = sample["bucket_key"]
        if isinstance(bucket_key, list):
            bucket_key = tuple(bucket_key)

        bucket_data[bucket_key]["keys"].append(sample["key"])
        bucket_data[bucket_key]["tensors"].append(sample["tensor"])
        bucket_data[bucket_key]["prompts"].append(sample["prompt"])

        if len(bucket_data[bucket_key]["tensors"]) >= args.bucket_size:
            flush_bucket(bucket_key)

    print("\nFlushing remaining partial buckets...")
    for bk in buckets:
        if bucket_data[bk]["tensors"]:
            flush_bucket(bk)

    if f_out is not None:
        f_out.close()
        
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done! {total_samples} latents written to {args.output_dir}")
    print(f"Time: {elapsed:.1f}s ({total_samples / max(elapsed, 1):.1f} img/s)")


if __name__ == "__main__":
    main()