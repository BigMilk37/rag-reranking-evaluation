"""
LiT5-Distill-large-v2 listwise re-ranker — standalone, no rank_llm.

Mirrors the exact calling convention from castorini/LiT5/FiD/LiT5-Distill.py:
  - Uses src.data.Collator for tokenization (handles FiD passage reshaping)
  - Flattens (batch, n_passages, seq_len) -> (batch, n_passages*seq_len)
  - Calls model.generate() which internally un-flattens for encoder

Requirements: transformers==4.44.2, torch, beir
Usage:
  python run_lit5.py --first-stage bm25
  python run_lit5.py --first-stage splade
"""

import argparse
import json
import os
import sys
import csv
import copy
import time
import subprocess
import torch
import numpy as np
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--first-stage", type=str, choices=["bm25", "splade"], required=True)
parser.add_argument("--model", type=str, default="castorini/LiT5-Distill-large-v2")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--text-maxlength", type=int, default=150)
parser.add_argument("--answer-maxlength", type=int, default=140)
parser.add_argument("--n-passages", type=int, default=20,
                    help="Passages per window (20 for v1, up to 100 for v2 on big GPUs)")
parser.add_argument("--n-rerank-passages", type=int, default=100,
                    help="Total candidates to rerank via sliding window")
parser.add_argument("--stride", type=int, default=10)
parser.add_argument("--skip-latency", action="store_true")
args, _ = parser.parse_known_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
datasets_list = ["nq", "scifact", "trec-covid", "nfcorpus", "fiqa", "arguana"]
model_nickname = args.model.split("/")[-1]

# ─────────────────────────────────────────────────────────────
# Clone LiT5 repo & setup imports
# ─────────────────────────────────────────────────────────────

LIT5_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LiT5")
if not os.path.exists(LIT5_DIR):
    print("Cloning castorini/LiT5 repo...")
    subprocess.run(["git", "clone", "https://github.com/castorini/LiT5.git", LIT5_DIR], check=True)

FID_DIR = os.path.join(LIT5_DIR, "FiD")
FID_SRC_DIR = os.path.join(LIT5_DIR, "FiD", "src")
for p in [FID_DIR, FID_SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Monkey-patch sys.argv before importing (their options.py calls argparse)
_original_argv = sys.argv
sys.argv = [sys.argv[0]]

import transformers
import src.data
import src.model

sys.argv = _original_argv

# ─────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────

print(f"Loading {args.model}...")
tokenizer = transformers.T5Tokenizer.from_pretrained(
    args.model, return_dict=False, legacy=False, use_fast=True
)
model = src.model.FiD.from_pretrained(args.model).cuda().eval().bfloat16()

# ─────────────────────────────────────────────────────────────
# Helper: convert our candidates to LiT5's expected format
# ─────────────────────────────────────────────────────────────

def candidates_to_lit5_format(qid, query_text, cand_list, corpus, max_passages):
    """Convert our JSON candidates to LiT5's eval_examples format."""
    ctxs = []
    for cand in cand_list[:max_passages]:
        doc_id = cand["doc_id"]
        if doc_id in corpus:
            doc = corpus[doc_id]
            ctxs.append({
                "docid": doc_id,
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            })
    return {
        "id": qid,
        "question": query_text,
        "ctxs": ctxs,
    }


def clean_response(response):
    new_response = ""
    for c in response:
        if not c.isdigit():
            new_response += " "
        else:
            new_response += c
    return new_response.strip()


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


# ─────────────────────────────────────────────────────────────
# Re-ranking loop (mirrors LiT5-Distill.py's sliding window)
# ─────────────────────────────────────────────────────────────

out_dir = f"results/reranked_{args.first_stage}/{model_nickname}"
os.makedirs(out_dir, exist_ok=True)

collator_function = src.data.Collator(
    args.text_maxlength, tokenizer,
    batch_size=args.batch_size,
    n_passages=args.n_passages,
    suffix=" Relevance Ranking: ",
)

MEASURE_LATENCY = not args.skip_latency
all_latencies = {}

for dataset in datasets_list:
    out_path = f"{out_dir}/{dataset}.json"

    if os.path.exists(out_path):
        print(f"Skipping {dataset}, already reranked with {model_nickname}.")
        continue

    cand_path = f"results/candidates_{args.first_stage}/{dataset}.json"
    data_path = f"datasets/{dataset}"

    if not os.path.exists(cand_path) or not os.path.exists(data_path):
        print(f"Missing candidates or data for {dataset}, skipping...")
        continue

    print(f"\nLoading corpus for {dataset}...")
    corpus, queries, _ = GenericDataLoader(data_path).load(split="test")

    with open(cand_path, 'r') as f:
        candidates = json.load(f)

    # Convert all queries to LiT5 format
    eval_examples = []
    qid_order = []
    for qid, cand_list in candidates.items():
        example = candidates_to_lit5_format(
            qid, queries[qid], cand_list, corpus, args.n_rerank_passages
        )
        if example["ctxs"]:
            eval_examples.append(example)
            qid_order.append(qid)

    print(f"Re-ranking {dataset} with {model_nickname} "
          f"({len(eval_examples)} queries, window={args.n_passages}, stride={args.stride})...")

    # Sliding window reranking — exactly as in LiT5-Distill.py
    query_latencies = []
    t_dataset_start = time.perf_counter()

    window_size = args.n_passages
    n_rerank = args.n_rerank_passages

    for window_start_idx in range(n_rerank - window_size, -1, -args.stride):
        print(f"  Window: passages {window_start_idx} to {window_start_idx + window_size}")

        eval_dataset = src.data.Dataset(
            eval_examples,
            args.n_passages,
            start_pos=window_start_idx,
            question_prefix='Search Query:',
            passage_prefix='Passage:',
            passage_numbering=True,
        )

        from torch.utils.data import DataLoader, SequentialSampler
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.batch_size,
            num_workers=2,
            collate_fn=collator_function,
        )

        # Generate permutations
        generated_permutations = []
        with torch.no_grad():
            for batch in eval_dataloader:
                (idx, passage_ids, passage_mask, query_batch) = batch
                passage_ids = passage_ids.contiguous().view(passage_ids.size(0), -1)
                passage_mask = passage_mask.contiguous().view(passage_mask.size(0), -1)

                outputs = model.generate(
                    input_ids=passage_ids.cuda(),
                    attention_mask=passage_mask.cuda(),
                    max_length=args.answer_maxlength,
                    do_sample=False,
                )

                for o in outputs:
                    output = tokenizer.decode(o, skip_special_tokens=True)
                    generated_permutations.append(output)

        # Apply permutations to reorder passages
        for i in range(len(eval_examples)):
            query_dict = eval_examples[i]
            permutation = generated_permutations[i]

            resort_passages = copy.deepcopy(
                query_dict['ctxs'][window_start_idx:window_start_idx + window_size]
            )
            if len(resort_passages) > 0:
                response = clean_response(permutation)
                response = [int(x) - 1 for x in response.split() if x.strip()]
                response = remove_duplicate(response)
                original_rank = list(range(len(resort_passages)))
                response = [s for s in response if s in original_rank]
                response = response + [t for t in original_rank if t not in response]
                for j, x in enumerate(response):
                    query_dict['ctxs'][j + window_start_idx] = resort_passages[x]

    t_dataset_end = time.perf_counter()

    if MEASURE_LATENCY and len(eval_examples) > 0:
        # Total time / num queries = avg per-query latency
        total_sec = t_dataset_end - t_dataset_start
        avg_per_query = total_sec / len(eval_examples)
        all_latencies[dataset] = {
            "total_sec": total_sec,
            "num_queries": len(eval_examples),
            "avg_ms": avg_per_query * 1000,
        }

    # Convert back to our JSON format
    reranked_results = {}
    for i, qid in enumerate(qid_order):
        query_dict = eval_examples[i]
        cands = []
        for rank, ctx in enumerate(query_dict['ctxs']):
            cands.append({
                "doc_id": ctx["docid"],
                "score": float(len(query_dict['ctxs']) - rank),
                "rank": rank + 1,
            })
        reranked_results[qid] = cands

    with open(out_path, 'w') as f:
        json.dump(reranked_results, f, indent=2)
    print(f">> Saved {dataset} to {out_path}")

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────
# Save latency
# ─────────────────────────────────────────────────────────────

if MEASURE_LATENCY and all_latencies:
    os.makedirs("results/metrics", exist_ok=True)
    csv_path = f"results/metrics/latency_{model_nickname}_{args.first_stage}.csv"

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "NumQueries", "Total_sec", "Avg_ms"])
        for ds in datasets_list:
            if ds not in all_latencies:
                continue
            info = all_latencies[ds]
            writer.writerow([
                ds, info["num_queries"],
                f"{info['total_sec']:.2f}",
                f"{info['avg_ms']:.2f}",
            ])

    total_queries = sum(v["num_queries"] for v in all_latencies.values())
    total_time = sum(v["total_sec"] for v in all_latencies.values())
    if total_queries > 0:
        print(f"\n>> {model_nickname} on {args.first_stage}: "
              f"avg={total_time/total_queries*1000:.1f}ms/query, "
              f"total={total_time/60:.1f}min "
              f"({total_queries} queries)")
    print(f">> Latency saved to {csv_path}")

print(f"\n=== DONE: {model_nickname} on {args.first_stage} ===")