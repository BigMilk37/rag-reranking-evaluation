import time
import torch
import numpy as np
from sentence_transformers import CrossEncoder
import argparse
import csv
import os
import json
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
datasets = ["nq", "scifact", "trec-covid", "nfcorpus", "fiqa", "arguana"]
model_nickname = args.model.split("/")[-1]

print(f"=== PHASE 3: LATENCY PROFILING ({model_nickname}) ===")

if "MiniLM" in args.model:
    BATCH_SIZE = 512
    model_kwargs = {}
elif "zephyr" in args.model.lower() or "lit5" in args.model.lower():
    BATCH_SIZE = 8
    model_kwargs = {"torch_dtype": torch.bfloat16}
else:
    BATCH_SIZE = 16
    model_kwargs = {"torch_dtype": torch.float16}

print(f"Loading {args.model} into VRAM...")
model = CrossEncoder(args.model, device=DEVICE, model_kwargs=model_kwargs)

def time_model(all_query_inputs):
    """Profile the re-ranker over a list of (query, doc) pair batches.

    Runs 3 warmup iterations then measures wall-clock latency per query
    with CUDA synchronization. Returns (avg_ms, p95_ms).
    """
    if not all_query_inputs: return 0.0, 0.0

    # Warmup to avoid cold-start GPU lag
    for _ in range(3):
        model.predict(all_query_inputs[0], batch_size=BATCH_SIZE)
            
    if DEVICE == "cuda": torch.cuda.synchronize()
        
    latencies = []
    for inputs in tqdm(all_query_inputs, desc=f"Profiling"):
        start = time.perf_counter()
        model.predict(inputs, batch_size=BATCH_SIZE)
        if DEVICE == "cuda": torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
        
    return np.mean(latencies) * 1000, np.percentile(latencies, 95) * 1000

results = {ds: {} for ds in datasets}

for stage in ["bm25", "splade"]:
    print(f"\n--- Profiling on {stage.upper()} candidates ---")
    for dataset in datasets:
        data_path = f"datasets/{dataset}"
        cand_path = f"results/candidates_{stage}/{dataset}.json"
        
        if not os.path.exists(data_path) or not os.path.exists(cand_path):
            continue

        corpus, queries, _ = GenericDataLoader(data_path).load(split="test")
        with open(cand_path, 'r') as f:
            candidates = json.load(f)

        all_real_pairs = []
        for qid, cands in candidates.items():
            pairs = []
            for c in cands[:100]:
                doc = corpus.get(c["doc_id"], {})
                d_text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
                
                # Apply mxbai instruction fix for latency too, because token count matters
                if "mxbai" in args.model.lower():
                    q_text = f"Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: {queries[qid]}"
                else:
                    q_text = queries[qid]
                    
                pairs.append((q_text, d_text))
                
            if pairs:
                all_real_pairs.append(pairs)

        if not all_real_pairs: continue
        results[dataset][stage] = time_model(all_real_pairs)

os.makedirs("results/metrics", exist_ok=True)
csv_path = f"results/metrics/latency_{model_nickname}.csv"

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Dataset", f"{model_nickname}_BM25_Avg(ms)", f"{model_nickname}_BM25_P95(ms)", f"{model_nickname}_SPLADE_Avg(ms)", f"{model_nickname}_SPLADE_P95(ms)"])
    
    for ds in datasets:
        if not results[ds]: continue
        r = results[ds]
        bm25_avg, bm25_p95 = r.get("bm25", (0.0, 0.0))
        splade_avg, splade_p95 = r.get("splade", (0.0, 0.0))
        writer.writerow([ds, f"{bm25_avg:.2f}", f"{bm25_p95:.2f}", f"{splade_avg:.2f}", f"{splade_p95:.2f}"])
        
print(f"\n>> Saved standalone latency stats to {csv_path}")