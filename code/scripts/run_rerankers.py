import argparse
import json
import os
import csv
import time
import torch
import numpy as np
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader

ALL_DATASETS = ["nq", "scifact", "trec-covid", "nfcorpus", "fiqa", "arguana"]

parser = argparse.ArgumentParser()
parser.add_argument("--first-stage", type=str, choices=["bm25", "splade"], required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                    help="Datasets to rerank. Defaults to all 6 BEIR subsets.")
parser.add_argument("--skip-latency", action="store_true", help="Skip latency measurement")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

datasets = args.datasets
model_nickname = args.model.split("/")[-1]
out_dir = f"results/reranked_{args.first_stage}/{model_nickname}"
os.makedirs(out_dir, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Model loading: each architecture needs its own loader
# ─────────────────────────────────────────────────────────────

MODEL_TYPE = "cross_encoder"  # default

if "mxbai" in args.model.lower() and "v2" in args.model.lower():
    # ── mxbai-rerank v2: Qwen-based, loaded via raw transformers ──
    # The mxbai_rerank package is broken on transformers>=5.x,
    # so we use the repackaged SequenceClassification variant directly.
    MODEL_TYPE = "mxbai_v2"
    print(f"Loading mxbai-rerank-v2 via raw transformers...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    MXBAI_SEQ_MODEL = "michaelfeil/mxbai-rerank-base-v2-seq"
    mxbai_tokenizer = AutoTokenizer.from_pretrained(MXBAI_SEQ_MODEL)
    mxbai_model = AutoModelForSequenceClassification.from_pretrained(
        MXBAI_SEQ_MODEL, torch_dtype=torch.float16
    ).to(DEVICE).eval()

    if mxbai_tokenizer.pad_token is None:
        mxbai_tokenizer.pad_token = mxbai_tokenizer.eos_token

    MXBAI_BATCH = 32

    def mxbai_v2_format(query, document):
        """Exact prompt template from the reference implementation.
        See: https://huggingface.co/michaelfeil/mxbai-rerank-base-v2-seq
        """
        system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        return (
            f"<|endoftext|><|im_start|>system\n{system_prompt}\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"query: {query} \n"
            f"document: {document} \n"
            f"You are a search relevance expert who evaluates how well documents match search queries. "
            f"For each query-document pair, carefully analyze the semantic relationship between them, "
            f"then provide your binary relevance judgment (0 for not relevant, 1 for relevant).\n"
            f"Relevance:<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

elif "monot5" in args.model.lower():
    # ── MonoT5: pointwise seq2seq ──
    MODEL_TYPE = "monot5"
    print(f"Loading MonoT5 via transformers...")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    monot5_tokenizer = AutoTokenizer.from_pretrained(args.model)
    monot5_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(DEVICE).eval()

    TRUE_ID = monot5_tokenizer.encode("true", add_special_tokens=False)[0]
    FALSE_ID = monot5_tokenizer.encode("false", add_special_tokens=False)[0]
    MONOT5_BATCH = 64

    def monot5_score_batch(query, doc_texts):
        """Score a batch of documents against a single query."""
        prompts = [f"Query: {query} Document: {d} Relevant:" for d in doc_texts]
        inputs = monot5_tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(DEVICE)

        decoder_input_ids = torch.full(
            (len(prompts), 1),
            monot5_model.config.decoder_start_token_id,
            dtype=torch.long, device=DEVICE,
        )

        with torch.no_grad():
            outputs = monot5_model(**inputs, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits[:, 0, :]
            scores = (logits[:, TRUE_ID] - logits[:, FALSE_ID]).cpu().numpy()

        return scores

else:
    # ── Standard cross-encoders: MiniLM, bge-reranker, etc. ──
    from sentence_transformers import CrossEncoder

    if "MiniLM" in args.model:
        BATCH_SIZE = 1024
        model_kwargs = {}
    else:
        BATCH_SIZE = 128
        model_kwargs = {"torch_dtype": torch.float16}

    print(f"Loading {args.model} as CrossEncoder onto {DEVICE}...")
    reranker = CrossEncoder(args.model, device=DEVICE, model_kwargs=model_kwargs)


# ─────────────────────────────────────────────────────────────
# Scoring function (dispatches by MODEL_TYPE, returns scores)
# ─────────────────────────────────────────────────────────────

def score_query(query_text, doc_texts):
    """Score all docs for a query. Returns np.array of scores."""
    if MODEL_TYPE == "mxbai_v2":
        all_scores = []
        for i in range(0, len(doc_texts), MXBAI_BATCH):
            batch_docs = doc_texts[i:i + MXBAI_BATCH]
            prompts = [mxbai_v2_format(query_text, d) for d in batch_docs]
            inputs = mxbai_tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=8192,
            ).to(DEVICE)
            with torch.no_grad():
                logits = mxbai_model(**inputs).logits.squeeze(-1)
            all_scores.extend(logits.cpu().float().numpy().tolist())
        return np.array(all_scores)

    elif MODEL_TYPE == "monot5":
        all_scores = []
        for i in range(0, len(doc_texts), MONOT5_BATCH):
            batch = doc_texts[i:i + MONOT5_BATCH]
            all_scores.extend(monot5_score_batch(query_text, batch).tolist())
        return np.array(all_scores)

    else:
        pairs = [(query_text, d) for d in doc_texts]
        return reranker.predict(pairs, batch_size=BATCH_SIZE)


# ─────────────────────────────────────────────────────────────
# Re-ranking loop with integrated latency profiling
# ─────────────────────────────────────────────────────────────

MEASURE_LATENCY = not args.skip_latency
all_latencies = {}  # dataset -> list of per-query latencies (seconds)

for dataset in datasets:
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

    print(f"Re-ranking {dataset} with {model_nickname}...")
    reranked_results = {}
    query_latencies = []

    for qi, (qid, cand_list) in enumerate(tqdm(candidates.items(), desc=dataset)):
        query_text = queries[qid]
        doc_ids = []
        doc_texts = []

        for cand in cand_list:
            doc_id = cand["doc_id"]
            if doc_id in corpus:
                d_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                doc_ids.append(doc_id)
                doc_texts.append(d_text)

        if not doc_texts:
            reranked_results[qid] = []
            continue

        # ── Score with optional latency measurement ──
        if MEASURE_LATENCY and qi >= 5:
            # Skip first 5 queries as implicit warmup
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            scores = score_query(query_text, doc_texts)

            if DEVICE == "cuda":
                torch.cuda.synchronize()
            query_latencies.append(time.perf_counter() - t0)
        else:
            scores = score_query(query_text, doc_texts)

        # ── Build output ──
        valid_cands = []
        for did, score in zip(doc_ids, scores):
            valid_cands.append({"doc_id": did, "score": float(score)})

        valid_cands.sort(key=lambda x: x["score"], reverse=True)
        for rank, cand in enumerate(valid_cands):
            cand["rank"] = rank + 1

        reranked_results[qid] = valid_cands

    # Save reranked results
    with open(out_path, 'w') as f:
        json.dump(reranked_results, f, indent=2)

    if query_latencies:
        all_latencies[dataset] = query_latencies

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────
# Save latency stats
# ─────────────────────────────────────────────────────────────

if MEASURE_LATENCY and all_latencies:
    os.makedirs("results/metrics", exist_ok=True)
    csv_path = f"results/metrics/latency_{model_nickname}_{args.first_stage}.csv"

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "NumQueries", "Avg_ms", "Median_ms", "P95_ms", "Std_ms"])
        for ds in datasets:
            if ds not in all_latencies:
                continue
            lats = all_latencies[ds]
            writer.writerow([
                ds,
                len(lats),
                f"{np.mean(lats) * 1000:.2f}",
                f"{np.median(lats) * 1000:.2f}",
                f"{np.percentile(lats, 95) * 1000:.2f}",
                f"{np.std(lats) * 1000:.2f}",
            ])

    # Print combined average across all datasets (method-level latency)
    all_lats = [l for lats in all_latencies.values() for l in lats]
    if all_lats:
        print(f"\n>> {model_nickname} on {args.first_stage}: "
              f"avg={np.mean(all_lats)*1000:.1f}ms, "
              f"median={np.median(all_lats)*1000:.1f}ms, "
              f"p95={np.percentile(all_lats, 95)*1000:.1f}ms "
              f"({len(all_lats)} queries)")
    print(f">> Per-dataset latency saved to {csv_path}")

print(f"\n=== DONE: {model_nickname} on {args.first_stage} ===")
