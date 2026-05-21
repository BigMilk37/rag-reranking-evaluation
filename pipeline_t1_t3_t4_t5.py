"""
Pipeline Tasks T1, T3, T4, T5
==============================
T1  – BM25 index each BEIR dataset corpus via Pyserini
T3  – BM25 retrieve top-100 for all datasets
T4  – SPLADE retrieve top-100 for all datasets  (reads indexes built by splade_encode.py)
T5  – Extract features + train LambdaMART on BEIR datasets

Conventions match the existing codebase:
  - Indexes   : indexes/splade/{dataset}_corpus.json   (written by splade_encode.py)
                indexes/bm25/{dataset}/                (written by T1)
  - Candidates: results/candidates_splade/{dataset}.json
                results/candidates_bm25/{dataset}.json
  - Format    : {"qid": [{"doc_id": ..., "score": ..., "rank": ...}, ...], ...}
  - Models    : models/lambdamart/model.pkl
                models/lambdamart/scaler.pkl

Run order:
    python pipeline_t1_t3_t4_t5.py t1          # build BM25 indexes
    python pipeline_t1_t3_t4_t5.py t3          # BM25 retrieval (after t1)
    python pipeline_t1_t3_t4_t5.py t4          # SPLADE retrieval (after splade_encode.py)
    python pipeline_t1_t3_t4_t5.py t5          # train LambdaMART (after t3 + t4)

    python pipeline_t1_t3_t4_t5.py t1 t3       # chain multiple tasks
    python pipeline_t1_t3_t4_t5.py all         # run everything in order
"""

import os
import json
import subprocess
import pickle
import logging
import argparse
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from beir.datasets.data_loader import GenericDataLoader
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%X")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATASETS   = ["nq", "scifact", "trec-covid", "nfcorpus", "fiqa", "arguana"]
TOP_K      = 100
BATCH_SIZE = 512   # RTX 5090

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASETS_DIR       = Path("datasets")
BM25_INDEX_DIR     = Path("indexes/bm25")
SPLADE_INDEX_DIR   = Path("indexes/splade")
RESULTS_BM25_DIR   = Path("results/candidates_bm25")
RESULTS_SPLADE_DIR = Path("results/candidates_splade")
MODELS_DIR         = Path("models/lambdamart")

for d in [BM25_INDEX_DIR, RESULTS_BM25_DIR, RESULTS_SPLADE_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(name: str):
    corpus, queries, qrels = GenericDataLoader(str(DATASETS_DIR / name)).load(split="test")
    log.info(f"[{name}] corpus={len(corpus):,}  queries={len(queries):,}")
    return corpus, queries, qrels


def save_candidates(candidates: dict, path: Path):
    with open(path, "w") as f:
        json.dump(candidates, f, indent=2)
    log.info(f"Saved {len(candidates):,} queries -> {path}")


def run_cmd(cmd: str, step_name: str):
    log.info(f"STARTING: {step_name}")
    t0 = time.time()
    subprocess.run(cmd, shell=True, check=True)
    log.info(f"FINISHED: {step_name} ({time.time() - t0:.1f}s)")


# ══════════════════════════════════════════════════════════════════════════════
# T1 – BUILD BM25 INDEXES (one per dataset)
# ══════════════════════════════════════════════════════════════════════════════

def _corpus_to_pyserini_jsonl(corpus: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for doc_id, doc in corpus.items():
            text = (doc.get("title", "") + " " + doc["text"]).strip()
            f.write(json.dumps({"id": doc_id, "contents": text}) + "\n")


def t1_build_bm25_indexes():
    log.info("=== T1: Building BM25 indexes ===")

    for dataset in DATASETS:
        index_dir = BM25_INDEX_DIR / dataset
        if index_dir.exists() and any(index_dir.iterdir()):
            log.info(f"[{dataset}] BM25 index exists, skipping.")
            continue

        corpus, _, _ = load_dataset(dataset)

        # write Pyserini-format JSONL to a temp dir
        jsonl_dir = Path(f"indexes/bm25_tmp/{dataset}")
        _corpus_to_pyserini_jsonl(corpus, jsonl_dir / "corpus.jsonl")

        run_cmd(
            f"python -m pyserini.index.lucene "
            f"--collection JsonCollection "
            f"--input {jsonl_dir} "
            f"--index {index_dir} "
            f"--generator DefaultLuceneDocumentGenerator "
            f"--threads 16 "
            f"--storeRaw",
            f"BM25 index [{dataset}]",
        )

    log.info("=== T1: All BM25 indexes built ===")


# ══════════════════════════════════════════════════════════════════════════════
# T3 – BM25 RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def t3_bm25_retrieve():
    from pyserini.search.lucene import LuceneSearcher

    log.info("=== T3: BM25 retrieval ===")

    for dataset in DATASETS:
        out_path = RESULTS_BM25_DIR / f"{dataset}.json"
        if out_path.exists():
            log.info(f"[{dataset}] BM25 candidates exist, skipping.")
            continue

        index_dir = BM25_INDEX_DIR / dataset
        if not index_dir.exists():
            log.warning(f"[{dataset}] BM25 index missing — run T1 first. Skipping.")
            continue

        _, queries, _ = load_dataset(dataset)

        searcher = LuceneSearcher(str(index_dir))
        searcher.set_bm25(k1=0.9, b=0.4)

        candidates = {}
        for qid, query_text in tqdm(queries.items(), desc=f"BM25 {dataset}"):
            hits = searcher.search(query_text, k=TOP_K)
            candidates[qid] = [
                {"doc_id": h.docid, "score": float(h.score), "rank": rank + 1}
                for rank, h in enumerate(hits)
            ]

        save_candidates(candidates, out_path)

    log.info("=== T3: BM25 retrieval complete ===")


# ══════════════════════════════════════════════════════════════════════════════
# T4 – SPLADE RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def _encode_queries_splade(queries: dict, model, tokenizer) -> dict:
    """
    Batch-encode queries with the same pooling formula as splade_encode.py:
        max(log(1 + relu(logits)) * attention_mask, dim=1)
    Returns {qid: cpu tensor (vocab_size,)}
    """
    q_ids   = list(queries.keys())
    q_texts = [queries[qid] for qid in q_ids]
    q_vecs  = {}

    with torch.no_grad():
        for i in tqdm(range(0, len(q_texts), BATCH_SIZE), desc="Encoding queries"):
            batch_ids   = q_ids[i : i + BATCH_SIZE]
            batch_texts = q_texts[i : i + BATCH_SIZE]
            tokens = tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            ).to(DEVICE)
            output = model(**tokens)
            vecs = torch.max(
                torch.log(1 + torch.relu(output.logits))
                * tokens.attention_mask.unsqueeze(-1),
                dim=1,
            )[0]
            for qid, vec in zip(batch_ids, vecs):
                q_vecs[qid] = vec.cpu()

    return q_vecs


def t4_splade_retrieve():
    """
    Reads per-dataset corpus indexes from splade_encode.py and scores queries
    against them via exact sparse dot-product using fast PyTorch GPU matrix math.
    """
    log.info(f"=== T4: SPLADE retrieval on {DEVICE} ===")

    VOCAB_SIZE = 30522  # Standard BERT vocab size
    CHUNK_SIZE = 50000  # Will use ~6GB of VRAM per chunk

    log.info("Loading SPLADE model…")
    tokenizer = AutoTokenizer.from_pretrained("naver/splade-v3")
    model     = AutoModelForMaskedLM.from_pretrained("naver/splade-v3").to(DEVICE).eval()

    for dataset in DATASETS:
        out_path   = RESULTS_SPLADE_DIR / f"{dataset}.json"
        index_path = SPLADE_INDEX_DIR / f"{dataset}_corpus.json"

        if out_path.exists():
            log.info(f"[{dataset}] SPLADE candidates exist, skipping.")
            continue

        if not index_path.exists():
            log.warning(f"[{dataset}] SPLADE index missing at {index_path} — run splade_encode.py first. Skipping.")
            continue

        log.info(f"[{dataset}] Loading corpus vectors…")
        with open(index_path) as f:
            corpus_dict = json.load(f)   # {doc_id: {str(token_id): weight}}
        doc_ids = list(corpus_dict.keys())

        _, queries, _ = load_dataset(dataset)
        query_vecs = _encode_queries_splade(queries, model, tokenizer)
        q_ids = list(queries.keys())

        log.info(f"[{dataset}] Building query matrix [{len(q_ids):,}, {VOCAB_SIZE}] on {DEVICE}…")
        # Build query matrix [Num_Queries, Vocab_Size] on DEVICE
        query_tensor = torch.zeros(len(q_ids), VOCAB_SIZE, device=DEVICE)
        for i, qid in enumerate(q_ids):
            query_tensor[i] = query_vecs[qid].to(DEVICE)

        global_top_scores = torch.full((len(q_ids), TOP_K), -float('inf'), device=DEVICE)
        global_top_indices = torch.zeros((len(q_ids), TOP_K), dtype=torch.long, device=DEVICE)

        log.info(f"[{dataset}] Engaging GPU matrix multiplication…")
        for i in tqdm(range(0, len(doc_ids), CHUNK_SIZE), desc=f"SPLADE Matrix Math {dataset}"):
            chunk_ids = doc_ids[i:i+CHUNK_SIZE]
            chunk_tensor = torch.zeros(len(chunk_ids), VOCAB_SIZE, device=DEVICE)

            for j, did in enumerate(chunk_ids):
                for idx, weight in corpus_dict[did].items():
                    idx_int = int(idx)
                    if idx_int < VOCAB_SIZE:
                        chunk_tensor[j, idx_int] = weight

            # Perform high-performance exact sparse dot-product matching
            # [Queries, Vocab] @ [Vocab, Chunk_Docs] -> [Queries, Chunk_Docs]
            scores = torch.matmul(query_tensor, chunk_tensor.T)

            # Merge and keep top K
            combined_scores = torch.cat([global_top_scores, scores], dim=1)
            chunk_global_indices = torch.arange(i, i + len(chunk_ids), device=DEVICE).unsqueeze(0).expand(len(q_ids), -1)
            combined_indices = torch.cat([global_top_indices, chunk_global_indices], dim=1)

            global_top_scores, top_idx = torch.topk(combined_scores, TOP_K, dim=1)
            global_top_indices = torch.gather(combined_indices, 1, top_idx)

        log.info(f"[{dataset}] Formatting candidates…")
        candidates = {}
        for i, qid in enumerate(q_ids):
            top_doc_indices = global_top_indices[i].cpu().tolist()
            top_doc_scores = global_top_scores[i].cpu().tolist()

            candidates[qid] = [
                {"doc_id": doc_ids[doc_idx], "score": float(score), "rank": rank + 1}
                for rank, (doc_idx, score) in enumerate(zip(top_doc_indices, top_doc_scores))
            ]

        save_candidates(candidates, out_path)

    log.info("=== T4: SPLADE retrieval complete ===")


# ══════════════════════════════════════════════════════════════════════════════
# T5 – LAMBDAMART: FEATURE EXTRACTION + TRAINING
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "bm25_score",            # raw BM25 score
    "bm25_normalised",       # BM25 / max(BM25) in candidate list
    "splade_score",          # raw SPLADE dot-product score
    "splade_normalised",     # SPLADE / max(SPLADE) in candidate list
    "bm25_rank",             # rank in BM25 list  (1 = best)
    "splade_rank",           # rank in SPLADE list (1 = best)
    "rank_diff",             # bm25_rank - splade_rank
    "reciprocal_rank_bm25",  # 1 / bm25_rank
    "reciprocal_rank_splade",# 1 / splade_rank
    "query_coverage",        # fraction of query unigrams found in doc
    "exact_match_count",     # raw count of query terms in doc
    "idf_weighted_overlap",  # IDF-weighted term overlap
    "tf_idf_cosine",         # TF-IDF cosine similarity
    "doc_length",            # doc token count (whitespace)
    "query_length",          # query token count
]


def _build_doc_freq(corpus: dict) -> tuple:
    df: dict[str, int] = defaultdict(int)
    for doc in corpus.values():
        text = (doc.get("title", "") + " " + doc.get("text", "")).lower()
        for term in set(text.split()):
            df[term] += 1
    return df, len(corpus)


def _idf(term: str, df: dict, n_docs: int) -> float:
    return np.log((n_docs + 1) / (df.get(term, 0) + 1)) + 1.0


def _tfidf_vec(text: str, df: dict, n_docs: int) -> dict:
    terms = text.lower().split()
    if not terms:
        return {}
    tf: dict[str, float] = defaultdict(float)
    for t in terms:
        tf[t] += 1.0 / len(terms)
    return {t: v * _idf(t, df, n_docs) for t, v in tf.items()}


def _cosine(v1: dict, v2: dict) -> float:
    if not v1 or not v2:
        return 0.0
    dot = sum(v1.get(t, 0.0) * s for t, s in v2.items())
    n1  = np.sqrt(sum(x * x for x in v1.values())) + 1e-9
    n2  = np.sqrt(sum(x * x for x in v2.values())) + 1e-9
    return dot / (n1 * n2)


def _extract_features(
    q_terms: list, doc_text: str,
    bm25_score: float, bm25_rank: int, bm25_max: float,
    splade_score: float, splade_rank: int, splade_max: float,
    df: dict, n_docs: int, q_tfidf: dict,
) -> list:
    d_terms  = set(doc_text.lower().split())
    coverage = sum(1 for t in q_terms if t in d_terms) / (len(q_terms) + 1e-9)
    exact_ct = float(sum(1 for t in q_terms if t in d_terms))
    idf_ovlp = sum(_idf(t, df, n_docs) for t in q_terms if t in d_terms)
    cos_sim  = _cosine(q_tfidf, _tfidf_vec(doc_text, df, n_docs))

    return [
        bm25_score,
        bm25_score   / (bm25_max   + 1e-9),
        splade_score,
        splade_score / (splade_max + 1e-9),
        float(bm25_rank),
        float(splade_rank),
        float(bm25_rank - splade_rank),
        1.0 / bm25_rank,
        1.0 / splade_rank,
        coverage,
        exact_ct,
        idf_ovlp,
        cos_sim,
        float(len(doc_text.split())),
        float(len(q_terms)),
    ]


def t5_lambdamart_train():
    """
    Pool candidates from all 6 BEIR datasets (BM25 + SPLADE runs),
    extract features, and train a LambdaMART ranker.
    Qrels from each dataset's test split serve as relevance labels.
    """
    log.warning("=" * 80)
    log.warning("METHODOLOGICAL WARNING (DATA LEAKAGE DETECTED)")
    log.warning("Training the LambdaMART ranker directly on the BEIR test splits violates")
    log.warning("the zero-shot evaluation protocol described in your report/README.")
    log.warning("To keep evaluation strictly zero-shot, you should train your LambdaMART")
    log.warning("model using MS MARCO only via: 'python t5_train_lambdamart.py'")
    log.warning("=" * 80)

    model_out  = MODELS_DIR / "model.pkl"
    scaler_out = MODELS_DIR / "scaler.pkl"

    if model_out.exists():
        log.info("LambdaMART model already exists, skipping T5.")
        return

    log.info("=== T5: LambdaMART feature extraction + training ===")

    X_all, y_all, groups_all = [], [], []

    for dataset in DATASETS:
        bm25_path   = RESULTS_BM25_DIR   / f"{dataset}.json"
        splade_path = RESULTS_SPLADE_DIR / f"{dataset}.json"

        if not bm25_path.exists() or not splade_path.exists():
            log.warning(f"[{dataset}] Missing candidates — skipping for LTR training.")
            continue

        log.info(f"[{dataset}] Extracting features…")
        corpus, queries, qrels = load_dataset(dataset)
        df, n_docs = _build_doc_freq(corpus)

        with open(bm25_path)   as f: bm25_run   = json.load(f)
        with open(splade_path) as f: splade_run = json.load(f)

        # {qid: {doc_id: (score, rank)}}
        def to_lookup(run):
            return {
                qid: {c["doc_id"]: (c["score"], c["rank"]) for c in cands}
                for qid, cands in run.items()
            }

        bm25_lut   = to_lookup(bm25_run)
        splade_lut = to_lookup(splade_run)

        for qid, qtext in tqdm(queries.items(), desc=f"Features {dataset}"):
            bm25_cands   = bm25_lut.get(qid, {})
            splade_cands = splade_lut.get(qid, {})
            all_doc_ids  = set(bm25_cands) | set(splade_cands)
            if not all_doc_ids:
                continue

            bm25_max   = max((s for s, _ in bm25_cands.values()),   default=1e-9)
            splade_max = max((s for s, _ in splade_cands.values()), default=1e-9)
            q_terms    = qtext.lower().split()
            q_tfidf    = _tfidf_vec(qtext, df, n_docs)
            rel_map    = qrels.get(qid, {})

            rows = []
            for doc_id in all_doc_ids:
                if doc_id not in corpus:
                    continue
                doc      = corpus[doc_id]
                doc_text = (doc.get("title", "") + " " + doc.get("text", "")).strip()

                bm25_score,   bm25_rank   = bm25_cands.get(doc_id,   (0.0, TOP_K + 1))
                splade_score, splade_rank = splade_cands.get(doc_id, (0.0, TOP_K + 1))

                feats = _extract_features(
                    q_terms, doc_text,
                    bm25_score, bm25_rank, bm25_max,
                    splade_score, splade_rank, splade_max,
                    df, n_docs, q_tfidf,
                )
                rows.append((feats, float(rel_map.get(doc_id, 0))))

            if not rows:
                continue

            X_all.extend(f for f, _ in rows)
            y_all.extend(r for _, r in rows)
            groups_all.append(len(rows))

    if not X_all:
        log.error("No training data extracted. Make sure T3 and T4 have run first.")
        return

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)
    log.info(f"Training set: {X.shape[0]:,} (query, doc) pairs, {len(groups_all):,} queries")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    log.info("Training LambdaMART…")
    ranker = lgb.LGBMRanker(
        objective         = "lambdarank",
        metric            = "ndcg",
        ndcg_eval_at      = [10],
        n_estimators      = 500,
        learning_rate     = 0.05,
        num_leaves        = 63,
        min_child_samples = 20,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        random_state      = 42,
        n_jobs            = -1,
        verbose           = 50,
    )
    ranker.fit(X, y, group=groups_all, feature_name=FEATURE_NAMES)

    with open(model_out,  "wb") as f: pickle.dump(ranker, f)
    with open(scaler_out, "wb") as f: pickle.dump(scaler, f)
    log.info(f"Saved model  -> {model_out}")
    log.info(f"Saved scaler -> {scaler_out}")

    imps = sorted(zip(FEATURE_NAMES, ranker.feature_importances_),
                  key=lambda x: x[1], reverse=True)
    log.info("Feature importances:")
    for name, imp in imps:
        log.info(f"  {name:<28} {imp:>6.0f}")

    log.info("=== T5: LambdaMART training complete ===")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

TASK_MAP = {
    "t1": t1_build_bm25_indexes,
    "t3": t3_bm25_retrieve,
    "t4": t4_splade_retrieve,
    "t5": t5_lambdamart_train,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline tasks T1, T3, T4, T5")
    parser.add_argument(
        "tasks", nargs="+",
        choices=[*TASK_MAP.keys(), "all"],
        help="Tasks to run in order, or 'all'",
    )
    args = parser.parse_args()

    task_list = list(TASK_MAP.keys()) if "all" in args.tasks else args.tasks
    for task in task_list:
        TASK_MAP[task]()
