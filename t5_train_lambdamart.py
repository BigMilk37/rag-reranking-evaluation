"""
t5_train_lambdamart.py
======================
Trains a LambdaMART ranker on MS MARCO passage ranking data.
BEIR datasets are never touched here — they remain fully zero-shot.

Inputs (from download_msmarco.py + T1/T3/T4 for MS MARCO):
  datasets/msmarco/collection.tsv       -- pid \t text
  datasets/msmarco/queries.train.tsv    -- qid \t text
  datasets/msmarco/qrels.train.tsv      -- qid 0 pid 1
  datasets/msmarco/top1000.train.tsv    -- qid pid query passage (BM25 top-1000 candidates)

Outputs:
  models/lambdamart/model.pkl
  models/lambdamart/scaler.pkl

How training candidates are built:
  - BM25 scores: re-scored from the official top1000.train.tsv BM25 candidates.
    The file already has BM25-retrieved passages so we don't need to run Pyserini
    over all 8.8M passages for every train query.
  - SPLADE scores: loaded from indexes/splade/msmarco_corpus.json if it exists.
    If not, SPLADE features are zeroed out and a warning is printed.
    (You can encode the MS MARCO corpus with splade_encode.py pointed at msmarco first.)

Usage:
    python t5_train_lambdamart.py
    python t5_train_lambdamart.py --n-queries 100000   # use more training queries
    python t5_train_lambdamart.py --splade-index indexes/splade/msmarco_corpus.json
"""

import json
import pickle
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%X")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

MSMARCO_DIR    = Path("datasets/msmarco")
MODELS_DIR     = Path("models/lambdamart")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TOP_K          = 100   # candidates per query (slice from top-1000)
DEFAULT_N_Q    = 50_000  # train queries to sample (full set is ~500k, adjust freely)

FEATURE_NAMES = [
    "bm25_score",
    "bm25_normalised",       # BM25 / max(BM25) in candidate list
    "splade_score",
    "splade_normalised",     # SPLADE / max(SPLADE) in candidate list
    "bm25_rank",             # rank in BM25 top-1000 (1 = best)
    "splade_rank",           # rank in SPLADE list   (1 = best, TOP_K+1 if absent)
    "rank_diff",             # bm25_rank - splade_rank
    "reciprocal_rank_bm25",
    "reciprocal_rank_splade",
    "query_coverage",        # fraction of query unigrams found in passage
    "exact_match_count",
    "idf_weighted_overlap",
    "tf_idf_cosine",
    "doc_length",
    "query_length",
]

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MS MARCO FILES
# ══════════════════════════════════════════════════════════════════════════════

def load_collection(path: Path) -> dict:
    """Returns {pid: passage_text}"""
    log.info("Loading MS MARCO collection…")
    collection = {}
    with open(path) as f:
        for line in tqdm(f, desc="collection.tsv"):
            pid, text = line.rstrip("\n").split("\t", 1)
            collection[pid] = text
    log.info(f"Loaded {len(collection):,} passages.")
    return collection


def load_queries(path: Path) -> dict:
    """Returns {qid: query_text}"""
    queries = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    log.info(f"Loaded {len(queries):,} train queries.")
    return queries


def load_qrels(path: Path) -> dict:
    """Returns {qid: {pid: relevance}}  — MS MARCO has binary relevance (0/1)."""
    qrels: dict[str, dict] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                qid, _, pid, rel = parts[0], parts[1], parts[2], int(parts[3])
                qrels[qid][pid] = rel
    log.info(f"Loaded qrels for {len(qrels):,} queries.")
    return qrels


def load_top1000(path: Path, valid_qids: set, n_queries: int) -> dict:
    """
    Parses top1000.train.tsv: qid \t pid \t query \t passage
    Returns {qid: [(pid, bm25_rank)]}  — rank inferred from file order.
    Only loads rows for valid_qids (those with at least one positive in qrels).
    Stops once n_queries unique queries are collected.
    """
    log.info(f"Loading top-1000 candidates (sampling {n_queries:,} queries)…")
    top1000: dict[str, list] = defaultdict(list)
    rank_counter: dict[str, int] = defaultdict(int)

    with open(path) as f:
        for line in tqdm(f, desc="top1000.train.tsv"):
            parts = line.rstrip("\n").split("\t", 3)
            if len(parts) < 2:
                continue
            qid, pid = parts[0], parts[1]

            if qid not in valid_qids:
                continue
            if qid not in top1000 and len(top1000) >= n_queries:
                continue  # already have enough queries

            rank_counter[qid] += 1
            top1000[qid].append((pid, rank_counter[qid]))

    log.info(f"Loaded candidates for {len(top1000):,} queries.")
    return top1000


def load_splade_index(path: Path) -> dict | None:
    """
    Load a SPLADE corpus index produced by splade_encode.py.
    Returns {pid: {token_id_str: weight}} or None if path doesn't exist.
    """
    if not path.exists():
        log.warning(f"SPLADE index not found at {path}. SPLADE features will be zero.")
        return None
    log.info(f"Loading SPLADE index from {path}…")
    with open(path) as f:
        index = json.load(f)
    log.info(f"Loaded SPLADE vectors for {len(index):,} passages.")
    return index


# ══════════════════════════════════════════════════════════════════════════════
# TEXT / FEATURE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def build_doc_freq(collection: dict, sample_size: int = 200_000) -> tuple:
    """
    Build term document-frequency over a sample of the collection.
    Using a sample keeps memory sane; IDF estimates are still good.
    """
    log.info(f"Building doc-freq over {sample_size:,} sampled passages…")
    df: dict[str, int] = defaultdict(int)
    pids = list(collection.keys())
    rng  = np.random.default_rng(42)
    sample = rng.choice(pids, size=min(sample_size, len(pids)), replace=False)
    for pid in tqdm(sample, desc="doc-freq"):
        for term in set(collection[pid].lower().split()):
            df[term] += 1
    return df, len(sample)


def idf(term: str, df: dict, n_docs: int) -> float:
    return np.log((n_docs + 1) / (df.get(term, 0) + 1)) + 1.0


def tfidf_vec(text: str, df: dict, n_docs: int) -> dict:
    terms = text.lower().split()
    if not terms:
        return {}
    tf: dict[str, float] = defaultdict(float)
    for t in terms:
        tf[t] += 1.0 / len(terms)
    return {t: v * idf(t, df, n_docs) for t, v in tf.items()}


def cosine(v1: dict, v2: dict) -> float:
    if not v1 or not v2:
        return 0.0
    dot = sum(v1.get(t, 0.0) * s for t, s in v2.items())
    n1  = np.sqrt(sum(x * x for x in v1.values())) + 1e-9
    n2  = np.sqrt(sum(x * x for x in v2.values())) + 1e-9
    return dot / (n1 * n2)


def sparse_dot(q_vec: dict, doc_sparse: dict) -> float:
    return sum(doc_sparse.get(k, 0.0) * v for k, v in q_vec.items())


def extract_features(
    q_terms: list, q_tfidf: dict,
    passage_text: str,
    bm25_rank: int, bm25_max_rank: int,
    splade_score: float, splade_rank: int,
    splade_max_score: float,
    df: dict, n_docs: int,
) -> list:
    d_terms  = set(passage_text.lower().split())
    coverage = sum(1 for t in q_terms if t in d_terms) / (len(q_terms) + 1e-9)
    exact_ct = float(sum(1 for t in q_terms if t in d_terms))
    idf_ovlp = sum(idf(t, df, n_docs) for t in q_terms if t in d_terms)
    cos_sim  = cosine(q_tfidf, tfidf_vec(passage_text, df, n_docs))

    # BM25: we don't have raw scores from top1000.train.tsv, only rank.
    # Use reciprocal-of-rank as a proxy score; normalise by list size.
    bm25_score     = 1.0 / bm25_rank
    bm25_norm      = bm25_score / (1.0 / 1 + 1e-9)          # max possible = 1/1
    splade_norm    = splade_score / (splade_max_score + 1e-9)

    return [
        bm25_score,
        bm25_norm,
        splade_score,
        splade_norm,
        float(bm25_rank),
        float(splade_rank),
        float(bm25_rank - splade_rank),
        1.0 / bm25_rank,
        1.0 / splade_rank,
        coverage,
        exact_ct,
        idf_ovlp,
        cos_sim,
        float(len(passage_text.split())),
        float(len(q_terms)),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# SPLADE QUERY ENCODING  (only used if SPLADE index is available)
# ══════════════════════════════════════════════════════════════════════════════

def encode_queries_splade(query_texts: list[str]) -> list[dict]:
    """
    Encode a list of query strings with SPLADE-v3.
    Returns list of sparse dicts {token_id_int: weight}.
    """
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    BATCH = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Encoding {len(query_texts):,} queries with SPLADE on {device}…")

    tokenizer = AutoTokenizer.from_pretrained("naver/splade-v3")
    model     = AutoModelForMaskedLM.from_pretrained("naver/splade-v3").to(device).eval()

    result = []
    with torch.no_grad():
        for i in tqdm(range(0, len(query_texts), BATCH), desc="SPLADE queries"):
            batch = query_texts[i : i + BATCH]
            tokens = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=256).to(device)
            output = model(**tokens)
            vecs = torch.max(
                torch.log(1 + torch.relu(output.logits))
                * tokens.attention_mask.unsqueeze(-1),
                dim=1,
            )[0]
            for vec in vecs:
                nz_idx = vec.nonzero().squeeze().cpu().tolist()
                if isinstance(nz_idx, int):
                    nz_idx = [nz_idx]
                weights = vec[nz_idx].cpu().tolist()
                if not isinstance(weights, list):
                    weights = [weights]
                result.append(dict(zip(nz_idx, weights)))

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train(n_queries: int, splade_index_path: Path):
    model_out  = MODELS_DIR / "model.pkl"
    scaler_out = MODELS_DIR / "scaler.pkl"

    if model_out.exists():
        log.info("LambdaMART model already exists, skipping.")
        return

    # ── load MS MARCO files ───────────────────────────────────────────────────
    collection = load_collection(MSMARCO_DIR / "collection.tsv")
    queries    = load_queries(MSMARCO_DIR / "queries.train.tsv")
    qrels      = load_qrels(MSMARCO_DIR / "qrels.train.tsv")

    # only use queries that have at least one positive passage
    valid_qids = set(qid for qid in queries if qrels.get(qid))
    log.info(f"{len(valid_qids):,} queries have at least one positive.")

    top1000 = load_top1000(
        MSMARCO_DIR / "top1000.train.tsv", valid_qids, n_queries
    )

    # ── optional SPLADE index ─────────────────────────────────────────────────
    splade_index = load_splade_index(splade_index_path)

    # ── encode queries with SPLADE (if index available) ───────────────────────
    splade_q_vecs: dict[str, dict] = {}
    if splade_index is not None:
        qids_to_encode = list(top1000.keys())
        texts          = [queries[qid] for qid in qids_to_encode]
        vecs           = encode_queries_splade(texts)
        splade_q_vecs  = dict(zip(qids_to_encode, vecs))

    # ── doc-freq for text features ────────────────────────────────────────────
    df, n_docs = build_doc_freq(collection)

    # ── feature extraction ────────────────────────────────────────────────────
    log.info("Extracting features…")
    X_all, y_all, groups_all = [], [], []

    for qid, cands in tqdm(top1000.items(), desc="Queries"):
        qtext    = queries[qid]
        q_terms  = qtext.lower().split()
        q_tfidf  = tfidf_vec(qtext, df, n_docs)
        q_splade = splade_q_vecs.get(qid, {})
        rel_map  = qrels.get(qid, {})

        # SPLADE scores for this query's candidates
        splade_scores: dict[str, float] = {}
        if splade_index is not None:
            for pid, _ in cands:
                if pid in splade_index:
                    splade_scores[pid] = sparse_dot(q_splade, splade_index[pid])

        # build SPLADE rank from scores
        splade_ranked = sorted(splade_scores.items(), key=lambda x: x[1], reverse=True)
        splade_rank_map = {pid: r + 1 for r, (pid, _) in enumerate(splade_ranked)}
        splade_max_score = splade_ranked[0][1] if splade_ranked else 1e-9

        rows = []
        for bm25_rank, (pid, _) in enumerate(cands[:TOP_K], start=1):
            if pid not in collection:
                continue
            passage_text  = collection[pid]
            splade_score  = splade_scores.get(pid, 0.0)
            splade_rank   = splade_rank_map.get(pid, TOP_K + 1)

            feats = extract_features(
                q_terms, q_tfidf,
                passage_text,
                bm25_rank, TOP_K,
                splade_score, splade_rank, splade_max_score,
                df, n_docs,
            )
            rows.append((feats, float(rel_map.get(pid, 0))))

        if not rows:
            continue

        X_all.extend(f for f, _ in rows)
        y_all.extend(r for _, r in rows)
        groups_all.append(len(rows))

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)
    log.info(f"Training set: {X.shape[0]:,} (query, passage) pairs, {len(groups_all):,} queries")

    scaler = StandardScaler()
    X      = scaler.fit_transform(X)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-queries", type=int, default=DEFAULT_N_Q,
        help=f"Number of MS MARCO train queries to use (default: {DEFAULT_N_Q:,})",
    )
    parser.add_argument(
        "--splade-index",
        type=Path,
        default=Path("indexes/splade/msmarco_corpus.json"),
        help="Path to SPLADE corpus index for MS MARCO (optional but recommended)",
    )
    args = parser.parse_args()
    train(n_queries=args.n_queries, splade_index_path=args.splade_index)
