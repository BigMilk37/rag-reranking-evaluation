import json
import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from beir.datasets.data_loader import GenericDataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--first-stage", type=str, choices=["bm25", "splade"], required=True)
args = parser.parse_args()

datasets = ["nq", "scifact", "trec-covid", "nfcorpus", "fiqa", "arguana"]
out_dir = f"results/reranked_{args.first_stage}/lambdamart"
os.makedirs(out_dir, exist_ok=True)

with open("models/lambdamart/model.pkl", "rb") as f:
    ranker = pickle.load(f)
with open("models/lambdamart/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def build_doc_freq(corpus):
    """Build term document-frequency table from corpus.
    Returns (df_dict, n_docs) where df_dict maps term -> doc count."""
    df = defaultdict(int)
    for doc in corpus.values():
        text = (doc.get("title", "") + " " + doc.get("text", "")).lower()
        for term in set(text.split()):
            df[term] += 1
    return df, len(corpus)

def idf(term, df, n_docs):
    """Compute smoothed IDF score for a term."""
    return np.log((n_docs + 1) / (df.get(term, 0) + 1)) + 1.0

def tfidf_vec(text, df, n_docs):
    """Return a sparse TF-IDF vector {term: weight} for the given text."""
    terms = text.lower().split()
    if not terms: return {}
    tf = defaultdict(float)
    for t in terms: tf[t] += 1.0 / len(terms)
    return {t: v * idf(t, df, n_docs) for t, v in tf.items()}

def cosine(v1, v2):
    """Cosine similarity between two sparse TF-IDF vectors."""
    if not v1 or not v2: return 0.0
    dot = sum(v1.get(t, 0.0) * s for t, s in v2.items())
    n1 = np.sqrt(sum(x * x for x in v1.values())) + 1e-9
    n2 = np.sqrt(sum(x * x for x in v2.values())) + 1e-9
    return dot / (n1 * n2)

def extract_features(q_terms, q_tfidf, passage_text, bm25_rank, splade_score, splade_rank, splade_max_score, df, n_docs):
    """Extract the 15-dimensional feature vector used by LambdaMART.

    Features: BM25/SPLADE scores and ranks, reciprocal ranks, rank diff,
    query coverage, exact match count, IDF-weighted overlap, TF-IDF cosine,
    doc length, query length.
    """
    d_terms = set(passage_text.lower().split())
    coverage = sum(1 for t in q_terms if t in d_terms) / (len(q_terms) + 1e-9)
    exact_ct = float(sum(1 for t in q_terms if t in d_terms))
    idf_ovlp = sum(idf(t, df, n_docs) for t in q_terms if t in d_terms)
    cos_sim = cosine(q_tfidf, tfidf_vec(passage_text, df, n_docs))

    # Strict proxy mirroring T5 training
    bm25_score = 1.0 / bm25_rank
    bm25_norm = bm25_score / (1.0 / 1 + 1e-9)
    splade_norm = splade_score / (splade_max_score + 1e-9)

    return [
        bm25_score, bm25_norm, splade_score, splade_norm,
        float(bm25_rank), float(splade_rank), float(bm25_rank - splade_rank),
        1.0 / bm25_rank, 1.0 / splade_rank,
        coverage, exact_ct, idf_ovlp, cos_sim,
        float(len(passage_text.split())), float(len(q_terms))
    ]

for dataset in datasets:
    print(f"\nProcessing {dataset}...")
    corpus, queries, _ = GenericDataLoader(f"datasets/{dataset}").load(split="test")
    df, n_docs = build_doc_freq(corpus)
    
    with open(f"results/candidates_bm25/{dataset}.json") as f: bm25_run = json.load(f)
    with open(f"results/candidates_splade/{dataset}.json") as f: splade_run = json.load(f)
    
    target_run = bm25_run if args.first_stage == "bm25" else splade_run
    
    bm25_lut = {q: {c["doc_id"]: c["rank"] for c in cands} for q, cands in bm25_run.items()}
    splade_lut = {q: {c["doc_id"]: (c["score"], c["rank"]) for c in cands} for q, cands in splade_run.items()}
    
    reranked_results = {}
    for qid, cands in tqdm(target_run.items(), desc=f"LambdaMART {dataset}"):
        qtext = queries[qid]
        q_terms = qtext.lower().split()
        q_tfidf = tfidf_vec(qtext, df, n_docs)
        
        b_cands = bm25_lut.get(qid, {})
        s_cands = splade_lut.get(qid, {})
        splade_max = max((s for s, _ in s_cands.values()), default=1e-9)
        
        features_list = []
        valid_cands = []
        
        for cand in cands:
            doc_id = cand["doc_id"]
            if doc_id not in corpus: continue
            passage_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
            
            b_rank = b_cands.get(doc_id, 101)
            s_score, s_rank = s_cands.get(doc_id, (0.0, 101))
            
            feats = extract_features(q_terms, q_tfidf, passage_text, b_rank, s_score, s_rank, splade_max, df, n_docs)
            features_list.append(feats)
            valid_cands.append(cand)
            
        if not features_list:
            reranked_results[qid] = []
            continue
            
        X = scaler.transform(np.array(features_list, dtype=np.float32))
        scores = ranker.predict(X)
        
        for cand, score in zip(valid_cands, scores):
            cand["score"] = float(score)
            
        valid_cands.sort(key=lambda x: x["score"], reverse=True)
        for rank, cand in enumerate(valid_cands):
            cand["rank"] = rank + 1
            
        reranked_results[qid] = valid_cands
        
    with open(f"{out_dir}/{dataset}.json", 'w') as f:
        json.dump(reranked_results, f, indent=2)