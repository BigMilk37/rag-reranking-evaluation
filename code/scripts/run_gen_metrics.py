import os
import json
import csv
import argparse
import torch
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from beir.datasets.data_loader import GenericDataLoader

ALL_DATASETS = ["nq", "scifact", "trec-covid", "nfcorpus", "fiqa", "arguana"]
ALL_STAGES = ["bm25", "splade"]

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                    help="Datasets to score. Defaults to all 6 BEIR subsets.")
parser.add_argument("--first-stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES,
                    help="First-stage retrievers to score.")
parser.add_argument("--rerankers", nargs="+", default=None,
                    help="Reranker nicknames to score. Defaults to auto-discover from results/.")
args = parser.parse_args()

def discover_rerankers(stages):
    found = set()
    for stage in stages:
        d = f"results/generated_{stage}"
        if os.path.isdir(d):
            for name in os.listdir(d):
                if os.path.isdir(os.path.join(d, name)):
                    found.add(name)
    return sorted(found)

datasets = args.datasets
first_stages = args.first_stages
reranker_nicknames = args.rerankers if args.rerankers else discover_rerankers(first_stages)
models = ["baseline"] + reranker_nicknames

print("=== PHASE 4: SCORING (ROUGE & BERTScore) ===")

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
os.makedirs("results/metrics", exist_ok=True)


for stage in first_stages:
    csv_path = f"results/metrics/generation_scores_{stage}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Model", "ROUGE-L", "BERTScore_F1"])
    
    for dataset in datasets:
        data_path = f"datasets/{dataset}"
        if not os.path.exists(data_path): continue
            
        corpus, _, qrels = GenericDataLoader(data_path).load(split="test")
        
        # Build gold references from actual relevant documents (cap at 3 to prevent OOM/dilution)
        references = {}
        for qid, rel_docs in qrels.items():
            gold_texts = []
            # Sort docs by relevance score descending to prioritize key passages
            sorted_rel_docs = sorted(rel_docs.items(), key=lambda x: x[1], reverse=True)
            for doc_id, rel in sorted_rel_docs:
                if rel > 0 and doc_id in corpus:
                    doc = corpus[doc_id]
                    gold_texts.append((doc.get("title", "") + " " + doc.get("text", "")).strip())
            references[qid] = " ".join(gold_texts[:3])

        for m_name in models:
            gen_path = f"results/generated_{stage}/{m_name}/{dataset}.json"
            if not os.path.exists(gen_path): continue
                
            with open(gen_path, 'r') as f:
                generations = json.load(f)
                
            cands, refs = [], []
            rouge_scores = []
            
            for qid, data in generations.items():
                if qid in references and references[qid].strip():
                    ans = data["answer"]
                    ref = references[qid]
                    cands.append(ans)
                    refs.append(ref)
                    rouge_scores.append(scorer.score(ref, ans)['rougeL'].fmeasure)
                    
            if not cands: continue
                
            print(f"Computing BERTScore for {dataset} ({stage} -> {m_name})...")
            # Deberta-xlarge is the standard for BERTScore, it will download automatically
            P, R, F1 = bert_score(cands, refs, lang="en", verbose=False, device="cuda" if torch.cuda.is_available() else "cpu")
            
            avg_rouge = np.mean(rouge_scores)
            avg_bert = F1.mean().item()
            
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([dataset, m_name, f"{avg_rouge:.4f}", f"{avg_bert:.4f}"])

    print(f">> Saved generation scores to {csv_path}")

print("\n=== SCORING COMPLETE ===")