import os
import json
import csv
import pytrec_eval
import numpy as np
from scipy.stats import ttest_rel
from beir.datasets.data_loader import GenericDataLoader

import argparse

ALL_DATASETS = ["nq", "scifact", "trec-covid", "nfcorpus", "fiqa", "arguana"]
ALL_STAGES = ["bm25", "splade"]

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                    help="Datasets to evaluate. Defaults to all 6 BEIR subsets.")
parser.add_argument("--first-stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES,
                    help="First-stage retrievers to evaluate.")
parser.add_argument("--rerankers", nargs="+", default=None,
                    help="Reranker nicknames to evaluate. Defaults to auto-discover from results/.")
args = parser.parse_args()

datasets = args.datasets
first_stages = args.first_stages

def discover_rerankers(stages):
    """Scan results/reranked_*/ dirs to find all reranker nicknames that exist."""
    found = set()
    for stage in stages:
        d = f"results/reranked_{stage}"
        if os.path.isdir(d):
            for name in os.listdir(d):
                if os.path.isdir(os.path.join(d, name)):
                    found.add(name)
    return sorted(found)

rerankers = args.rerankers if args.rerankers else discover_rerankers(first_stages)


# pytrec_eval strictly requires these exact names
metrics = {'ndcg_cut_10', 'recip_rank', 'map_cut_100', 'recall_10', 'P_1'}

print("=== PHASE 2: METRICS & SIGNIFICANCE ===")
print("Crunching numbers... this might actually take a minute.")

# Store results: stage -> dataset -> model -> {metrics}
summary = {stage: {ds: {} for ds in datasets} for stage in first_stages}

for dataset in datasets:
    data_path = f"datasets/{dataset}"
    if not os.path.exists(data_path):
        continue
        
    _, queries, qrels = GenericDataLoader(data_path).load(split="test")
    
    # Convert BEIR qrels to pytrec_eval format
    pe_qrels = {str(q): {str(d): int(rel) for d, rel in ddict.items()} for q, ddict in qrels.items()}
    evaluator = pytrec_eval.RelevanceEvaluator(pe_qrels, metrics)
    
    for stage in first_stages:
        cand_path = f"results/candidates_{stage}/{dataset}.json"
        if not os.path.exists(cand_path):
            continue
            
        with open(cand_path) as f:
            cands = json.load(f)
            
        run = {str(q): {str(c["doc_id"]): float(c["score"]) for c in clist} for q, clist in cands.items()}
        base_results = evaluator.evaluate(run)
        
        avg_ndcg = np.mean([v.get('ndcg_cut_10', 0) for v in base_results.values()])
        avg_mrr = np.mean([v.get('recip_rank', 0) for v in base_results.values()])
        base_ndcg_list = [base_results.get(str(q), {}).get('ndcg_cut_10', 0) for q in queries.keys()]
        
        summary[stage][dataset]['Baseline'] = {'ndcg': avg_ndcg, 'mrr': avg_mrr, 'pval': 1.0, 'sig': ''}
        
        for reranker in rerankers:
            rr_path = f"results/reranked_{stage}/{reranker}/{dataset}.json"
            if not os.path.exists(rr_path):
                continue
                
            with open(rr_path) as f:
                rr_cands = json.load(f)
                
            rr_run = {str(q): {str(c["doc_id"]): float(c["score"]) for c in clist} for q, clist in rr_cands.items()}
            rr_results = evaluator.evaluate(rr_run)
            
            rr_avg_ndcg = np.mean([v.get('ndcg_cut_10', 0) for v in rr_results.values()])
            rr_avg_mrr = np.mean([v.get('recip_rank', 0) for v in rr_results.values()])
            rr_ndcg_list = [rr_results.get(str(q), {}).get('ndcg_cut_10', 0) for q in queries.keys()]
            
            # Paired t-test with Bonferroni correction
            stat, pval = ttest_rel(base_ndcg_list, rr_ndcg_list)
            alpha = 0.05 / len(rerankers)
            sig = "*" if pval < alpha else ""
            
            summary[stage][dataset][reranker] = {'ndcg': rr_avg_ndcg, 'mrr': rr_avg_mrr, 'pval': pval, 'sig': sig}

# Export and Print Beautiful Tables
os.makedirs("results/metrics", exist_ok=True)
models = ['Baseline'] + rerankers

for stage in first_stages:
    csv_path = f"results/metrics/summary_{stage}.csv"
    print(f"\n==================== FIRST STAGE: {stage.upper()} ====================")
    
    # ASCII Table Header
    header = f"{'Dataset':<12} | " + " | ".join([f"{m[:14]:<15}" for m in models])
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    # ASCII Table Rows
    for ds in datasets:
        if not summary[stage][ds]: continue
        row_str = f"{ds:<12} | "
        for m in models:
            if m in summary[stage][ds]:
                res = summary[stage][ds][m]
                val = f"{res['ndcg']:.4f}{res['sig']}"
                row_str += f"{val:<15} | "
            else:
                row_str += f"{'N/A':<15} | "
        print(row_str)
    print("-" * len(header))
    print("(*) indicates statistical significance (p < 0.05, Bonferroni corrected)")
    
    # Save Kitchen Sink CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        csv_headers = ['Dataset']
        for m in models:
            csv_headers.extend([f"{m}_NDCG@10", f"{m}_MRR@10", f"{m}_p-value", f"{m}_significant"])
        writer.writerow(csv_headers)
        
        for ds in datasets:
            if not summary[stage][ds]: continue
            row = [ds]
            for m in models:
                if m in summary[stage][ds]:
                    res = summary[stage][ds][m]
                    row.extend([res['ndcg'], res['mrr'], res['pval'], res['sig'] == '*'])
                else:
                    row.extend(['', '', '', ''])
            writer.writerow(row)
            
    print(f"\n>> Detailed spreadsheet saved to {csv_path}")