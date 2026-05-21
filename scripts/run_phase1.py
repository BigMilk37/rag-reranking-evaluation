import subprocess
import sys
import argparse

ALL_DATASETS = ["nq", "scifact", "trec-covid", "nfcorpus", "fiqa", "arguana"]
ALL_STAGES = ["bm25", "splade"]
DEFAULT_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "mixedbread-ai/mxbai-rerank-base-v2",
    "castorini/monot5-base-msmarco-v2",
    "BAAI/bge-reranker-base",
]

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="HF model IDs to run as re-rankers.")
parser.add_argument("--first-stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES,
                    help="First-stage retrievers to re-rank.")
parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                    help="Datasets to re-rank. Defaults to all 6 BEIR subsets.")
parser.add_argument("--skip-lambdamart", action="store_true",
                    help="Skip LambdaMART inference.")
args = parser.parse_args()

def run(cmd):
    print(f"\n🚀 Executing: {cmd}")
    if subprocess.run(cmd, shell=True).returncode != 0:
        print(f"💀 Crash detected. Fix it.")
        sys.exit(1)

print("=== PHASE 1 INITIATION ===")

datasets_arg = " ".join(args.datasets)

for base in args.first_stages:
    print(f"\n--- Processing {base.upper()} Candidates ---")

    for model in args.models:
        run(f"python scripts/run_rerankers.py --first-stage {base} --model {model} --datasets {datasets_arg}")

    if not args.skip_lambdamart:
        run(f"python scripts/infer_lambdamart.py --first-stage {base} --datasets {datasets_arg}")

print("\n=== PHASE 1 COMPLETE ===")