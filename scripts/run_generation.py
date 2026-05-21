import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from beir.datasets.data_loader import GenericDataLoader

ALL_DATASETS = ["nq", "scifact", "trec-covid", "nfcorpus", "fiqa", "arguana"]
ALL_STAGES = ["bm25", "splade"]

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                    help="Datasets to generate for. Defaults to all 6 BEIR subsets.")
parser.add_argument("--first-stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES,
                    help="First-stage retrievers to generate for.")
parser.add_argument("--rerankers", nargs="+", default=None,
                    help="Reranker nicknames to generate for. Defaults to auto-discover from results/.")
parser.add_argument("--max-queries", type=int, default=500,
                    help="Max queries per dataset (default: 500).")
args = parser.parse_args()

def discover_rerankers(stages):
    found = set()
    for stage in stages:
        d = f"results/reranked_{stage}"
        if os.path.isdir(d):
            for name in os.listdir(d):
                if os.path.isdir(os.path.join(d, name)):
                    found.add(name)
    return sorted(found)

datasets = args.datasets
first_stages = args.first_stages
reranker_nicknames = args.rerankers if args.rerankers else discover_rerankers(first_stages)
# Always include baseline (unranked candidates)
run_models = ["baseline"] + reranker_nicknames
MAX_QUERIES = args.max_queries

print("=== PHASE 4: RAG GENERATION ===")
print("Waking up Qwen2.5-7B. Hide your VRAM.")

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

for stage in first_stages:
    for dataset in datasets:
        data_path = f"datasets/{dataset}"
        if not os.path.exists(data_path): continue
            
        corpus, queries, _ = GenericDataLoader(data_path).load(split="test")
        query_ids = list(queries.keys())[:MAX_QUERIES]
        
        for m_name in run_models:
            out_dir = f"results/generated_{stage}/{m_name}"
            os.makedirs(out_dir, exist_ok=True)
            out_path = f"{out_dir}/{dataset}.json"
            
            if os.path.exists(out_path):
                print(f"Skipping {stage}/{dataset}/{m_name}, already generated.")
                continue

            if m_name == "baseline":
                run_path = f"results/candidates_{stage}/{dataset}.json"
            else:
                run_path = f"results/reranked_{stage}/{m_name}/{dataset}.json"
                
            if not os.path.exists(run_path): continue
                
            with open(run_path, 'r') as f:
                run_data = json.load(f)
                
            print(f"Generating answers for {dataset} ({stage} -> {m_name})...")
            results = {}
            
            for qid in tqdm(query_ids, desc=m_name):
                q_text = queries[qid]
                cands = run_data.get(qid, [])[:5] 
                
                context = ""
                for idx, c in enumerate(cands):
                    doc = corpus.get(c["doc_id"], {})
                    d_text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
                    context += f"Passage {idx+1}:\n{d_text}\n\n"
                    
                messages = [
                    {"role": "system", "content": "You are a precise AI assistant. Answer the user's question using ONLY the provided passages. Be concise and direct."},
                    {"role": "user", "content": f"Passages:\n{context}\nQuestion: {q_text}"}
                ]
                
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False, temperature=None, top_p=None)
                
                answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                results[qid] = {"query": q_text, "answer": answer}
                
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)

print("\n=== GENERATION COMPLETE ===")