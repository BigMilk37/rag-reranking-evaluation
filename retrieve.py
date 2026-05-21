import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from beir.datasets.data_loader import GenericDataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, choices=["splade", "bm25"], required=True)
args = parser.parse_args()

datasets = ["nq", "scifact", "trec-covid", "nfcorpus", "fiqa", "arguana"]
os.makedirs(f"results/candidates_{args.method}", exist_ok=True)

if args.method == "bm25": exit(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 30522  # Standard BERT vocab size
CHUNK_SIZE = 50000  # Will use ~6GB of your 32GB VRAM per chunk

tokenizer = AutoTokenizer.from_pretrained("naver/splade-v3")
model = AutoModelForMaskedLM.from_pretrained("naver/splade-v3").to(DEVICE).eval()

def encode_queries(queries_dict):
    """Encode all queries with SPLADE-v3 in batches of 64.

    Returns (q_vecs dict {qid: dense tensor}, q_ids list).
    Uses the same pooling formula as splade_encode.py for consistency.
    """
    q_ids = list(queries_dict.keys())
    q_texts = [queries_dict[qid] for qid in q_ids]
    q_vecs = {}
    
    with torch.no_grad():
        for i in range(0, len(q_texts), 64):
            batch_ids = q_ids[i:i+64]
            tokens = tokenizer(q_texts[i:i+64], return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
            output = model(**tokens)
            vecs = torch.max(
                torch.log(1 + torch.relu(output.logits)) * tokens.attention_mask.unsqueeze(-1), dim=1
            )[0]
            for qid, vec in zip(batch_ids, vecs):
                q_vecs[qid] = vec
    return q_vecs, q_ids

for dataset in datasets:
    print(f"\nRetrieving {dataset} with SPLADE via GPU Matrix Math...")
    data_path = f"datasets/{dataset}"
    index_path = f"indexes/splade/{dataset}_corpus.json"
    
    if not os.path.exists(data_path) or not os.path.exists(index_path): continue
        
    _, queries, _ = GenericDataLoader(data_path).load(split="test")
    
    print("Loading corpus vectors...")
    with open(index_path, 'r') as f:
        corpus_dict = json.load(f)
    doc_ids = list(corpus_dict.keys())
        
    print("Encoding queries...")
    query_vecs, q_ids = encode_queries(queries)
    
    # Build query matrix [Num_Queries, Vocab_Size]
    query_tensor = torch.zeros(len(q_ids), VOCAB_SIZE, device=DEVICE)
    for i, qid in enumerate(q_ids):
        query_tensor[i] = query_vecs[qid]
        
    global_top_scores = torch.full((len(q_ids), 100), -float('inf'), device=DEVICE)
    global_top_indices = torch.zeros((len(q_ids), 100), dtype=torch.long, device=DEVICE)
    
    print("GPU matrix multiplication engaged...")
    for i in tqdm(range(0, len(doc_ids), CHUNK_SIZE)):
        chunk_ids = doc_ids[i:i+CHUNK_SIZE]
        chunk_tensor = torch.zeros(len(chunk_ids), VOCAB_SIZE, device=DEVICE)
        
        for j, did in enumerate(chunk_ids):
            for idx, weight in corpus_dict[did].items():
                chunk_tensor[j, int(idx)] = weight
                
        # The actual math: [Queries, Vocab] @ [Vocab, Chunk_Docs] -> [Queries, Chunk_Docs]
        scores = torch.matmul(query_tensor, chunk_tensor.T)
        
        # Merge and keep top 100
        combined_scores = torch.cat([global_top_scores, scores], dim=1)
        chunk_global_indices = torch.arange(i, i + len(chunk_ids), device=DEVICE).unsqueeze(0).expand(len(q_ids), -1)
        combined_indices = torch.cat([global_top_indices, chunk_global_indices], dim=1)
        
        global_top_scores, top_idx = torch.topk(combined_scores, 100, dim=1)
        global_top_indices = torch.gather(combined_indices, 1, top_idx)

    print("Formatting candidates...")
    candidates = {}
    for i, qid in enumerate(q_ids):
        top_doc_indices = global_top_indices[i].cpu().tolist()
        top_doc_scores = global_top_scores[i].cpu().tolist()
        
        candidates[qid] = [
            {"doc_id": doc_ids[doc_idx], "score": score, "rank": rank+1}
            for rank, (doc_idx, score) in enumerate(zip(top_doc_indices, top_doc_scores))
        ]
                           
    with open(f"results/candidates_splade/{dataset}.json", 'w') as f:
        json.dump(candidates, f, indent=2)