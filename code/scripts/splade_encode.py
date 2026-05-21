import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from beir.datasets.data_loader import GenericDataLoader

# RTX 5090 privilege
BATCH_SIZE = 256  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading SPLADE-v3 onto {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained("naver/splade-v3")
model = AutoModelForMaskedLM.from_pretrained("naver/splade-v3").to(DEVICE).eval()

datasets = ["nq", "scifact", "trec-covid", "nfcorpus", "fiqa", "arguana"]
os.makedirs("indexes/splade", exist_ok=True)

def encode_batch(texts):
    """Encode a batch of texts with SPLADE-v3.

    Applies the official SPLADE pooling: max(log(1 + relu(logits)) * mask, dim=1).
    Returns a dense tensor of shape (batch_size, vocab_size).
    """
    with torch.no_grad():
        tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
        output = model(**tokens)
        # The official SPLADE pooling formula
        vecs = torch.max(
            torch.log(1 + torch.relu(output.logits)) * tokens.attention_mask.unsqueeze(-1), 
            dim=1
        )[0]
    return vecs

for dataset in datasets:
    print(f"\nEncoding corpus for {dataset}...")
    data_path = f"datasets/{dataset}"
    if not os.path.exists(data_path):
        continue
        
    corpus, _, _ = GenericDataLoader(data_path).load(split="test")
    doc_ids = list(corpus.keys())
    texts = [f"{corpus[did].get('title', '')} {corpus[did].get('text', '')}" for did in doc_ids]
    
    encoded_corpus = {}
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_ids = doc_ids[i:i + BATCH_SIZE]
        batch_texts = texts[i:i + BATCH_SIZE]
        
        vecs = encode_batch(batch_texts)
        
        # Sparsify and save to dict to avoid nuking your RAM
        for doc_id, vec in zip(batch_ids, vecs):
            non_zero_idx = vec.nonzero().squeeze().cpu().tolist()
            if isinstance(non_zero_idx, int): # Handle single active token edge case
                non_zero_idx = [non_zero_idx]
                
            weights = vec[non_zero_idx].cpu().tolist()
            if not isinstance(weights, list):
                weights = [weights]
                
            encoded_corpus[doc_id] = dict(zip(non_zero_idx, weights))
            
    # Save the dataset index
    out_file = f"indexes/splade/{dataset}_corpus.json"
    with open(out_file, 'w') as f:
        json.dump(encoded_corpus, f)
    print(f"Saved {len(encoded_corpus)} docs to {out_file}")