import os
from beir import util as beir_util

# The newly approved non-bloated datasets
datasets = ["nq", "scifact", "trec-covid", "nfcorpus", "fiqa", "arguana"]
os.makedirs("datasets", exist_ok=True)

print("Pulling datasets down from the void...")
for dataset in datasets:
    out_dir = os.path.join("datasets", dataset)
    if os.path.exists(out_dir):
        print(f"Skipping {dataset}, already exists.")
        continue
        
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    try:
        beir_util.download_and_unzip(url, "datasets")
        print(f"Downloaded: {dataset}")
    except Exception as e:
        print(f"Failed to download {dataset}: {e}")