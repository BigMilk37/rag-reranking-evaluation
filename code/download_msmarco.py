"""
download_msmarco.py
===================
Downloads the MS MARCO passage ranking dataset for LambdaMART training (T5).
"""

import os
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

OUT_DIR = Path("datasets/msmarco")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "collection.tsv": "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz",
    "queries.train.tsv": "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz",
    "qrels.train.tsv": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv",
    "top1000.train.tsv": "https://msmarco.z22.web.core.windows.net/msmarcoranking/top1000.train.tar.gz",
}

class ProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        """Callback for urllib.request.urlretrieve to update tqdm bar."""
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download(url: str, dest: Path):
    """Download a file from url to dest with a tqdm progress bar."""
    print(f"Downloading {dest.name}...")
    with ProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_to)

def extract_tar(tar_path: Path, out_dir: Path):
    """Extract a tar.gz archive. For top1000.train truncates to top-100
    per query on the fly to avoid materializing the full 67GB file.
    """
    print(f"Extracting {tar_path.name}...")
    
    if "top1000" in tar_path.name:
        print("Truncating the 67GB bloatware to Top-100 to save your pod...")
        out_file = out_dir / "top1000.train.tsv"
        counts = defaultdict(int)
        
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if "top1000" in member.name:
                    f = tar.extractfile(member)
                    with open(out_file, "w", encoding="utf-8") as out:
                        for line in f:
                            line_str = line.decode("utf-8", errors="ignore")
                            qid = line_str.split("\t", 1)[0]
                            if counts[qid] < 100:
                                out.write(line_str)
                                counts[qid] += 1
                    break
        return

    with tarfile.open(tar_path) as tar:
        tar.extractall(out_dir)

if __name__ == "__main__":
    print("=== Downloading MS MARCO passage ranking dataset ===")

    qrels_dest = OUT_DIR / "qrels.train.tsv"
    if not qrels_dest.exists():
        download(FILES["qrels.train.tsv"], qrels_dest)
    else:
        print("Skipping qrels.train.tsv, already exists.")

    for filename, url in FILES.items():
        if not url.endswith(".tar.gz"):
            continue
            
        final_path = OUT_DIR / filename
        
        # Hard check for the extracted file. 
        if final_path.exists():
            print(f"Skipping {filename}, extracted file is already safe on your hard drive.")
            continue
            
        tar_dest = OUT_DIR / (filename + ".tar.gz")
        if not tar_dest.exists():
            download(url, tar_dest)
        extract_tar(tar_dest, OUT_DIR)

    print("\n=== MS MARCO download complete ===")
    print("Files in datasets/msmarco/:")
    for f in sorted(OUT_DIR.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name:<30} {size_mb:>8.1f} MB")