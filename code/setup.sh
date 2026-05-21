#!/bin/bash
echo "Installing system dependencies (Java is required for Pyserini)..."
sudo apt-get update
sudo apt-get install -y default-jdk

echo "Installing Python libraries..."
python3 -m pip install torch transformers accelerate sentence-transformers beir tqdm huggingface_hub pyserini rouge-score bert-score pandas numpy scipy scikit-learn lightgbm pytrec-eval

echo "Creating directory structure..."
mkdir -p scripts datasets indexes/splade results/candidates_splade results/candidates_bm25 results/metrics results/generated_bm25 results/generated_splade logs

echo ""
echo "Naver locked the doors. You need a Hugging Face token to proceed."
echo "1. Go to https://huggingface.co/naver/splade-v3 and accept the terms."
echo "2. Go to your HF account settings and copy your access token."
echo "Paste your token below when prompted (it will be invisible as you type):"
huggingface-cli login

echo ""
echo "Setup complete. Try not to delete anything important this time."