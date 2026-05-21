import subprocess
import time
import getpass
from huggingface_hub import login

def run_cmd(cmd, step_name):
    print(f"[{time.strftime('%X')}] STARTING: {step_name}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"[{time.strftime('%X')}] FINISHED: {step_name}")
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {step_name}")
        exit(1)

if __name__ == "__main__":
    print("=== STARTING SPLADE PREP (Go watch a movie) ===")
    
    hf_token = getpass.getpass("Paste your Hugging Face token here (it will be invisible): ")
    login(token=hf_token)
    
    # Assuming you have the actual python scripts written for these
    run_cmd("python scripts/download_data.py", "T0: Download Data")
    run_cmd("python scripts/splade_encode.py", "T2: SPLADE Encode Corpus")
    run_cmd("python scripts/retrieve.py --method splade", "T4: SPLADE Retrieve Top-100")
    
    print("=== SPLADE PREP COMPLETE ===")