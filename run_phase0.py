import subprocess
import sys

def run_step(cmd, description):
    print(f"\n🚀 STARTING: {description}")
    print(f"Command: {cmd}")
    
    # We use shell=True but wrap the weird python file name in quotes just in case
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n💀 FATAL ERROR: {description} crashed.")
        print("Fix the stack trace above before trying again.")
        sys.exit(1)
    
    print(f"✅ FINISHED: {description}")

if __name__ == "__main__":
    print("=== INITIATING CORRECTED PHASE 0 ===")

    # 1. Ensure MS MARCO is downloaded
    run_step("python download_msmarco.py", "Download MS MARCO dataset")

    # 2. T1: Build BM25 Indexes (Will crash if Java isn't installed)
    run_step('python "pipeline_t1_t3_t4_t5.py" t1', "T1: BM25 Indexing")

    # 3. T3: Retrieve BM25 Candidates
    run_step('python "pipeline_t1_t3_t4_t5.py" t3', "T3: BM25 Retrieval")

    # 4. T4: Retrieve SPLADE Candidates (Skipped because you already did it manually)
    # run_step("python scripts/retrieve.py --method splade", "T4: SPLADE Fast GPU Retrieval")

    # 5. T5: Train LambdaMART (SPLADE features will be zeroed out)
    run_step("python t5_train_lambdamart.py", "T5: Train LambdaMART")

    print("\n=== PHASE 0 COMPLETE ===")
    print("Your candidate files and LightGBM model are ready for Phase 1.")