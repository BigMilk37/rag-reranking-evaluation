import subprocess
import sys

if __name__ == "__main__":
    # Forward all arguments directly to the central phase 1 script
    args_str = " ".join(sys.argv[1:])
    cmd = f"python scripts/run_phase1.py {args_str}".strip()
    
    print(f"=== PHASE 1 WRAPPER ===")
    print(f"Delegating execution to: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    sys.exit(result.returncode)