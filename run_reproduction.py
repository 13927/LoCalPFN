import subprocess
import pandas as pd
import os
import time

# Datasets identified from OpenML-CC18 matching LoCalPFN criteria
# Filtered by: 2k < instances < 200k, features <= 100, classes <= 10
DATASET_IDS = [
    3, 14, 16, 18, 22, 28, 32, 44, 46, 151, 
    182, 38, 1053, 1067, 1590, 1489, 1497, 1487, 
    1475, 4534, 1461, 4538, 40668, 40983, 40984, 
    41027, 23517, 40701
]

# For quick demonstration, we pick 3 diverse datasets
# ID 3: kr-vs-kp (Chess), 3196 instances, 36 features, 2 classes
# ID 14: mfeat-fourier, 2000 instances, 76 features, 10 classes
# ID 32: pendigits, 10992 instances, 16 features, 10 classes
TARGET_IDS = [3, 14, 32] 

def run_experiment(dataset_id, method, num_epochs=21):
    exp_name = f"repro_{method}_{dataset_id}"
    cmd = [
        "python", "main.py",
        f"--exp_name={exp_name}",
        "--datasets=openml",
        f"--openml_dataset_id={dataset_id}",
        "--device=cuda:0", # Use GPU as requested
        method
    ]
    
    if method == "ft":
        cmd.append(f"--num_epochs={num_epochs}")
    # else:
    #     cmd.append(method) # This was redundant because method is already in cmd above

    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    # Run the command directly using subprocess.run
    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except Exception as e:
        print(f"Failed to execute command: {e}")
        return None

    duration = time.time() - start_time
    
    # Parse output for results
    # Output format: "Best: Loss: ..., Accuracy: ..., F1: ..., AUC: ..."
    output = process.stdout
    print(output)
    
    if process.returncode != 0:
        print(f"Error running {method} on {dataset_id}:")
        print(process.stderr)
        return None
        
    acc = 0.0
    f1 = 0.0
    auc = 0.0
    
    for line in output.split('\n'):
        if "Best: Loss:" in line:
            # Format: Best: Loss: 0.1234, Accuracy: 0.5678, F1: 0.9012, AUC: 0.3456
            parts = line.split(',')
            for part in parts:
                if "Accuracy" in part:
                    try:
                        acc = float(part.split(':')[1].strip())
                    except: pass
                if "F1" in part:
                    try:
                        f1 = float(part.split(':')[1].strip())
                    except: pass
                if "AUC" in part:
                    try:
                        auc = float(part.split(':')[1].strip())
                    except: pass
                    
    return {
        "dataset_id": dataset_id,
        "method": method,
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "duration": duration
    }

def main():
    results = []
    
    print(f"Starting reproduction on {len(TARGET_IDS)} datasets...")
    
    for did in TARGET_IDS:
        print(f"\n=== Processing Dataset {did} ===")
        
        # 1. Run kNN Baseline
        print(f"--- Running kNN Baseline ---")
        res_knn = run_experiment(did, "knn")
        if res_knn:
            results.append(res_knn)
            
        # 2. Run LoCalPFN (Fine-tuning)
        # Using 5 epochs for speed in demo, paper uses 21
        print(f"--- Running LoCalPFN (FT) ---")
        res_ft = run_experiment(did, "ft", num_epochs=5) 
        if res_ft:
            results.append(res_ft)
            
    # Save Results
    df = pd.DataFrame(results)
    print("\n=== Final Results ===")
    print(df)
    
    df.to_csv("reproduction_results.csv", index=False)
    print("Results saved to reproduction_results.csv")

if __name__ == "__main__":
    main()
