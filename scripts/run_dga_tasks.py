#!/usr/bin/env python3
import subprocess
import sys
import time
import re
import os
from pathlib import Path
from datetime import datetime
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from hydra.utils import get_original_cwd

@dataclass
class RunnerConfig:
    dataset: str = "all"

cs = ConfigStore.instance()
cs.store(name="config", node=RunnerConfig)

def get_tasks(dataset_name):
    """
    Generate list of (source, target) tuples for the given dataset.
    """
    if dataset_name == "office-31":
        domains = ["amazon", "webcam", "dslr"]
    elif dataset_name == "office-home":
        domains = ["Art", "Clipart", "Product", "RealWorld"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    tasks = []
    for source in domains:
        for target in domains:
            if source != target:
                tasks.append((source, target))
    return tasks

def run_task(dataset, source, target, project_root, exp_dir):
    """
    Run a single DGA task.
    Returns the best H-score found in the logs, or 0.0 if failed.
    """
    # Task output directory (absolute path)
    task_dir = exp_dir / f"{source}_{target}"
    
    # Construct command with specific output directory
    cmd = [
        "uv", "run", "python", "src/main.py",
        "method=dga",
        f"dataset={dataset}",
        f"dataset.source={source}",
        f"dataset.target={target}",
        "exp_name=dga_osda",
        f"hydra.run.dir={task_dir}" # Override hydra run dir for the subprocess
    ]
    
    print(f"\n{'='*60}")
    print(f"Starting Task: {dataset} | {source} -> {target}")
    print(f"Output Directory: {task_dir}")
    print(f"{'='*60}\n")

    best_h_score = 0.0
    
    try:
        # Check if uv is available
        subprocess.run(["uv", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("Error: 'uv' command not found. Please install uv or run in an environment where it is available.")
        sys.exit(1)
        
    try:
        # Run command and capture output continuously
        # Run from project root so src/main.py works
        process = subprocess.Popen(
            cmd,
            cwd=project_root, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        for line in process.stdout:
            print(line, end="") # Stream to console
            
            # Check for best H-score in final summary
            match_final = re.search(r"Best:\s+(\d+\.\d+)%", line)
            if match_final:
                best_h_score = max(best_h_score, float(match_final.group(1)))
            
            # Also track H-scores during training if final line missed
            match_intermediate = re.search(r"H-score=(\d+\.\d+)%", line)
            if match_intermediate:
                best_h_score = max(best_h_score, float(match_intermediate.group(1)))

        process.wait()
        
        if process.returncode == 0:
            print(f"\n[SUCCESS] Task {source} -> {target} completed. Best H-score: {best_h_score:.2f}%")
        else:
            print(f"\n[FAILURE] Task {source} -> {target} failed with return code {process.returncode}.")
            
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILURE] Task {source} -> {target} exception: {e}")
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user.")
        raise

    return best_h_score

@hydra.main(version_base=None, config_name="config", config_path=None)
def main(cfg: RunnerConfig):
    datasets_to_run = []
    
    dataset_choice = cfg.dataset
    
    if dataset_choice == "all":
        datasets_to_run = ["office-31", "office-home"]
    elif dataset_choice in ["office-31", "office-home"]:
        datasets_to_run = [dataset_choice]
    else:
        print(f"Invalid dataset choice: {dataset_choice}. support: office-31, office-home, all")
        return

    # Get paths
    # Since we set hydra.run.dir, CWD is now the experiment directory
    exp_dir = Path.cwd() 
    project_root = Path(get_original_cwd())
    
    print(f"Experiment Directory: {exp_dir}")

    total_start_time = time.time()
    
    all_results = {} # dataset -> list of (source, target, score)

    try:
        for dataset in datasets_to_run:
            tasks = get_tasks(dataset)
            print(f"\n>>> Processing dataset: {dataset} ({len(tasks)} tasks)")
            
            dataset_results = []
            
            for source, target in tasks:
                score = run_task(dataset, source, target, project_root, exp_dir)
                dataset_results.append((source, target, score))
            
            all_results[dataset] = dataset_results

    except KeyboardInterrupt:
        print("\nExiting early...")
    
    total_duration = time.time() - total_start_time
    
    # Generate Summary
    summary_path = exp_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Experiment: {exp_dir.name}\n") # Directory name is the timestamp
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Duration: {total_duration/60:.2f} minutes\n\n")
        
        for dataset, results in all_results.items():
            f.write(f"Dataset: {dataset}\n")
            f.write("-" * 30 + "\n")
            
            scores = [r[2] for r in results]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            for source, target, score in results:
                f.write(f"{source} -> {target}: {score:.2f}%\n")
            
            f.write("-" * 30 + "\n")
            f.write(f"Average H-score: {avg_score:.2f}%\n\n")
            
    print(f"\n{'='*60}")
    print(f"All requested tasks completed.")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")
    
    # Print summary to console
    with open(summary_path, "r") as f:
        print(f.read())

if __name__ == "__main__":
    # Dynamically set the Hydra run directory to results/dga_osda_all/TIMESTAMP
    # This prevents the default outputs/ directory creation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Hack: Inject into sys.argv if not present
    has_run_dir = any(arg.startswith("hydra.run.dir=") for arg in sys.argv)
    if not has_run_dir:
        # Use nested structure: results/dga_osda_all/{timestamp}
        sys.argv.append(f"hydra.run.dir=results/dga_osda_all/{timestamp}")
        
    # Also set job.chdir=True so main() runs inside that dir
    has_chdir = any(arg.startswith("hydra.job.chdir=") for arg in sys.argv)
    if not has_chdir:
        sys.argv.append("hydra.job.chdir=True")
        
    main()
