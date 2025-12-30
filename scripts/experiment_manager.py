"""
Run multiple experiments sequentially or in parallel.
"""
import subprocess
import yaml
import os
from pathlib import Path
from multiprocessing import Process
import time

def run_experiment(config_path, gpu_ids="0,1", port=29501, dry_run=False):
    """Run a single experiment."""
    cmd = [
        "accelerate", "launch",
        "--main_process_port", str(port),
        "scripts/train_wm.py",
        "--config", config_path
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    
    print(f"\n{'='*80}")
    print(f"Running experiment: {config_path}")
    print(f"GPUs: {gpu_ids}")
    print(f"Port: {port}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    if dry_run:
        print("DRY RUN - Skipping execution")
        return 0
    
    result = subprocess.run(cmd, env=env)
    return result.returncode

def assign_gpus(available_gpus, gpus_per_experiment, experiment_idx):
    """Assign GPU IDs for an experiment based on available GPUs."""
    gpu_list = [int(g.strip()) for g in available_gpus.split(',')]
    start_idx = (experiment_idx * gpus_per_experiment) % len(gpu_list)
    
    assigned = []
    for i in range(gpus_per_experiment):
        gpu_idx = (start_idx + i) % len(gpu_list)
        assigned.append(gpu_list[gpu_idx])
    
    return ','.join(map(str, assigned))

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments_dir', type=str, default='configs/experiments')
    parser.add_argument('--experiments', nargs='+', help='Specific experiments to run')
    parser.add_argument('--available_gpus', type=str, default='0,1,2,3,4,5,6,7', 
                        help='Comma-separated list of available GPU IDs')
    parser.add_argument('--gpus_per_experiment', type=int, default=1,
                        help='Number of GPUs to assign to each experiment')
    parser.add_argument('--parallel', action='store_true',
                        help='Run experiments in parallel')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    
    # Find all experiment configs
    experiments_dir = Path(args.experiments_dir)
    if args.experiments:
        configs = [experiments_dir / f"{exp}.yaml" for exp in args.experiments]
    else:
        configs = list(experiments_dir.glob("*.yaml"))
    
    print(f"Found {len(configs)} experiments to run")
    print(f"Available GPUs: {args.available_gpus}")
    print(f"GPUs per experiment: {args.gpus_per_experiment}")
    print(f"Mode: {'Parallel' if args.parallel else 'Sequential'}\n")
    
    if args.parallel:
        # Run experiments in parallel
        processes = []
        for idx, config in enumerate(configs):
            gpu_ids = assign_gpus(args.available_gpus, args.gpus_per_experiment, idx)
            port = 29501 + idx  # Use different port for each experiment
            
            p = Process(target=run_experiment, args=(str(config), gpu_ids, port, args.dry_run))
            p.start()
            processes.append((p, config))
            time.sleep(2)  # Small delay to avoid race conditions
        
        # Wait for all processes to complete
        for p, config in processes:
            p.join()
            if p.exitcode != 0:
                print(f"❌ Experiment failed: {config}")
            else:
                print(f"✅ Experiment completed: {config}")
    else:
        # Run experiments sequentially
        for idx, config in enumerate(configs):
            gpu_ids = assign_gpus(args.available_gpus, args.gpus_per_experiment, idx)
            returncode = run_experiment(str(config), gpu_ids, 29501, args.dry_run)
            if returncode != 0:
                print(f"❌ Experiment failed: {config}")
                break
            print(f"✅ Experiment completed: {config}")