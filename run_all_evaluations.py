#!/usr/bin/env python3
import subprocess
import yaml
import sys
from pathlib import Path

# Configuration
MODEL_DETAILS_FILE = "model_details.yaml"
BASE_OUTPUT_DIR = "/home/fleetprobe/fleetprobe/output_final_final"
DATA_DIR = "babylm_full_bpe_100M_8k"
MODEL_TYPE = "nanogpt"
TASKS = "dep_dist"

def load_model_details(filepath):
    """Load model details from YAML file."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def run_evaluation(model_id, model_name, output_dir, data_dir, model_type, tasks):
    """Run evaluation for a single model."""
    cmd = [
        "python", "babylm_eval.py",
        f"--model_type={model_type}",
        f"--output_dir={output_dir}",
        f"--tasks={tasks}",
        f"--model_name={model_name}",
        f"--data_dir={data_dir}"
    ]
    
    print(f"\n{'='*60}")
    print(f"Running evaluation for model {model_id}: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ Successfully completed evaluation for model {model_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running evaluation for model {model_id}: {e}")
        return False

def main():
    # Load model details
    try:
        models = load_model_details(MODEL_DETAILS_FILE)
    except FileNotFoundError:
        print(f"Error: {MODEL_DETAILS_FILE} not found")
        sys.exit(1)
    
    total_models = len(models)
    successful = 0
    failed = 0
    failed_models = []
    
    print(f"Starting evaluation of {total_models} models...\n")
    
    for model_id, model_info in models.items():
        model_name = model_info['output_folder_name']


        # set name for saving
        model_seed = model_info['seed']
        model_type = "mask" if model_info["masking"] else "nomask" if model_info["masking"] == False else "unknown"
        output_dir = f"{BASE_OUTPUT_DIR}/s{model_seed}_{model_type}_depdist"
        
        success = run_evaluation(
            model_id,
            model_name,
            output_dir,
            DATA_DIR,
            MODEL_TYPE,
            TASKS
        )
        
        if success:
            successful += 1
        else:
            failed += 1
            failed_models.append(model_id)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total models: {total_models}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed_models:
        print(f"\nFailed models: {', '.join(failed_models)}")
    
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()