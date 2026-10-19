import os
import re
import json
import subprocess
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


MODEL_DIR = Path(
    "/home/andsch1/fmt-llm/output/"
    "out-babylm_100M_full_bpe_8k-6x6-mask_dev_pl20to1_r0.99954_em05-seed1337-1788779135"
) # you specify seed here 

DATA_DIR = Path(
    "/home/andsch1/fmt-llm/data/babylm_100M_full_bpe_8k"
)

EVAL_SCRIPT = Path(
    "/home/andsch1/evaluation-pipeline-2023/babylm_eval.py"
)

OUTPUT_ROOT = MODEL_DIR / "checkpoint_evaluation"


PHENOMENA = [
    "anaphor_agreement",
    "argument_structure",
    "binding",
    "control_raising",
    "determiner_noun_agreement",
    "ellipsis",
    "filler_gap",
    "irregular_forms",
    "island_effects",
    "npi_licensing",
    "quantifiers",
    "subject_verb_agreement",
]


def checkpoint_step(path):
    match = re.match(r"ckpt_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else None


def alpha_at_step(step, alpha0=6.0, alpha_adult=1.0, r=0.9999):
    return alpha_adult + (alpha0 - alpha_adult) * (r ** step)


def find_checkpoints():
    checkpoints = []

    for path in MODEL_DIR.glob("ckpt_*.pt"):
        step = checkpoint_step(path)
        if step is not None:
            checkpoints.append((step, path))

    return sorted(checkpoints)


def evaluate_checkpoint(step, checkpoint_path):
    checkpoint_output = OUTPUT_ROOT / f"ckpt_{step}"
    checkpoint_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(EVAL_SCRIPT),
        "--model_type", "nanogpt-dev-fm",
        "--output_dir", str(MODEL_DIR),
        "--data_dir", str(DATA_DIR),
        "--checkpoint_name", checkpoint_path.name,
        "--tasks", "blimp",
    ]

    print("\nRunning:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    # Current babylm_eval.py writes into MODEL_DIR/zeroshot.
    # Immediately copy/read results before evaluating the next checkpoint.
    results = {}

    for phenomenon in PHENOMENA:
        result_path = (
            MODEL_DIR
            / "zeroshot"
            / phenomenon
            / "eval_results.json"
        )

        with open(result_path) as f:
            result = json.load(f)

        accuracy = result["eval_accuracy"]
        results[phenomenon] = accuracy

        # Save checkpoint-specific copy
        dest_dir = checkpoint_output / phenomenon
        dest_dir.mkdir(parents=True, exist_ok=True)

        with open(dest_dir / "eval_results.json", "w") as f:
            json.dump(result, f, indent=2)

    return results


def make_plots(df):
    # Diagnostic mean across the 12 broad phenomena.
    # Do not call this official BLiMP overall unless we later reproduce
    # the official 67-subtask weighting.
    df["mean_12_phenomena"] = df[PHENOMENA].mean(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(
        df["step"],
        df["mean_12_phenomena"] * 100,
        marker="o"
    )
    plt.xlabel("Training step")
    plt.ylabel("Mean accuracy across 12 BLiMP phenomena (%)")
    plt.title("Development of BLiMP performance across training")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_ROOT / "blimp_overall_trajectory.png",
        dpi=300
    )
    plt.close()

    fig, axes = plt.subplots(
        4, 3,
        figsize=(13, 12),
        sharex=True
    )

    axes = axes.flatten()

    for ax, phenomenon in zip(axes, PHENOMENA):
        ax.plot(
            df["step"],
            df[phenomenon] * 100,
            marker="o"
        )
        ax.set_title(phenomenon.replace("_", " ").title())
        ax.set_ylabel("Accuracy (%)")

    for ax in axes[-3:]:
        ax.set_xlabel("Training step")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_ROOT / "blimp_phenomena_trajectory.png",
        dpi=300
    )
    plt.close()


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []

    for step, checkpoint_path in find_checkpoints():
        scores = evaluate_checkpoint(step, checkpoint_path)

        row = {
            "step": step,
            "checkpoint": checkpoint_path.name,
            "alpha": alpha_at_step(step),
            **scores,
        }

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("step")

    df["mean_12_phenomena"] = df[PHENOMENA].mean(axis=1)

    csv_path = OUTPUT_ROOT / "checkpoint_scores.csv"
    df.to_csv(csv_path, index=False)

    print("\nSaved:")
    print(csv_path)

    make_plots(df)


if __name__ == "__main__":
    main()