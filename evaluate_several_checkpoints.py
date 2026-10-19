import argparse
import json
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path(
    "/home/andsch1/fmt-llm/data/babylm_100M_full_bpe_8k"
)

EVAL_SCRIPT = Path(
    "/home/andsch1/evaluation-pipeline-2023/babylm_eval.py"
)

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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to one trained model directory.",
    )
    return parser.parse_args()


def checkpoint_step(path):
    match = re.fullmatch(r"ckpt_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else None


def find_checkpoints(model_dir):
    checkpoints = []

    for path in model_dir.glob("ckpt_*.pt"):
        step = checkpoint_step(path)
        if step is not None:
            checkpoints.append((step, path))

    return sorted(checkpoints)


def get_r_from_model_name(model_dir):
    match = re.search(r"_r([0-9.]+)_em", model_dir.name)

    if match is None:
        raise ValueError(
            f"Could not extract r from model folder name: {model_dir.name}"
        )

    return float(match.group(1))


def alpha_at_step(
    step,
    r,
    alpha0=6.0,
    alpha_adult=1.0,
):
    return alpha_adult + (alpha0 - alpha_adult) * (r ** step)


def checkpoint_results_complete(checkpoint_output):
    if not checkpoint_output.is_dir():
        return False

    for phenomenon in PHENOMENA:
        result_path = (
            checkpoint_output
            / phenomenon
            / "eval_results.json"
        )

        if not result_path.is_file():
            return False

        try:
            with open(result_path) as f:
                result = json.load(f)

            if "eval_accuracy" not in result:
                return False

        except (OSError, json.JSONDecodeError):
            return False

    return True


def load_checkpoint_results(checkpoint_output):
    results = {}

    for phenomenon in PHENOMENA:
        result_path = (
            checkpoint_output
            / phenomenon
            / "eval_results.json"
        )

        with open(result_path) as f:
            result = json.load(f)

        results[phenomenon] = result["eval_accuracy"]

    return results


def overall_evaluation_complete(output_root, checkpoints):
    if not output_root.is_dir():
        return False

    for step, _ in checkpoints:
        checkpoint_output = output_root / f"ckpt_{step}"

        if not checkpoint_results_complete(checkpoint_output):
            return False

    required_summary_files = [
        output_root / "checkpoint_scores.csv",
        output_root / "blimp_overall_trajectory.png",
        output_root / "blimp_phenomena_trajectory.png",
    ]

    if not all(path.is_file() for path in required_summary_files):
        return False

    try:
        df = pd.read_csv(output_root / "checkpoint_scores.csv")
    except Exception:
        return False

    required_columns = {
        "step",
        "checkpoint",
        "alpha",
        "mean_12_phenomena",
        *PHENOMENA,
    }

    if not required_columns.issubset(df.columns):
        return False

    expected_steps = {step for step, _ in checkpoints}
    recorded_steps = set(df["step"].astype(int))

    return expected_steps == recorded_steps


def evaluate_checkpoint(
    model_dir,
    output_root,
    step,
    checkpoint_path,
):
    checkpoint_output = output_root / f"ckpt_{step}"
    checkpoint_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(EVAL_SCRIPT),
        "--model_type", "nanogpt-dev-fm",
        "--output_dir", str(model_dir),
        "--data_dir", str(DATA_DIR),
        "--checkpoint_name", checkpoint_path.name,
        "--tasks", "blimp",
    ]

    print(f"\nEvaluating checkpoint {step}")
    print(" ".join(cmd), flush=True)

    subprocess.run(cmd, check=True)

    results = {}

    for phenomenon in PHENOMENA:
        result_path = (
            model_dir
            / "zeroshot"
            / phenomenon
            / "eval_results.json"
        )

        if not result_path.is_file():
            raise FileNotFoundError(
                f"Evaluation did not produce {result_path}"
            )

        with open(result_path) as f:
            result = json.load(f)

        if "eval_accuracy" not in result:
            raise KeyError(
                f"eval_accuracy missing from {result_path}"
            )

        results[phenomenon] = result["eval_accuracy"]

        destination = (
            checkpoint_output
            / phenomenon
            / "eval_results.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)

        with open(destination, "w") as f:
            json.dump(result, f, indent=2)

    return results


def make_plots(df, output_root):
    plt.figure(figsize=(8, 5))
    plt.plot(
        df["step"],
        df["mean_12_phenomena"] * 100,
        marker="o",
    )
    plt.xlabel("Training step")
    plt.ylabel("Mean accuracy across 12 BLiMP phenomena (%)")
    plt.title("Development of BLiMP performance across training")
    plt.tight_layout()
    plt.savefig(
        output_root / "blimp_overall_trajectory.png",
        dpi=300,
    )
    plt.close()

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(13, 12),
        sharex=True,
    )

    axes = axes.flatten()

    for ax, phenomenon in zip(axes, PHENOMENA):
        ax.plot(
            df["step"],
            df[phenomenon] * 100,
            marker="o",
        )
        ax.set_title(phenomenon.replace("_", " ").title())
        ax.set_ylabel("Accuracy (%)")

    for ax in axes[-3:]:
        ax.set_xlabel("Training step")

    plt.tight_layout()
    plt.savefig(
        output_root / "blimp_phenomena_trajectory.png",
        dpi=300,
    )
    plt.close(fig)


def main():
    args = parse_arguments()
    model_dir = args.model_dir.resolve()
    output_root = model_dir / "checkpoint_evaluation"

    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Model directory does not exist: {model_dir}"
        )

    checkpoints = find_checkpoints(model_dir)

    if not checkpoints:
        raise FileNotFoundError(
            f"No ckpt_*.pt files found in {model_dir}"
        )

    print(f"Model: {model_dir}")
    print(
        "Checkpoints:",
        ", ".join(str(step) for step, _ in checkpoints),
    )

    if overall_evaluation_complete(output_root, checkpoints):
        print("\nEvaluation is already complete.")
        print("Nothing needs to be run.")
        return

    output_root.mkdir(parents=True, exist_ok=True)

    r = get_r_from_model_name(model_dir)
    print(f"Using r={r}")

    rows = []

    for step, checkpoint_path in checkpoints:
        checkpoint_output = output_root / f"ckpt_{step}"

        if checkpoint_results_complete(checkpoint_output):
            print(
                f"\nCheckpoint {step} is already complete; "
                "loading saved results."
            )
            scores = load_checkpoint_results(checkpoint_output)
        else:
            print(
                f"\nCheckpoint {step} is missing or incomplete; "
                "running evaluation."
            )
            scores = evaluate_checkpoint(
                model_dir,
                output_root,
                step,
                checkpoint_path,
            )

        rows.append(
            {
                "step": step,
                "checkpoint": checkpoint_path.name,
                "alpha": alpha_at_step(step, r=r),
                **scores,
            }
        )

    df = pd.DataFrame(rows).sort_values("step")
    df["mean_12_phenomena"] = df[PHENOMENA].mean(axis=1)

    csv_path = output_root / "checkpoint_scores.csv"
    df.to_csv(csv_path, index=False)

    make_plots(df, output_root)

    print("\nEvaluation complete.")
    print(f"Saved CSV: {csv_path}")
    print(
        "Saved plots:",
        output_root / "blimp_overall_trajectory.png",
        output_root / "blimp_phenomena_trajectory.png",
        sep="\n",
    )


if __name__ == "__main__":
    main()