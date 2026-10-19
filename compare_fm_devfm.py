import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

EVAL_ROOT = Path("/home/andsch1/evaluation-pipeline-2023")
EVAL_SCRIPT = EVAL_ROOT / "babylm_eval.py"

DATA_DIR = Path(
    "/home/andsch1/fmt-llm/data/babylm_100M_full_bpe_8k"
)

CHECKPOINT_NAME = "ckpt_44000.pt"

# Fill these with the exact completed run folders.
# Explicit paths are safer because you have duplicate/aborted runs.

DEV_MODELS = {
    42: Path(
        "/home/andsch1/fmt-llm/output/"
        "out-babylm_100M_full_bpe_8k-6x6-mask_dev_pl6to1_r0.9999_em05-seed42-1787319250"
    ),
    123: Path(
        "/home/andsch1/fmt-llm/output/"
        "out-babylm_100M_full_bpe_8k-6x6-mask_dev_pl6to1_r0.9999_em05-seed123-1787426408"
    ),
    456: Path(
        "/home/andsch1/fmt-llm/output/"
        "out-babylm_100M_full_bpe_8k-6x6-mask_dev_pl6to1_r0.9999_em05-seed456-1787426511"
    ),
    789: Path(
        "/home/andsch1/fmt-llm/output/"
        "out-babylm_100M_full_bpe_8k-6x6-mask_dev_pl6to1_r0.9999_em05-seed789-1787426564"
    ),
    1337: Path(
        "/home/andsch1/fmt-llm/output/"
        "out-babylm_100M_full_bpe_8k-6x6-mask_dev_pl6to1_r0.9999_em05-seed1337-1787426273"
    ),
}


FM_MODELS = {
    42: Path(
        "/home/andsch1/fmt-llm/output/"
        "out-babylm_100M_full_bpe_8k-6x6-mask_pl002_em05-seed42-1787323118"
    ),
    123: Path(
        "/home/andsch1/fmt-llm/output/"
        "out-babylm_100M_full_bpe_8k-6x6-mask_pl002_em05-seed123-1787477389"
    ),
    456: Path(
        "/home/andsch1/fmt-llm/output/"
        "out-babylm_100M_full_bpe_8k-6x6-mask_pl002_em05-seed456-1787477445"
    ),
    789: Path(
        "/home/andsch1/fmt-llm/output/"
        "out-babylm_100M_full_bpe_8k-6x6-mask_pl002_em05-seed789-1787477506"
    ),
    1337: Path(
        "/home/andsch1/fmt-llm/output/"
        "out-babylm_100M_full_bpe_8k-6x6-mask_pl002_em05-seed1337-1787459333"
    ),
}

OUTPUT_DIR = EVAL_ROOT / "fm_vs_devfm_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

# Set False if BLiMP has already been evaluated
# and zeroshot/.../eval_results.json exists.
RUN_EVALUATION = True

N_BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 12345


# ============================================================
# CHECKPOINT VERIFICATION
# ============================================================

def verify_checkpoint(model_dir, expected_seed, expected_type):
    import torch

    ckpt_path = model_dir / CHECKPOINT_NAME

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg = checkpoint.get("config", {})

    actual_step = checkpoint.get("iter_num")
    actual_seed = cfg.get("torch_seed_default")
    dataset = cfg.get("dataset")
    developmental = cfg.get("wm_developmental")

    print("\nChecking:", model_dir)
    print("  step:", actual_step)
    print("  seed:", actual_seed)
    print("  dataset:", dataset)
    print("  developmental:", developmental)

    if actual_step != 44000:
        raise ValueError(
            f"{model_dir.name}: expected step 44000, got {actual_step}"
        )

    if actual_seed != expected_seed:
        raise ValueError(
            f"{model_dir.name}: expected seed {expected_seed}, "
            f"got {actual_seed}"
        )

    if dataset != "babylm_100M_full_bpe_8k":
        raise ValueError(
            f"{model_dir.name}: unexpected dataset {dataset}"
        )

    if expected_type == "dev" and developmental is not True:
        raise ValueError(
            f"{model_dir.name}: expected developmental model."
        )

    if expected_type == "fm" and developmental is not False:
        raise ValueError(
            f"{model_dir.name}: expected static FM model."
        )


# ============================================================
# RUN BLIMP
# ============================================================

def run_blimp(model_dir):
    cmd = [
        "python",
        str(EVAL_SCRIPT),
        "--model_type", "nanogpt-dev-fm",
        "--output_dir", str(model_dir),
        "--data_dir", str(DATA_DIR),
        "--checkpoint_name", CHECKPOINT_NAME,
        "--tasks", "blimp",
    ]

    print("\nRunning:")
    print(" ".join(cmd))

    subprocess.run(
        cmd,
        check=True,
        cwd=EVAL_ROOT,
    )


# ============================================================
# READ BLIMP RESULTS
# ============================================================

def read_results(model_dir):
    scores = {}

    for phenomenon in PHENOMENA:
        result_path = (
            model_dir
            / "zeroshot"
            / phenomenon
            / "eval_results.json"
        )

        if not result_path.exists():
            raise FileNotFoundError(
                f"Missing result: {result_path}"
            )

        with open(result_path) as f:
            result = json.load(f)

        scores[phenomenon] = result["eval_accuracy"]

    return scores


# ============================================================
# BOOTSTRAP T-TEST
# ============================================================

def bootstrap_paired_ttest(dev, fm, n_boot=10000, seed=12345):
    """
    Paired bootstrap t-test following the logic described
    in Thamma & Heilbron:

    1. Compute paired differences.
    2. Mean-center differences to create a zero-mean null.
    3. Bootstrap from this null distribution.
    4. Compute bootstrap t statistics.
    5. Use equal-tail two-sided p-value.
    6. Bootstrap the original differences for the 95% CI.
    """

    dev = np.asarray(dev, dtype=float)
    fm = np.asarray(fm, dtype=float)

    if len(dev) != len(fm):
        raise ValueError("Dev-FM and FM arrays must have equal length.")

    diff = dev - fm
    n = len(diff)

    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)

    if sd_diff == 0:
        observed_t = np.inf if mean_diff != 0 else 0.0
    else:
        observed_t = mean_diff / (sd_diff / np.sqrt(n))

    # Create zero-mean null distribution
    centered = diff - mean_diff

    rng = np.random.default_rng(seed)

    bootstrap_t = []

    for _ in range(n_boot):
        sample = rng.choice(centered, size=n, replace=True)

        sample_sd = np.std(sample, ddof=1)

        if sample_sd == 0:
            continue

        t_value = np.mean(sample) / (
            sample_sd / np.sqrt(n)
        )

        bootstrap_t.append(t_value)

    bootstrap_t = np.asarray(bootstrap_t)

    # Equal-tail two-sided bootstrap p-value
    lower_tail = np.mean(bootstrap_t <= observed_t)
    upper_tail = np.mean(bootstrap_t >= observed_t)

    p_value = min(
        1.0,
        2 * min(lower_tail, upper_tail)
    )

    # Bootstrap CI for mean paired difference
    boot_means = []

    for _ in range(n_boot):
        sample = rng.choice(diff, size=n, replace=True)
        boot_means.append(np.mean(sample))

    ci_low, ci_high = np.percentile(
        boot_means,
        [2.5, 97.5]
    )

    return {
        "mean_difference": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t_value": observed_t,
        "p_value": p_value,
    }


# ============================================================
# COLLECT ALL FINAL-MODEL RESULTS
# ============================================================

def collect_models():
    rows = []

    model_sets = [
        ("FM", FM_MODELS, "fm"),
        ("Dev-FM", DEV_MODELS, "dev"),
    ]

    for model_label, model_dict, expected_type in model_sets:

        for seed, model_dir in model_dict.items():

            verify_checkpoint(
                model_dir,
                expected_seed=seed,
                expected_type=expected_type,
            )

            if RUN_EVALUATION:
                run_blimp(model_dir)

            scores = read_results(model_dir)

            for phenomenon, accuracy in scores.items():
                rows.append(
                    {
                        "seed": seed,
                        "model": model_label,
                        "phenomenon": phenomenon,
                        "accuracy": accuracy,
                    }
                )

    return pd.DataFrame(rows)


# ============================================================
# STATISTICS FOR 12 BROAD PHENOMENA
# ============================================================

def calculate_phenomenon_statistics(df):
    rows = []

    for phenomenon in PHENOMENA:

        sub = df[
            df["phenomenon"] == phenomenon
        ]

        dev = (
            sub[sub["model"] == "Dev-FM"]
            .set_index("seed")
            .sort_index()["accuracy"]
        )

        fm = (
            sub[sub["model"] == "FM"]
            .set_index("seed")
            .sort_index()["accuracy"]
        )

        if not dev.index.equals(fm.index):
            raise ValueError(
                f"Seed pairing mismatch for {phenomenon}"
            )

        stats = bootstrap_paired_ttest(
            dev.values,
            fm.values,
            n_boot=N_BOOTSTRAPS,
            seed=BOOTSTRAP_SEED,
        )

        rows.append(
            {
                "phenomenon": phenomenon,
                "fm_mean": fm.mean(),
                "devfm_mean": dev.mean(),
                "mean_difference": stats["mean_difference"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "t_value": stats["t_value"],
                "p_value": stats["p_value"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MEAN OF 12 PHENOMENA
# Diagnostic only — not official BLiMP overall.
# ============================================================

def calculate_mean12(df):
    mean12 = (
        df.groupby(["seed", "model"])["accuracy"]
        .mean()
        .reset_index()
        .rename(
            columns={
                "accuracy": "mean_12_accuracy"
            }
        )
    )

    dev = (
        mean12[mean12["model"] == "Dev-FM"]
        .set_index("seed")
        .sort_index()["mean_12_accuracy"]
    )

    fm = (
        mean12[mean12["model"] == "FM"]
        .set_index("seed")
        .sort_index()["mean_12_accuracy"]
    )

    if not dev.index.equals(fm.index):
        raise ValueError(
            "Seed pairing mismatch for mean-of-12 score."
        )

    stats = bootstrap_paired_ttest(
        dev.values,
        fm.values,
        n_boot=N_BOOTSTRAPS,
        seed=BOOTSTRAP_SEED,
    )

    print("\n========================================")
    print("MEAN ACROSS 12 BROAD BLIMP PHENOMENA")
    print("NOT official BLiMP overall")
    print("========================================")

    print(f"FM mean:       {fm.mean() * 100:.2f}%")
    print(f"Dev-FM mean:   {dev.mean() * 100:.2f}%")

    print(
        f"Mean paired Δ: "
        f"{stats['mean_difference'] * 100:+.2f} percentage points"
    )

    print(
        f"95% CI:        "
        f"[{stats['ci_low'] * 100:+.2f}, "
        f"{stats['ci_high'] * 100:+.2f}]"
    )

    print(f"Bootstrap t:   {stats['t_value']:.3f}")
    print(f"Bootstrap p:   {stats['p_value']:.4f}")

    return mean12, stats


# ============================================================
# SIGNIFICANCE LABEL
# ============================================================

def significance_label(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


# ============================================================
# PLOT: MEAN OF 12
# ============================================================

def plot_mean12(mean12, overall_stats):
    fig, ax = plt.subplots(figsize=(5, 6))

    seeds = sorted(mean12["seed"].unique())

    for seed in seeds:
        seed_data = (
            mean12[
                mean12["seed"] == seed
            ]
            .set_index("model")
        )

        fm = (
            seed_data.loc["FM", "mean_12_accuracy"]
            * 100
        )

        dev = (
            seed_data.loc["Dev-FM", "mean_12_accuracy"]
            * 100
        )

        ax.plot(
            [0, 1],
            [fm, dev],
            marker="o",
            alpha=0.65,
        )

    means = (
        mean12.groupby("model")["mean_12_accuracy"]
        .mean()
        * 100
    )

    ax.scatter(
        [0, 1],
        [means["FM"], means["Dev-FM"]],
        s=100,
        zorder=5,
    )

    sig = significance_label(
        overall_stats["p_value"]
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [
            "Fleeting\nMemory",
            "Developmental\nFleeting Memory",
        ]
    )

    ax.set_ylabel(
        "Mean accuracy across 12 BLiMP phenomena (%)"
    )

    ax.set_title(
        f"100M BabyLM\n{sig}"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "blimp_mean12_comparison.png",
        dpi=300,
    )

    plt.close(fig)


# ============================================================
# PLOT: FIGURE-10 STYLE
# ============================================================

def plot_phenomena(df, stats):
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(12, 14)
    )

    axes = axes.flatten()

    # Same idea as paper:
    # sort by numerical Dev-FM improvement
    order = (
        stats
        .sort_values(
            "mean_difference",
            ascending=False,
        )["phenomenon"]
        .tolist()
    )

    for ax, phenomenon in zip(axes, order):

        sub = df[
            df["phenomenon"] == phenomenon
        ]

        seeds = sorted(sub["seed"].unique())

        for seed in seeds:
            seed_data = (
                sub[
                    sub["seed"] == seed
                ]
                .set_index("model")
            )

            fm = (
                seed_data.loc["FM", "accuracy"]
                * 100
            )

            dev = (
                seed_data.loc["Dev-FM", "accuracy"]
                * 100
            )

            ax.plot(
                [0, 1],
                [fm, dev],
                marker="o",
                alpha=0.65,
            )

        means = (
            sub.groupby("model")["accuracy"]
            .mean()
            * 100
        )

        ax.scatter(
            [0, 1],
            [means["FM"], means["Dev-FM"]],
            s=70,
            zorder=5,
        )

        stat_row = stats[
            stats["phenomenon"] == phenomenon
        ].iloc[0]

        sig = significance_label(
            stat_row["p_value"]
        )

        title = (
            phenomenon
            .replace("_", " ")
            .title()
        )

        ax.set_title(
            f"{title}\n{sig}"
        )

        ax.set_xticks([0, 1])

        ax.set_xticklabels(
            ["FM", "Dev-FM"]
        )

        ax.set_ylabel("Accuracy (%)")

    fig.suptitle(
        "100M BLiMP: Fleeting Memory vs Developmental Fleeting Memory",
        y=1.01,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "blimp_phenomena_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    df = collect_models()

    df.to_csv(
        OUTPUT_DIR / "blimp_results_long.csv",
        index=False,
    )

    stats = calculate_phenomenon_statistics(df)

    stats.to_csv(
        OUTPUT_DIR / "blimp_phenomenon_statistics.csv",
        index=False,
    )

    mean12, overall_stats = calculate_mean12(df)

    mean12.to_csv(
        OUTPUT_DIR / "blimp_mean12_by_seed.csv",
        index=False,
    )

    plot_mean12(
        mean12,
        overall_stats,
    )

    plot_phenomena(
        df,
        stats,
    )

    print("\nSaved all results to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()