from pathlib import Path
import argparse

try:
    import pandas as pd
    import seaborn as sns
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency. Install with: python3 -m pip install pandas seaborn matplotlib"
    ) from exc


FEATURE_ORDER = ["brightness", "contrast", "colorfulness", "sharpness", "noise"]
FEATURE_TRANSLATION = {
    "brightness": "Helligkeit",
    "contrast": "Kontrast",
    "colorfulness": "Farbigkeit",
    "sharpness": "Schärfe",
    "noise": "Rauschen",
}
SPLIT_ORDER = ["bottom_10", "top_10"]
SOURCE_ORDER = ["Handcrafted", "CLIP"]
SOURCE_COLORS = {"Handcrafted": "#1f77b4", "CLIP": "#d62728"}
DATASET_LABELS = {"ava": "AVA", "para": "PARA", "tad66k": "TAD66K"}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[4]
    phase2_root = project_root / "analysis" / "phase02"
    parser = argparse.ArgumentParser(description="Plot Top/Bottom SRCC (Handcrafted vs CLIP).")
    parser.add_argument("--dataset", type=str, default="ava", choices=["ava", "para", "tad66k"])
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--plcc-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--phase2-root", type=Path, default=phase2_root)
    return parser.parse_args()


def _default_input_csv(phase2_root: Path, dataset: str) -> Path:
    return phase2_root / "data" / dataset / "top_bottom" / f"{dataset}_top_bottom_srcc.csv"


def _default_out_dir(phase2_root: Path, dataset: str) -> Path:
    return phase2_root / "figures" / dataset / "top_bottom"


def _default_plcc_csv(phase2_root: Path, dataset: str) -> Path:
    return phase2_root / "data" / dataset / "top_bottom" / f"{dataset}_top_bottom_plcc.csv"


def _infer_plcc_csv(args: argparse.Namespace, dataset: str, input_csv: Path) -> Path:
    if args.plcc_csv is not None:
        return args.plcc_csv
    if args.input_csv is None:
        return _default_plcc_csv(args.phase2_root, dataset)
    if input_csv.name.endswith("_srcc.csv"):
        return input_csv.with_name(input_csv.name.replace("_srcc.csv", "_plcc.csv"))
    return _default_plcc_csv(args.phase2_root, dataset)


def _prepare_long(df: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "feature", "srcc_handcrafted", "srcc_clip"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in SRCC CSV: {sorted(missing)}")

    df = df.copy()
    df = df[df["split"].isin(SPLIT_ORDER)]
    df = df[df["feature"].isin(FEATURE_ORDER)]
    df["split"] = pd.Categorical(df["split"], categories=SPLIT_ORDER, ordered=True)
    df["feature"] = pd.Categorical(df["feature"], categories=FEATURE_ORDER, ordered=True)
    df = df.sort_values(["split", "feature"])

    long_df = df.melt(
        id_vars=["split", "feature"],
        value_vars=["srcc_handcrafted", "srcc_clip"],
        var_name="source",
        value_name="srcc",
    )
    long_df["source"] = long_df["source"].map(
        {"srcc_handcrafted": "Handcrafted", "srcc_clip": "CLIP"}
    )
    long_df["source"] = pd.Categorical(long_df["source"], categories=SOURCE_ORDER, ordered=True)
    long_df["feature_de"] = long_df["feature"].astype(str).map(FEATURE_TRANSLATION)
    return long_df


def main() -> None:
    args = parse_args()
    dataset = args.dataset
    dataset_label = DATASET_LABELS[dataset]
    input_csv = args.input_csv or _default_input_csv(args.phase2_root, dataset)
    plcc_csv = _infer_plcc_csv(args, dataset, input_csv)
    out_dir = args.out_dir or _default_out_dir(args.phase2_root, dataset)
    prefix = args.prefix or f"{dataset}_top_bottom_srcc_handcrafted_vs_clip"

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    long_df = _prepare_long(df)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{dataset_label}] SRCC plot source: {input_csv}")
    print(df[["split", "feature", "srcc_handcrafted", "srcc_clip", "delta_srcc_clip_minus_handcrafted"]].to_string(index=False))

    plcc_cols = {"split", "feature", "plcc_handcrafted", "plcc_clip", "delta_plcc_clip_minus_handcrafted"}
    if plcc_cols.issubset(df.columns):
        plcc_df = df
    elif plcc_csv.exists():
        plcc_df = pd.read_csv(plcc_csv)
    else:
        plcc_df = None

    if plcc_df is not None and plcc_cols.issubset(plcc_df.columns):
        print(f"\n[{dataset_label}] PLCC source: {plcc_csv if plcc_df is not df else input_csv}")
        print(plcc_df[["split", "feature", "plcc_handcrafted", "plcc_clip", "delta_plcc_clip_minus_handcrafted"]].to_string(index=False))
    else:
        print(f"\n[{dataset_label}] PLCC source not found or incomplete: {plcc_csv}")

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    panels = [("bottom_10", "Schlechteste 10%"), ("top_10", "Beste 10%")]
    for idx, (split_key, split_title) in enumerate(panels):
        ax = axes[idx]
        sub = long_df[long_df["split"] == split_key]
        sns.barplot(
            data=sub,
            x="feature_de",
            y="srcc",
            hue="source",
            hue_order=SOURCE_ORDER,
            palette=SOURCE_COLORS,
            ax=ax,
        )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(split_title, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Technische Merkmale", fontsize=13, fontweight="bold")
        ax.set_ylabel("SRCC" if idx == 0 else "", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", labelsize=12, rotation=20)
        ax.tick_params(axis="y", labelsize=12)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=2, fontsize=10)
        if idx == 1:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        else:
            legend = ax.legend(title="Quelle")
            if legend is not None:
                legend.get_title().set_fontsize(12)
                for text in legend.get_texts():
                    text.set_fontsize(12)

    fig.suptitle(
        f"{dataset_label}: Top/Bottom 10% SRCC (Handcrafted vs CLIP)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf_path = out_dir / f"{prefix}.pdf"
    png_path = out_dir / f"{prefix}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
