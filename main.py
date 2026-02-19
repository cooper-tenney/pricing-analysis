"""
Entry point: load data, clean, summarize, audit, train, and report.
Works on any pricing CSV. Target is inferred or specified via --target.
CLI delegates to runner.run_pipeline.
Loads model config from artifacts/model_config.json if present.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from config import read_config
from run_utils import generate_run_id, write_latest_run_id
from runner import run_pipeline

ARTIFACTS_DIR = Path("artifacts")


def _wrap_list(items: list[str], width: int = 70, indent: str = "    ") -> str:
    if not items:
        return ""
    lines = []
    current = indent
    for name in items:
        sep = ", " if current.strip() else ""
        candidate = current + sep + name
        if len(candidate) > width and current.strip():
            lines.append(current.rstrip())
            current = indent + name
        else:
            current = candidate
    if current.strip():
        lines.append(current.rstrip())
    return "\n".join(lines)


def print_ui_summary(ui_df: pd.DataFrame, top_n: int, max_reason_len: int = 45) -> None:
    if top_n <= 0:
        return
    worst = ui_df.sort_values(
        ["severity", "missing_pct", "nunique"],
        ascending=[False, False, False],
    ).head(top_n)
    if worst.empty:
        print("  No columns to display.")
        return
    print(f"  {'column':<32} {'dtype':<12} {'sev':<3} {'miss%':<6} {'nunique':<7} flags | reasons")
    print("  " + "-" * 110)
    for _, row in worst.iterrows():
        col = str(row["column"])[:31]
        dtype = str(row["dtype"])[:11]
        sev = int(row["severity"])
        mpct = row["missing_pct"] if pd.notna(row["missing_pct"]) else 0
        nu = row["nunique"] if pd.notna(row["nunique"]) else 0
        flags = str(row["flags"])[:25] if pd.notna(row["flags"]) else ""
        reasons = str(row["reasons"])[:max_reason_len] if pd.notna(row["reasons"]) else ""
        print(f"  {col:<32} {dtype:<12} {sev:<3} {mpct:<6.1f} {nu:<7} {flags} | {reasons}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pricing regression: clean, featurize, train, evaluate.")
    parser.add_argument(
        "data_path",
        nargs="?",
        default="Messy Data - Closed Won - Routing.csv",
        help="Path to the pricing CSV (or PDF)",
    )
    parser.add_argument("--target", type=str, default=None, help="Target column name (else auto-inferred)")
    parser.add_argument("--dry-run", action="store_true", help="Run through clean/audit, exit before training")
    parser.add_argument("--keep-negatives", action="store_true", help="Keep rows with negative target (default: drop)")
    parser.add_argument(
        "--target-transform",
        choices=["auto", "none", "log1p"],
        default=None,
        help="Target transform: auto (heuristic), none, log1p",
    )
    parser.add_argument("--force-log1p", action="store_true", help="Force log1p transform (sets target_transform=log1p)")
    parser.add_argument("--no-log1p", action="store_true", help="Disable log1p transform (sets target_transform=none)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for test set")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed")
    parser.add_argument("--summarize-columns", action="store_true", help="Use LLM for column descriptions if OPENAI_API_KEY is set")
    parser.add_argument("--no-report", action="store_true", help="Disable matplotlib PDF report")
    parser.add_argument("--show-ui", type=int, default=15, metavar="N", help="Number of worst columns to print (0 = none)")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}", file=sys.stderr)
        return 1

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    run_id = generate_run_id()
    run_dir = ARTIFACTS_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_config = read_config(ARTIFACTS_DIR / "model_config.json")
    if args.target_transform is not None:
        model_config = model_config.model_copy(update={"target_transform": args.target_transform})
    if args.force_log1p:
        model_config = model_config.model_copy(update={"target_transform": "log1p"})
    if args.no_log1p:
        model_config = model_config.model_copy(update={"target_transform": "none"})
    params_source = "best_params (from Optuna)" if model_config.best_params else "param_space defaults"
    print(f"Model: {model_config.model_name} ({params_source})")

    print("Loading data...")
    suf = data_path.suffix.lower()
    if suf == ".pdf":
        try:
            from pdf_extract import extract_table_from_pdf
            df_raw = extract_table_from_pdf(data_path)
        except ImportError:
            print("Error: PDF support requires pdfplumber or camelot-py. Install: pip install pdfplumber", file=sys.stderr)
            return 1
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        df_raw = pd.read_csv(data_path)
    print(f"  Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns.")

    try:
        summary = run_pipeline(
            df_raw=df_raw,
            input_name=str(data_path.name),
            run_dir=run_dir,
            user_target=args.target,
            dry_run=args.dry_run,
            random_state=args.random_state,
            test_size=args.test_size,
            use_openai=bool(os.environ.get("OPENAI_API_KEY")) and args.summarize_columns,
            keep_negatives=args.keep_negatives,
            force_log1p=args.force_log1p,
            no_log1p=args.no_log1p,
            no_report=args.no_report,
            model_config=model_config,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if summary.get("error"):
        print(f"Error: {summary['error']}", file=sys.stderr)
        return 1

    write_latest_run_id(ARTIFACTS_DIR, run_id)

    print(f"Run ID: {run_id}")
    print(f"Selected target: {summary['selected_target']} ({summary['target_source']})")
    if summary["dropped_leakage"]:
        print(f"\nDropped leakage: {len(summary['dropped_leakage'])} columns")
        print(_wrap_list(summary["dropped_leakage"]))
    if summary["dropped_id_like"]:
        print(f"Dropped id_like: {len(summary['dropped_id_like'])} columns")
        print(_wrap_list(summary["dropped_id_like"]))

    ui_csv = run_dir / "ui_summary.csv"
    ui_md = run_dir / "ui_summary.md"
    print(f"\nSaved to {ui_csv}")
    print(f"Saved to {ui_md}")
    if ui_csv.exists() and args.show_ui > 0:
        ui_df = pd.read_csv(ui_csv)
        if "severity" in ui_df.columns:
            ui_df["severity"] = ui_df["severity"].fillna(0).astype(int)
            print(f"\n--- Faulty columns (top {args.show_ui} worst by severity) ---")
            print_ui_summary(ui_df, top_n=args.show_ui)

    for k, v in summary["paths"].items():
        print(f"  {k} -> {v}")
    print(f"\nArtifacts: {run_dir}")

    if summary["metrics"]:
        print("\n--- Train metrics ---")
        for k, v in summary["metrics"]["train"].items():
            print(f"  {k}: {v:.4f}")
        print("\n--- Test metrics ---")
        for k, v in summary["metrics"]["test"].items():
            print(f"  {k}: {v:.4f}")
        print("\n--- Top 5 permutation importance ---")
        for row in summary["top_perm_importance"][:5]:
            print(f"  {row['column']}: {row['importance']:.4f}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
