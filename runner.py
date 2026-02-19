"""
Pipeline runner: reusable execution of the full pricing analysis pipeline.
Writes all artifacts to run_dir. Returns a summary dict for web/CLI.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np


def _json_serializable(obj):
    """Convert numpy/bool to native Python for JSON."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
import pandas as pd
from scipy.stats import skew

from cleaning import clean_data
from config import RunConfig, read_config, default_config
from data_audit import audit_columns, recommend_drops
from features import infer_target_column, prepare_model_data
from llm_schema import summarize_columns
from model import permutation_importance_by_column, route_columns, train_and_evaluate


def _feature_to_column(name: str) -> str:
    if "__" not in name:
        return name
    rest = name.split("__", 1)[1]
    if name.startswith("onehotencoder__"):
        return rest.split("_", 1)[0] if "_" in rest else rest
    if "tfidf" in name.lower() or "pipeline__" in name.lower():
        return "text_features"
    return rest


def build_ui_df(summary_df: pd.DataFrame, audit_df: pd.DataFrame) -> pd.DataFrame:
    summary_sub = summary_df[["column", "dtype", "missing_pct", "nunique", "description", "confidence"]].copy()
    audit_sub = audit_df[["column", "flags", "severity", "reason"]].copy()
    audit_sub = audit_sub.rename(columns={"reason": "reasons"})
    ui = summary_sub.merge(audit_sub, on="column", how="outer")
    ui["flags"] = ui["flags"].fillna("")
    ui["severity"] = ui["severity"].fillna(0).astype(int)
    ui["reasons"] = ui["reasons"].fillna("")
    ui["description"] = ui["description"].fillna("")
    ui["confidence"] = ui["confidence"].fillna("")
    return ui


def run_pipeline(
    df_raw: pd.DataFrame,
    input_name: str,
    run_dir: Path,
    user_target: str | None = None,
    dry_run: bool = False,
    random_state: int = 0,
    test_size: float = 0.2,
    use_openai: bool = False,
    keep_negatives: bool = False,
    force_log1p: bool = False,
    no_log1p: bool = False,
    no_report: bool = False,
    model_config: "RunConfig | None" = None,
) -> dict:
    """
    Run the full pricing analysis pipeline. Writes artifacts to run_dir.
    Returns dict with: selected_target, target_source, dropped_leakage, dropped_id_like,
    metrics, top_perm_importance, ui_summary_preview, paths, error (if any).

    Audit-only (dry_run=True): runs clean_data, summarize_columns, audit_columns,
    recommend_drops, build_ui_df, audit_report.pdf, route_columns (after drops),
    run_config.json. Skips only: model training, permutation importance, report.pdf.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "selected_target": None,
        "target_source": None,
        "dropped_leakage": [],
        "dropped_id_like": [],
        "metrics": {},
        "top_perm_importance": [],
        "ui_summary_preview": [],
        "paths": {},
        "log1p_used": False,
        "log1p_reason": "",
        "dry_run": dry_run,
        "error": None,
        "model_name": None,
        "model_params": None,
    }

    try:
        df_clean = clean_data(df_raw)
        target_name = infer_target_column(df_clean, user_target=user_target)
        target_source = "user" if user_target else "auto"
        summary["selected_target"] = target_name
        summary["target_source"] = target_source

        summary_df = summarize_columns(df_clean, sample_n=8, use_openai=use_openai)
        audit_df = audit_columns(df_clean, target_col=target_name)
        cfg = model_config if model_config is not None else read_config()
        drop_review_leakage = getattr(cfg, "drop_review_leakage", False)
        drops = recommend_drops(audit_df, target_name, drop_review_leakage=drop_review_leakage)
        summary["dropped_leakage"] = drops["leakage"]
        summary["dropped_id_like"] = drops["id_like"]

        if drops["all"]:
            df_clean = df_clean.drop(columns=drops["all"], errors="ignore")

        ui_df = build_ui_df(summary_df, audit_df)
        ui_csv = run_dir / "ui_summary.csv"
        ui_md = run_dir / "ui_summary.md"
        ui_df.to_csv(ui_csv, index=False)
        try:
            md = ui_df.to_markdown(index=False)
        except (AttributeError, ImportError):
            md = ui_df.to_string(index=False)
        ui_md.write_text(md, encoding="utf-8")

        safe_numeric_cols = drops.get("safe_numeric_cols", [])
        unsafe_numeric_cols = drops.get("unsafe_numeric_cols", [])

        numeric_fe = getattr(cfg, "numeric_feature_engineering", "basic")
        X, y, column_info, numeric_eng_info = prepare_model_data(
            df_clean,
            target_name=target_name,
            numeric_feature_engineering=numeric_fe,
            safe_numeric_cols=safe_numeric_cols,
        )
        engineered_numeric_cols = numeric_eng_info.get("engineered_numeric_cols", [])
        skipped_tech = numeric_eng_info.get("skipped_numeric_cols", {})
        skipped_numeric_cols = {
            **{c: "suspicious_id_like or potential_leakage" for c in unsafe_numeric_cols},
            **skipped_tech,
        }

        from audit_viz import create_audit_report_pdf
        create_audit_report_pdf(
            ui_df,
            output_path=run_dir / "audit_report.pdf",
            plots_dir=run_dir / "audit_plots",
            engineered_numeric_cols=engineered_numeric_cols,
            skipped_numeric_cols=skipped_numeric_cols,
        )

        if not keep_negatives:
            mask = y >= 0
            X = X.loc[mask].copy()
            y = y.loc[mask].copy()

        y_original = y.copy()

        target_transform = getattr(cfg, "target_transform", "auto")
        if force_log1p:
            target_transform = "log1p"
        elif no_log1p:
            target_transform = "none"

        if target_transform == "none":
            use_log1p = False
            log1p_reason = "disabled"
        elif target_transform == "log1p":
            use_log1p = not (y < 0).any()
            log1p_reason = "forced" if use_log1p else "skipped: negatives present"
        else:
            if (y < 0).any():
                use_log1p = False
                log1p_reason = "auto: negatives present"
            else:
                p50 = np.percentile(y.values, 50)
                p99 = np.percentile(y.values, 99)
                denom = max(float(p50), 1.0)
                p99_p50 = p99 / denom
                skewness_val = float(skew(y.values))
                use_log1p = skewness_val > 1.0 or p99_p50 > 20
                log1p_reason = (
                    f"auto: skewness={skewness_val:.2f}, p99/p50={p99_p50:.1f} "
                    f"=> {'log1p' if use_log1p else 'none'}"
                )

        summary["log1p_used"] = use_log1p
        summary["log1p_reason"] = log1p_reason

        if use_log1p:
            y = np.log1p(y)

        routing = route_columns(X)
        numeric_cols = routing["numeric"]
        low_card_cat_cols = routing["low_card_cat"]
        high_card_text_cols = routing["high_card_text"]

        model_params = cfg.get_params_for_training()
        model_name = cfg.model_name

        run_config = {
            "input_name": input_name,
            "selected_target": target_name,
            "target_source": target_source,
            "definite_drop": drops.get("definite_drop", drops["id_like"]),
            "review_drop": drops.get("review_drop", drops["leakage"]),
            "dropped_leakage": drops["leakage"],
            "dropped_id_like": drops["id_like"],
            "target_transform": target_transform,
            "log1p_used": use_log1p,
            "log1p_reason": log1p_reason,
            "engineered_numeric_cols": engineered_numeric_cols,
            "skipped_numeric_cols": skipped_numeric_cols,
            "model_name": model_name,
            "model_params": model_params,
            "numeric_cols": numeric_cols,
            "low_card_cat_cols": low_card_cat_cols,
            "high_card_text_cols": high_card_text_cols,
            "test_size": test_size,
            "random_state": random_state,
            "timestamp": datetime.now().isoformat(),
        }
        (run_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2, default=_json_serializable), encoding="utf-8"
        )

        summary["model_name"] = model_name
        summary["model_params"] = model_params
        summary["paths"]["ui_summary.csv"] = str(ui_csv)
        summary["paths"]["ui_summary.md"] = str(ui_md)
        summary["paths"]["audit_report.pdf"] = str(run_dir / "audit_report.pdf")
        summary["paths"]["run_config.json"] = str(run_dir / "run_config.json")

        worst = ui_df.sort_values(["severity", "missing_pct", "nunique"], ascending=[False, False, False]).head(20)
        summary["ui_summary_preview"] = worst.to_dict(orient="records")

        # Audit-only: stop here. Full pipeline continues with training, perm importance, report.pdf.
        if dry_run:
            artifacts_created = list(summary["paths"].keys())
            print("--- Dry run complete (no training) ---")
            for p in artifacts_created:
                print(f"  {p}")
            return summary

        if not numeric_cols and not low_card_cat_cols and not high_card_text_cols:
            summary["error"] = "No columns left for modeling."
            return summary

        (
            pipeline,
            train_metrics,
            test_metrics,
            importance_df,
            X_test,
            y_test,
            y_train,
            y_train_pred,
            y_test_pred,
        ) = train_and_evaluate(
            X, y,
            numeric_cols=numeric_cols,
            low_card_cat_cols=low_card_cat_cols,
            high_card_text_cols=high_card_text_cols,
            model_name=model_name,
            model_params=model_params,
            text_mode=cfg.text_mode,
            tfidf_max_features=cfg.tfidf_max_features,
            tfidf_min_df=getattr(cfg, "tfidf_min_df", 3),
            tfidf_ngram_range=getattr(cfg, "tfidf_ngram_range", None),
            tfidf_ngram_max=cfg.tfidf_ngram_max,
            test_size=test_size,
            random_state=random_state,
            y_in_log_space=use_log1p,
        )

        summary["metrics"] = {"train": train_metrics, "test": test_metrics}

        metrics_out = {
            "train": train_metrics,
            "test": test_metrics,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
        summary["paths"]["metrics.json"] = str(run_dir / "metrics.json")

        def to_dollars(ya: np.ndarray, in_log: bool) -> np.ndarray:
            if in_log:
                return np.expm1(np.clip(ya, None, 700))
            return np.asarray(ya)

        y_train_d = to_dollars(y_train.values, use_log1p)
        y_test_d = to_dollars(y_test.values, use_log1p)
        y_train_pred_d = to_dollars(y_train_pred, use_log1p)
        y_test_pred_d = to_dollars(y_test_pred, use_log1p)

        pred_rows = []
        for yt, yp, sp in [
            (y_train_d, y_train_pred_d, "train"),
            (y_test_d, y_test_pred_d, "test"),
        ]:
            res = yt - yp
            for i in range(len(yt)):
                pred_rows.append({"y_true": yt[i], "y_pred": yp[i], "residual": res[i], "split": sp})
        pred_df = pd.DataFrame(pred_rows)
        pred_df.to_csv(run_dir / "predictions.csv", index=False)
        summary["paths"]["predictions.csv"] = str(run_dir / "predictions.csv")

        from viz import create_residuals_report
        create_residuals_report(
            y_test_d,
            y_test_pred_d,
            X_test,
            output_path=run_dir / "residuals_report.pdf",
            plots_dir=run_dir / "residuals",
        )
        summary["paths"]["residuals_report.pdf"] = str(run_dir / "residuals_report.pdf")

        perm_df = permutation_importance_by_column(
            pipeline, X_test, y_test,
            cols=X_test.columns.tolist(),
            n_repeats=3,
            random_state=random_state,
        )
        perm_path = run_dir / "permutation_importance.csv"
        perm_df.to_csv(perm_path, index=False)
        summary["paths"]["permutation_importance.csv"] = str(perm_path)
        summary["top_perm_importance"] = perm_df.head(10).to_dict(orient="records")

        if not no_report:
            from viz import create_report_pdf
            top_5 = perm_df.head(5)["column"].tolist()
            create_report_pdf(
                X, y_original.values, top_5, perm_df,
                output_path=run_dir / "report.pdf",
                plots_dir=run_dir / "plots",
            )
            summary["paths"]["report.pdf"] = str(run_dir / "report.pdf")

        return summary

    except Exception as e:
        summary["error"] = str(e)
        raise


def prepare_data_for_optimization(
    df_raw: pd.DataFrame,
    user_target: str | None = None,
    random_state: int = 0,
    model_config: "RunConfig | None" = None,
) -> tuple[pd.DataFrame, pd.Series, dict, bool]:
    """
    Run clean, audit, prep. Returns (X, y, routing, use_log1p).
    y is in training space (log1p applied if needed).
    """
    df_clean = clean_data(df_raw)
    target_name = infer_target_column(df_clean, user_target=user_target)
    audit_df = audit_columns(df_clean, target_col=target_name)
    cfg = model_config if model_config is not None else read_config()
    drop_review_leakage = getattr(cfg, "drop_review_leakage", False)
    drops = recommend_drops(audit_df, target_name, drop_review_leakage=drop_review_leakage)
    if drops["all"]:
        df_clean = df_clean.drop(columns=drops["all"], errors="ignore")
    numeric_fe = getattr(cfg, "numeric_feature_engineering", "basic")
    safe_numeric_cols = drops.get("safe_numeric_cols", [])
    X, y, _, _ = prepare_model_data(
        df_clean,
        target_name=target_name,
        numeric_feature_engineering=numeric_fe,
        safe_numeric_cols=safe_numeric_cols,
    )
    mask = y >= 0
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()
    p50 = np.percentile(y.values, 50)
    p99 = np.percentile(y.values, 99)
    skew_ratio = p99 / max(float(p50), 1.0)
    skewness_val = float(skew(y.values))
    use_log1p = not (y < 0).any() and (skewness_val > 1.0 or skew_ratio > 20)
    if use_log1p:
        y = np.log1p(y)
    routing = route_columns(X)
    return X, y, routing, use_log1p
