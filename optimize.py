"""
Optuna Bayesian optimization for model hyperparameters.
Uses KFold CV and RMSE. Writes optuna_study.csv and optuna_best.json.
Supports opt_status.json updates and stop_event (threading.Event) for interruption.
"""

import json
from datetime import datetime
from pathlib import Path
from threading import Event

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import cross_val_score

from config import RunConfig
from model import build_regression_pipeline


def _write_opt_status(
    artifacts_dir: Path,
    status: str,
    *,
    completed_trials: int = 0,
    total_trials: int = 0,
    best_score: float | None = None,
    error: str | None = None,
    finished_at: str | None = None,
    last_updated: str | None = None,
):
    data = {}
    status_path = artifacts_dir / "opt_status.json"
    if status_path.exists():
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    data["status"] = status
    data["last_updated"] = last_updated or datetime.now().isoformat()
    if completed_trials is not None:
        data["completed_trials"] = completed_trials
    if total_trials is not None:
        data["total_trials"] = total_trials
    if best_score is not None:
        data["best_score"] = best_score
    if error is not None:
        data["error"] = error
    if finished_at is not None:
        data["finished_at"] = finished_at
    if status == "running" and "started_at" not in data:
        data["started_at"] = data["last_updated"]
    status_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def suggest_params(trial: optuna.Trial, model_name: str, space_dict: dict) -> dict:
    """Suggest hyperparameters for a trial based on model and space dict."""
    out = {}
    if model_name == "xgboost":
        out["max_depth"] = trial.suggest_int("max_depth", 3, 10)
        out["n_estimators"] = trial.suggest_int("n_estimators", 400, 1500)
        out["learning_rate"] = trial.suggest_float("learning_rate", 0.02, 0.15, log=True)
        out["min_child_weight"] = trial.suggest_float("min_child_weight", 1.0, 20.0, log=True)
        out["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        out["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        out["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True)
        out["reg_lambda"] = trial.suggest_float("reg_lambda", 0.5, 20.0, log=True)
        out["gamma"] = trial.suggest_float("gamma", 0.0, 5.0)
    elif model_name in ("random_forest", "extra_trees"):
        out["n_estimators"] = trial.suggest_int("n_estimators", 50, 400)
        out["max_depth"] = trial.suggest_int("max_depth", 4, 24)
        out["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 10)
        out["max_features"] = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0])
    elif model_name == "ridge":
        out["alpha"] = trial.suggest_float("alpha", 1e-4, 1e4, log=True)
    elif model_name == "lasso":
        out["alpha"] = trial.suggest_float("alpha", 1e-4, 1e4, log=True)
    elif model_name == "elastic_net":
        out["alpha"] = trial.suggest_float("alpha", 1e-4, 1e4, log=True)
        out["l1_ratio"] = trial.suggest_float("l1_ratio", 0.1, 1.0)
    elif model_name == "linear":
        pass
    elif model_name == "svr":
        out["C"] = trial.suggest_float("C", 0.1, 1000.0, log=True)
        out["epsilon"] = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)
        out["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
    elif model_name == "lgbm_optional":
        out["n_estimators"] = trial.suggest_int("n_estimators", 100, 600)
        out["max_depth"] = trial.suggest_int("max_depth", 3, 12)
        out["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        out["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        out["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    return out


def objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    routing: dict,
    config: RunConfig,
    stop_event: Event | None = None,
) -> float:
    """Optuna objective: build pipeline, run CV, return RMSE. Checks stop_event at start."""
    if stop_event and stop_event.is_set():
        raise optuna.TrialPruned()
    params = suggest_params(trial, config.model_name, config.param_space)
    try:
        pipeline = build_regression_pipeline(
            numeric_cols=routing["numeric"],
            low_card_cat_cols=routing["low_card_cat"],
            high_card_text_cols=routing["high_card_text"],
            model_name=config.model_name,
            model_params=params,
            text_mode=config.text_mode,
            tfidf_max_features=config.tfidf_max_features,
            tfidf_min_df=getattr(config, "tfidf_min_df", 3),
            tfidf_ngram_range=getattr(config, "tfidf_ngram_range", None),
            tfidf_ngram_max=config.tfidf_ngram_max,
            random_state=config.random_state,
        )
        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=config.cv_folds,
            scoring="neg_root_mean_squared_error",
        )
        return float(-scores.mean())
    except Exception as e:
        raise optuna.TrialPruned() from e


def run_optimization(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    routing: dict,
    config: RunConfig,
    artifacts_dir: Path | str = "artifacts",
    stop_event: Event | None = None,
) -> tuple[dict, float, pd.DataFrame]:
    """
    Run Optuna study. Returns (best_params, best_score, study_summary_df).
    Writes artifacts/optuna_study.csv, optuna_best.json, and opt_status.json.
    If stop_event is set (from threading.Event), stops at next trial boundary.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stop_flag_path = artifacts_dir / "stop_opt.flag"

    def remove_stop_flag():
        if stop_flag_path.exists():
            stop_flag_path.unlink(missing_ok=True)

    remove_stop_flag()

    def should_stop() -> bool:
        if stop_event and stop_event.is_set():
            return True
        if stop_flag_path.exists():
            return True
        return False

    def status_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        best = study.best_value if study.best_trial else None
        if should_stop():
            remove_stop_flag()
            finished = datetime.now().isoformat()
            _write_opt_status(
                artifacts_dir,
                "stopped",
                completed_trials=completed,
                total_trials=config.optuna_trials,
                best_score=best,
                finished_at=finished,
            )
            study.stop()
            return
        _write_opt_status(
            artifacts_dir,
            "running",
            completed_trials=completed,
            total_trials=config.optuna_trials,
            best_score=best,
        )

    _write_opt_status(
        artifacts_dir,
        "running",
        completed_trials=0,
        total_trials=config.optuna_trials,
    )

    def obj_wrapper(trial):
        return objective(trial, X, y, routing, config, stop_event=stop_event)

    try:
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=config.random_state, n_startup_trials=5),
            pruner=MedianPruner(),
        )
        study.optimize(
            obj_wrapper,
            n_trials=config.optuna_trials,
            timeout=config.optuna_timeout_sec,
            show_progress_bar=False,
            callbacks=[status_callback],
        )

        best_params = study.best_params
        best_score = study.best_value
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        status_path = artifacts_dir / "opt_status.json"
        current_status = "finished"
        if status_path.exists():
            try:
                current = json.loads(status_path.read_text(encoding="utf-8"))
                if current.get("status") == "stopped":
                    current_status = "stopped"
            except Exception:
                pass
        _write_opt_status(
            artifacts_dir,
            current_status,
            completed_trials=completed,
            total_trials=config.optuna_trials,
            best_score=best_score,
            finished_at=datetime.now().isoformat(),
        )

    except Exception as e:
        _write_opt_status(
            artifacts_dir,
            "failed",
            error=str(e),
            finished_at=datetime.now().isoformat(),
        )
        raise

    # Trial table
    rows = []
    for t in study.trials:
        rows.append({
            "number": t.number,
            "value": t.value,
            "state": str(t.state),
            **t.params,
        })
    study_df = pd.DataFrame(rows)
    study_path = artifacts_dir / "optuna_study.csv"
    study_df.to_csv(study_path, index=False)

    # Best params
    best_data = {"best_params": best_params, "best_score": best_score}
    best_path = artifacts_dir / "optuna_best.json"
    best_path.write_text(json.dumps(best_data, indent=2), encoding="utf-8")

    return best_params, best_score, study_df
