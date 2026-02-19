"""
FastAPI web UI for pricing analysis pipeline.
Upload CSV or PDF, run pipeline, view results and download artifacts.

Testing steps:
  uvicorn app:app --reload --port 8001
  open http://127.0.0.1:8001
  1. Upload CSV via / (index)
  2. Go to /models/config, run optimization (e.g. 10 trials)
  3. Go to /results to view run summary and optimization best params
"""

import io
import json
import threading
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import RunConfig, read_config, write_config
from run_utils import generate_run_id, read_latest_run_id, write_latest_run_id
from pdf_extract import extract_table_from_pdf
from runner import run_pipeline

app = FastAPI(title="Pricing Analysis")
BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
ARTIFACTS = BASE / "artifacts"
RUNS = ARTIFACTS / "runs"

UPLOADS.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))

ALLOWED_FILES = frozenset({".csv", ".pdf"})
SAFE_ARTIFACTS = frozenset({
    "ui_summary.csv", "ui_summary.md", "permutation_importance.csv",
    "report.pdf", "audit_report.pdf", "run_config.json",
    "metrics.json", "predictions.csv", "residuals_report.pdf",
})


def _get_run_dir(run_id: str) -> Path:
    run_dir = RUNS / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    return run_dir


def _safe_filename(name: str) -> bool:
    return name in SAFE_ARTIFACTS and ".." not in name


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    last_run_id = read_latest_run_id(ARTIFACTS)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "last_run_id": last_run_id}
    )


@app.post("/run", response_class=RedirectResponse)
async def run(
    request: Request,
    file: UploadFile = File(...),
    target: str = Form(""),
    dry_run: str = Form(""),
    summarize_columns: str = Form(""),
):
    target = target.strip() or None
    dry_run_bool = dry_run.lower() in ("1", "true", "on", "yes")
    use_openai = summarize_columns.lower() in ("1", "true", "on", "yes")

    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    suf = Path(file.filename).suffix.lower()
    if suf not in ALLOWED_FILES:
        raise HTTPException(status_code=400, detail="Upload CSV or PDF only")

    run_id = generate_run_id()
    upload_dir = UPLOADS / run_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / file.filename
    content = await file.read()
    upload_path.write_bytes(content)

    # Store last upload for /models/optimize
    last_upload = {
        "path": str(upload_path.resolve()),
        "run_id": run_id,
        "filename": file.filename,
    }
    (ARTIFACTS / "last_upload.json").write_text(json.dumps(last_upload), encoding="utf-8")

    try:
        if suf == ".csv":
            df_raw = pd.read_csv(upload_path)
        else:
            df_raw = extract_table_from_pdf(upload_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    model_config = read_config(ARTIFACTS / "model_config.json")
    try:
        summary = run_pipeline(
            df_raw=df_raw,
            input_name=file.filename,
            run_dir=run_dir,
            user_target=target,
            dry_run=dry_run_bool,
            random_state=0,
            test_size=0.2,
            use_openai=use_openai,
            keep_negatives=False,
            force_log1p=False,
            no_log1p=False,
            model_config=model_config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Persist run summary as single source of truth for summary pages
    run_summary_data = {
        "run_id": run_id,
        "model_name": summary.get("model_name"),
        "model_params": summary.get("model_params"),
        "train_metrics": summary.get("metrics", {}).get("train"),
        "test_metrics": summary.get("metrics", {}).get("test"),
        "timestamp": datetime.now().isoformat(),
        "selected_target": summary.get("selected_target"),
        "target_source": summary.get("target_source"),
        "dataset_shape": list(df_raw.shape),
        "log1p_used": summary.get("log1p_used", False),
        "log1p_reason": summary.get("log1p_reason", ""),
        "dropped_leakage": summary.get("dropped_leakage", []),
        "dropped_id_like": summary.get("dropped_id_like", []),
        "dry_run": summary.get("dry_run", False),
        "error": summary.get("error"),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "run_summary.json").write_text(
        json.dumps(run_summary_data, indent=2), encoding="utf-8"
    )

    write_latest_run_id(ARTIFACTS, run_id)
    return RedirectResponse(url="/results", status_code=303)


def _read_run_summary() -> dict | None:
    """Return persisted run summary from artifacts/run_summary.json, or None."""
    p = ARTIFACTS / "run_summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _get_last_run_id() -> str | None:
    """Return run_id from artifacts/latest.txt, fallback to last_upload.run_id."""
    run_id = read_latest_run_id(ARTIFACTS)
    if run_id:
        return run_id
    p = ARTIFACTS / "last_upload.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("run_id")
    except Exception:
        return None


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    """Render run summary from persisted data only. Single source of truth for summary pages."""
    run_summary = _read_run_summary()
    run_id = (run_summary.get("run_id") if run_summary else None) or _get_last_run_id()
    run_dir = RUNS / run_id if run_id else None

    # Load from persisted run artifacts
    ui_summary_preview = []
    ui_summary_columns = []
    top_perm_importance = []
    if run_dir:
        ui_csv = run_dir / "ui_summary.csv"
        if ui_csv.exists():
            try:
                ui_preview = pd.read_csv(ui_csv).head(20)
                ui_summary_preview = ui_preview.to_dict(orient="records")
                ui_summary_columns = list(ui_preview.columns)
            except Exception:
                pass
        perm_csv = run_dir / "permutation_importance.csv"
        if perm_csv.exists():
            try:
                perm_df = pd.read_csv(perm_csv)
                top_perm_importance = perm_df.head(10).to_dict(orient="records")
            except Exception:
                pass

    has_report = bool(run_dir and (run_dir / "report.pdf").exists())
    has_perm = bool(run_dir and (run_dir / "permutation_importance.csv").exists())
    has_config = bool(run_dir and (run_dir / "run_config.json").exists())

    engineered_numeric_cols = []
    skipped_numeric_cols = {}
    if has_config and run_dir:
        try:
            cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            engineered_numeric_cols = cfg.get("engineered_numeric_cols", [])
            skipped_numeric_cols = cfg.get("skipped_numeric_cols", {})
            if isinstance(skipped_numeric_cols, list):
                skipped_numeric_cols = dict(skipped_numeric_cols) if skipped_numeric_cols else {}
        except Exception:
            pass

    optuna_best = _read_optuna_best()
    opt_status = _read_opt_status()
    config = read_config(ARTIFACTS / "model_config.json")

    # Build summary from run_summary + loaded artifacts (matches result.html structure)
    summary = dict(run_summary) if run_summary else {"error": None}
    summary["ui_summary_preview"] = ui_summary_preview
    summary["ui_summary_columns"] = ui_summary_columns
    summary["top_perm_importance"] = top_perm_importance
    summary["metrics"] = {}
    if summary.get("train_metrics"):
        summary["metrics"]["train"] = summary["train_metrics"]
    if summary.get("test_metrics"):
        summary["metrics"]["test"] = summary["test_metrics"]

    return templates.TemplateResponse("results.html", {
        "request": request,
        "run_id": run_id,
        "summary": summary,
        "has_report": has_report,
        "has_perm": has_perm,
        "has_config": has_config,
        "engineered_numeric_cols": engineered_numeric_cols,
        "skipped_numeric_cols": skipped_numeric_cols,
        "optuna_best": optuna_best,
        "opt_status": opt_status,
        "config": config,
    })


@app.get("/download/{run_id}/{filename}")
async def download(run_id: str, filename: str):
    if not _safe_filename(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    run_dir = _get_run_dir(run_id)
    path = run_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, filename=filename)


@app.get("/view/{run_id}/{filename}")
async def view_pdf(run_id: str, filename: str):
    if filename not in ("report.pdf", "audit_report.pdf", "residuals_report.pdf"):
        raise HTTPException(status_code=400, detail="Only PDFs can be viewed")
    run_dir = _get_run_dir(run_id)
    path = run_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return Response(
        content=path.read_bytes(),
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"},
    )


@app.get("/download_zip/{run_id}")
async def download_zip(run_id: str):
    run_dir = _get_run_dir(run_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in SAFE_ARTIFACTS:
            p = run_dir / name
            if p.exists():
                zf.write(p, name)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=pricing_run_{run_id}.zip"},
    )


# --- Model settings & optimization (job manager) ---

JOBS: dict = {}  # run_id -> { "thread": Thread, "stop_event": Event, "status": str }
OPT_JOB_ID = "optuna"


def _write_opt_status(
    status: str,
    *,
    completed_trials: int | None = None,
    total_trials: int | None = None,
    best_score: float | None = None,
    error: str | None = None,
    last_updated: str | None = None,
):
    data = _read_opt_status() or {}
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
    if status in ("finished", "stopped", "failed", "stopping"):
        data["finished_at"] = data["last_updated"]
    (ARTIFACTS / "opt_status.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_opt_status_init(total_trials: int):
    _write_opt_status(
        "running",
        completed_trials=0,
        total_trials=total_trials,
    )


def _read_opt_status():
    p = ARTIFACTS / "opt_status.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_config_meta():
    p = ARTIFACTS / "model_config_meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_optuna_best():
    p = ARTIFACTS / "optuna_best.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _models_page_context(request: Request, error: str | None = None):
    opt_status = None
    p = ARTIFACTS / "opt_status.json"
    if p.exists():
        try:
            opt_status = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "request": request,
        "config": read_config(ARTIFACTS / "model_config.json"),
        "optuna_best": _read_optuna_best(),
        "config_meta": _read_config_meta(),
        "opt_status": opt_status,
        "error": error,
    }


def _run_optimization_thread(stop_event: threading.Event, target: str, upload_path: Path):
    """Background thread: load data, run optimization, update config. Uses stop_event to allow interruption."""
    from runner import prepare_data_for_optimization
    from optimize import run_optimization

    try:
        suf = upload_path.suffix.lower()
        if suf == ".csv":
            df_raw = pd.read_csv(upload_path)
        else:
            df_raw = extract_table_from_pdf(upload_path)
        config = read_config(ARTIFACTS / "model_config.json")
        X, y, routing, _ = prepare_data_for_optimization(
            df_raw,
            user_target=target.strip() or None,
            random_state=0,
            model_config=config,
        )
        best_params, best_score, _ = run_optimization(
            X, y, routing, config,
            artifacts_dir=ARTIFACTS,
            stop_event=stop_event,
        )
        config = config.model_copy(update={"best_params": best_params, "best_params_model": config.model_name})
        write_config(config, ARTIFACTS / "model_config.json")
    except Exception as e:
        _write_opt_status("failed", error=str(e))
    finally:
        if OPT_JOB_ID in JOBS:
            JOBS[OPT_JOB_ID]["status"] = "idle"


@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    """Model settings: status, optimization progress. Config form at /models/config."""
    return templates.TemplateResponse("models.html", _models_page_context(request))


@app.get("/models/config", response_class=HTMLResponse)
async def models_config_page(request: Request):
    """Model config form: save config, run optimization. Loads from artifacts/model_config.json."""
    return templates.TemplateResponse("models_config.html", _models_page_context(request))


@app.post("/models/save", response_class=RedirectResponse)
async def models_save(
    model_name: str = Form("xgboost"),
    text_mode: str = Form("tfidf"),
    tfidf_max_features: int = Form(5000),
    tfidf_ngram_max: int = Form(2),
    cv_folds: int = Form(3),
    optuna_trials: int = Form(25),
    optuna_timeout: str = Form(""),
):
    config = read_config(ARTIFACTS / "model_config.json")
    updates = {
        "model_name": model_name,
        "text_mode": text_mode,
        "tfidf_max_features": tfidf_max_features,
        "tfidf_ngram_max": tfidf_ngram_max,
        "cv_folds": cv_folds,
        "optuna_trials": optuna_trials,
        "optuna_timeout_sec": int(optuna_timeout) if optuna_timeout.strip() else None,
    }
    if config.model_name != model_name:
        updates["best_params"] = None
        updates["best_params_model"] = None
    config = config.model_copy(update=updates)
    write_config(config, ARTIFACTS / "model_config.json")
    meta = {"saved_at": datetime.now().isoformat()}
    (ARTIFACTS / "model_config_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return RedirectResponse(url="/models/config", status_code=303)


@app.post("/models/stop_opt", response_class=RedirectResponse)
async def models_stop_opt():
    job = JOBS.get(OPT_JOB_ID)
    if job and job.get("status") == "running":
        job["stop_event"].set()
        data = _read_opt_status() or {}
        data["status"] = "stopping"
        data["stop_requested_at"] = datetime.now().isoformat()
        data["last_updated"] = data["stop_requested_at"]
        data["finished_at"] = data["stop_requested_at"]
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "opt_status.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "stop_opt.flag").touch()
    return RedirectResponse(url="/models", status_code=303)


@app.get("/models/download_study")
async def models_download_study():
    path = ARTIFACTS / "optuna_study.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No optimization study found. Run optimization first.")
    return FileResponse(path, filename="optuna_study.csv")


@app.post("/models/optimize", response_class=RedirectResponse)
async def models_optimize(
    request: Request,
    target: str = Form(""),
):
    job = JOBS.get(OPT_JOB_ID)
    if job and job.get("status") == "running":
        return templates.TemplateResponse("models_config.html", _models_page_context(
            request, error="Optimization already running. Wait for it to finish or click Stop."
        ))
    last_path = ARTIFACTS / "last_upload.json"
    if not last_path.exists():
        raise HTTPException(status_code=400, detail="No dataset uploaded. Upload a CSV via the main page first.")
    last = json.loads(last_path.read_text(encoding="utf-8"))
    upload_path = Path(last["path"])
    if not upload_path.exists():
        raise HTTPException(status_code=400, detail="Uploaded file no longer exists.")
    config = read_config(ARTIFACTS / "model_config.json")

    stop_event = threading.Event()
    thread = threading.Thread(
        target=_run_optimization_thread,
        args=(stop_event, target or "", upload_path),
        daemon=True,
    )
    JOBS[OPT_JOB_ID] = {
        "thread": thread,
        "stop_event": stop_event,
        "status": "running",
    }
    _write_opt_status_init(config.optuna_trials)
    thread.start()
    return RedirectResponse(url="/models", status_code=303)
