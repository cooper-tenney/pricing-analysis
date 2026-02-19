"""
Model configuration schema and persistence.
RunConfig defines model selection, text settings, and Optuna tuning params.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


MODEL_NAMES = (
    "xgboost",
    "random_forest",
    "extra_trees",
    "elastic_net",
    "ridge",
    "lasso",
    "linear",
    "svr",
    "lgbm_optional",
)

TEXT_MODES = ("tfidf", "none")

DEFAULT_PARAM_SPACE: dict[str, dict[str, Any]] = {
    "xgboost": {
        "max_depth": 3,
        "n_estimators": 400,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "gamma": 0.0,
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    "extra_trees": {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    "elastic_net": {"alpha": 1.0, "l1_ratio": 0.5},
    "ridge": {"alpha": 1.0},
    "lasso": {"alpha": 1.0},
    "linear": {},
    "svr": {"C": 1.0, "epsilon": 0.1, "gamma": "scale"},
    "lgbm_optional": {},
}


def default_config() -> "RunConfig":
    return RunConfig(
        model_name="xgboost",
        text_mode="tfidf",
        tfidf_max_features=15000,
        tfidf_min_df=3,
        tfidf_ngram_range=(1, 2),
        tfidf_ngram_max=2,
        cv_folds=3,
        optuna_trials=25,
        optuna_timeout_sec=None,
        metric="rmse",
        random_state=0,
        param_space=DEFAULT_PARAM_SPACE["xgboost"].copy(),
        best_params=None,
        target_transform="auto",
        numeric_feature_engineering="basic",
        drop_review_leakage=True,
    )


class RunConfig(BaseModel):
    model_name: str = Field(default="xgboost", description="One of: xgboost, random_forest, extra_trees, elastic_net, ridge, lasso, linear, svr, lgbm_optional")
    text_mode: str = Field(default="tfidf", description="tfidf or none")
    tfidf_max_features: int = Field(default=15000, ge=100, le=50000)
    tfidf_min_df: int = Field(default=3, ge=1, le=100)
    tfidf_ngram_range: tuple[int, int] = Field(default=(1, 2), description="(min_n, max_n) for ngrams")

    @field_validator("tfidf_ngram_range", mode="before")
    @classmethod
    def coerce_ngram_range(cls, v):
        if isinstance(v, list) and len(v) == 2:
            return (int(v[0]), int(v[1]))
        return v
    tfidf_ngram_max: int = Field(default=2, ge=1, le=3)
    target_transform: str = Field(default="auto", description="auto | none | log1p")
    numeric_feature_engineering: str = Field(default="basic", description="none | basic")
    drop_review_leakage: bool = Field(default=True, description="Auto-include review_drop columns in drops")
    cv_folds: int = Field(default=3, ge=2, le=10)
    optuna_trials: int = Field(default=25, ge=5, le=200)
    optuna_timeout_sec: int | None = Field(default=None, ge=None)
    metric: str = Field(default="rmse")
    random_state: int = Field(default=0)
    param_space: dict[str, Any] = Field(default_factory=dict)
    best_params: dict[str, Any] | None = Field(default=None)
    best_params_model: str | None = Field(default=None, description="Model name that best_params were optimized for")

    model_config = ConfigDict(extra="forbid")

    def get_params_for_training(self) -> dict[str, Any]:
        """Return best_params if set and for current model, else param_space."""
        if self.best_params and self.best_params_model == self.model_name:
            return self.best_params.copy()
        base = DEFAULT_PARAM_SPACE.get(self.model_name, {})
        return {**base, **(self.param_space or {})}


def _config_path(path: Path | str | None) -> Path:
    return Path(path) if path else Path("artifacts/model_config.json")


def read_config(path: str | Path | None = None) -> RunConfig:
    """Load RunConfig from JSON. Return default if file missing or invalid."""
    p = _config_path(path)
    if not p.exists():
        return default_config()
    try:
        data = __import__("json").loads(p.read_text(encoding="utf-8"))
        return RunConfig.model_validate(data)
    except Exception:
        return default_config()


def write_config(config: RunConfig, path: str | Path | None = None) -> Path:
    """Write RunConfig to JSON."""
    p = Path(_config_path(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return p
