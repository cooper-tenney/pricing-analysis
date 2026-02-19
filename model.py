"""
Regression pipeline: ColumnTransformer with numeric (StandardScaler),
low-card categorical (OneHotEncoder), and high-card/text (TF-IDF).
Supports multiple regressors: XGBoost, RF, ET, Ridge, Lasso, ElasticNet, Linear, SVR.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

LOW_CARD_NUNIQUE_THRESHOLD = 50
HIGH_CARD_NUNIQUE_THRESHOLD = 50
HIGH_CARD_MEDIAN_STR_LEN_THRESHOLD = 25


def route_columns(
    X: pd.DataFrame,
    high_card_text_cols: list[str] | None = None,
    low_card_threshold: int = LOW_CARD_NUNIQUE_THRESHOLD,
) -> dict[str, list[str]]:
    """
    Route columns into numeric, low_card_cat (<=threshold unique), high_card_text.
    High-card text detection: dtype object/category AND (nunique > 50 OR median string length >= 25).
    Optional high_card_text_cols override: columns in this list are forced to high_card_text.
    Returns {numeric_cols, low_card_cat_cols, high_card_text_cols}.
    """
    override_high = set(high_card_text_cols or [])
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_like = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    low_card_cat_cols = []
    high_card_text_cols_out = []
    for c in cat_like:
        if c in override_high:
            high_card_text_cols_out.append(c)
            continue
        ser = X[c]
        is_object_or_cat = ser.dtype == "object" or str(ser.dtype) == "category"
        if not is_object_or_cat:
            low_card_cat_cols.append(c)
            continue
        nunique = ser.nunique()
        median_str_len = ser.astype(str).str.len().median()
        median_str_len = float(median_str_len) if not pd.isna(median_str_len) else 0.0
        is_high_card = (
            nunique > HIGH_CARD_NUNIQUE_THRESHOLD
            or median_str_len >= HIGH_CARD_MEDIAN_STR_LEN_THRESHOLD
        )
        if is_high_card:
            high_card_text_cols_out.append(c)
        else:
            low_card_cat_cols.append(c)

    return {
        "numeric": numeric_cols,
        "low_card_cat": low_card_cat_cols,
        "high_card_text": high_card_text_cols_out,
    }


class TextConcatenator(BaseEstimator, TransformerMixin):
    """
    Concatenate multiple text columns into one string per row.
    Missing values become empty strings. Format: col1=val1 | col2=val2 | ...
    """

    def __init__(self, columns: list[str], sep: str = " | "):
        self.columns = columns
        self.sep = sep

    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return np.array(["_text_"])

    def transform(self, X):
        if not self.columns:
            return np.array([""] * len(X))
        if hasattr(X, "columns"):
            X = X[self.columns].copy()
        else:
            X = pd.DataFrame(X, columns=self.columns)
        if isinstance(X, pd.Series):
            X = X.to_frame()
        X = X.fillna("").astype(str)
        texts = []
        for i in range(len(X)):
            parts = []
            for col in self.columns:
                val = X.iloc[i][col].strip() if isinstance(X.iloc[i][col], str) else str(X.iloc[i][col])
                prefix = col.lower().replace(" ", "_").replace(":", "_").replace("-", "_") + "="
                parts.append(prefix + val)
            texts.append(self.sep.join(parts))
        return np.asarray(texts)


def build_preprocessor(
    numeric_cols: list[str],
    low_card_cat_cols: list[str],
    high_card_text_cols: list[str],
    text_mode: str = "tfidf",
    tfidf_max_features: int = 5000,
    tfidf_min_df: int = 3,
    tfidf_ngram_range: tuple[int, int] | None = None,
    tfidf_ngram_max: int = 2,
):
    """
    Build ColumnTransformer: numeric (StandardScaler), low_card_cat (OneHotEncoder),
    high_card_text: TextConcatenator + optional TfidfVectorizer.
    TF-IDF: lowercase=True, strip_accents="unicode", token_pattern ignores 1-char tokens.
    """
    ngram_range = tfidf_ngram_range if tfidf_ngram_range is not None else (1, tfidf_ngram_max)
    parts = []
    if numeric_cols:
        parts.append((StandardScaler(), numeric_cols))
    if low_card_cat_cols:
        parts.append((
            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            low_card_cat_cols,
        ))
    if high_card_text_cols and text_mode == "tfidf":
        tfidf_pipe = Pipeline([
            ("concat", TextConcatenator(high_card_text_cols)),
            ("tfidf", TfidfVectorizer(
                max_features=tfidf_max_features,
                min_df=tfidf_min_df,
                ngram_range=ngram_range,
                lowercase=True,
                strip_accents="unicode",
                token_pattern=r"(?u)\b\w{2,}\b",
            )),
        ])
        parts.append((tfidf_pipe, high_card_text_cols))

    if not parts:
        raise ValueError("At least one column group must be non-empty.")
    return make_column_transformer(*parts, remainder="drop", verbose_feature_names_out=True)


def _build_regressor(
    model_name: str,
    params: dict,
    random_state: int,
) -> tuple[object, str]:
    """Build sklearn/XGB/LGBM regressor. Returns (estimator, step_name)."""
    p = {k: v for k, v in params.items()}
    p["random_state"] = random_state

    if model_name == "xgboost":
        if XGBRegressor is None:
            raise ImportError("xgboost required. pip install xgboost")
        step = XGBRegressor(
            objective="reg:squarederror",
            max_depth=p.pop("max_depth", 4),
            n_estimators=p.pop("n_estimators", 400),
            learning_rate=p.pop("learning_rate", 0.05),
            subsample=p.pop("subsample", 0.8),
            colsample_bytree=p.pop("colsample_bytree", 0.8),
            min_child_weight=p.pop("min_child_weight", 1.0),
            reg_alpha=p.pop("reg_alpha", 0.1),
            reg_lambda=p.pop("reg_lambda", 1.0),
            gamma=p.pop("gamma", 0.0),
            tree_method="hist",
            **{k: v for k, v in p.items() if k in ("random_state",)},
        )
        return step, "xgbregressor"

    if model_name == "random_forest":
        step = RandomForestRegressor(
            n_estimators=p.pop("n_estimators", 200),
            max_depth=p.pop("max_depth", 12),
            min_samples_leaf=p.pop("min_samples_leaf", 2),
            max_features=p.pop("max_features", "sqrt"),
            n_jobs=-1,
            random_state=random_state,
        )
        return step, "randomforestregressor"

    if model_name == "extra_trees":
        step = ExtraTreesRegressor(
            n_estimators=p.pop("n_estimators", 200),
            max_depth=p.pop("max_depth", 12),
            min_samples_leaf=p.pop("min_samples_leaf", 2),
            max_features=p.pop("max_features", "sqrt"),
            n_jobs=-1,
            random_state=random_state,
        )
        return step, "extratreesregressor"

    if model_name == "ridge":
        step = Ridge(alpha=p.pop("alpha", 1.0), random_state=random_state)
        return step, "ridge"

    if model_name == "lasso":
        step = Lasso(alpha=p.pop("alpha", 1.0), random_state=random_state)
        return step, "lasso"

    if model_name == "elastic_net":
        step = ElasticNet(
            alpha=p.pop("alpha", 1.0),
            l1_ratio=p.pop("l1_ratio", 0.5),
            random_state=random_state,
        )
        return step, "elasticnet"

    if model_name == "linear":
        step = LinearRegression()
        return step, "linearregression"

    if model_name == "svr":
        step = SVR(
            C=p.pop("C", 1.0),
            epsilon=p.pop("epsilon", 0.1),
            gamma=p.pop("gamma", "scale"),
        )
        return step, "svr"

    if model_name == "lgbm_optional":
        if not LGBM_AVAILABLE:
            raise ImportError("lightgbm required for lgbm_optional. pip install lightgbm")
        step = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=p.pop("n_estimators", 400),
            max_depth=p.pop("max_depth", 4),
            learning_rate=p.pop("learning_rate", 0.05),
            subsample=p.pop("subsample", 0.8),
            colsample_bytree=p.pop("colsample_bytree", 0.8),
            n_jobs=-1,
            random_state=random_state,
        )
        return step, "lgbmregressor"

    raise ValueError(f"Unknown model_name: {model_name}")


def build_regression_pipeline(
    numeric_cols: list[str],
    low_card_cat_cols: list[str],
    high_card_text_cols: list[str],
    model_name: str = "xgboost",
    model_params: dict | None = None,
    text_mode: str = "tfidf",
    tfidf_max_features: int = 5000,
    tfidf_min_df: int = 3,
    tfidf_ngram_range: tuple[int, int] | None = None,
    tfidf_ngram_max: int = 2,
    random_state: int = 0,
):
    """
    Build regression pipeline: preprocessor + regressor.
    model_name: xgboost, random_forest, extra_trees, elastic_net, ridge, lasso, linear, svr, lgbm_optional.
    """
    model_params = model_params or {}
    preprocessor = build_preprocessor(
        numeric_cols, low_card_cat_cols, high_card_text_cols,
        text_mode=text_mode,
        tfidf_max_features=tfidf_max_features,
        tfidf_min_df=tfidf_min_df,
        tfidf_ngram_range=tfidf_ngram_range,
        tfidf_ngram_max=tfidf_ngram_max,
    )
    regressor, step_name = _build_regressor(model_name, model_params, random_state)
    return Pipeline([
        ("columntransformer", preprocessor),
        (step_name, regressor),
    ])


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics: MAE, RMSE, R²."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def get_feature_names_from_pipeline(pipeline) -> list[str]:
    """Get interpretable feature names from fitted ColumnTransformer."""
    ct = pipeline.named_steps["columntransformer"]
    return ct.get_feature_names_out().tolist()


def _get_regressor_step_name(pipeline) -> str:
    """Return the regressor step name (last step)."""
    steps = list(pipeline.named_steps.keys())
    # First is columntransformer, second is regressor
    for s in reversed(steps):
        if s != "columntransformer":
            return s
    return steps[-1]


def _get_model_importance(pipeline) -> np.ndarray | None:
    """Extract feature importances or coef from pipeline. Returns None if not available."""
    step_name = _get_regressor_step_name(pipeline)
    reg = pipeline.named_steps[step_name]
    if hasattr(reg, "feature_importances_"):
        return reg.feature_importances_
    if hasattr(reg, "coef_"):
        return np.abs(reg.coef_).ravel()
    return None


def permutation_importance_by_column(
    pipeline,
    X_eval: pd.DataFrame,
    y_eval: pd.Series | np.ndarray,
    cols: list[str],
    n_repeats: int = 3,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Compute permutation importance on raw input columns.
    Returns DataFrame with column, importance (mean RMSE increase), std.
    """
    rng = np.random.default_rng(random_state)
    y_eval = np.asarray(y_eval)
    baseline_pred = pipeline.predict(X_eval)
    baseline_rmse = np.sqrt(mean_squared_error(y_eval, baseline_pred))

    results = []
    for col in cols:
        if col not in X_eval.columns:
            continue
        deltas = []
        for _ in range(n_repeats):
            X_perm = X_eval.copy()
            perm_idx = rng.permutation(len(X_perm))
            X_perm[col] = X_eval[col].iloc[perm_idx].values
            pred = pipeline.predict(X_perm)
            rmse_perm = np.sqrt(mean_squared_error(y_eval, pred))
            deltas.append(rmse_perm - baseline_rmse)
        results.append({
            "column": col,
            "importance": np.mean(deltas),
            "std": np.std(deltas),
        })
    df = pd.DataFrame(results)
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    low_card_cat_cols: list[str],
    high_card_text_cols: list[str],
    model_name: str = "xgboost",
    model_params: dict | None = None,
    text_mode: str = "tfidf",
    tfidf_max_features: int = 5000,
    tfidf_min_df: int = 3,
    tfidf_ngram_range: tuple[int, int] | None = None,
    tfidf_ngram_max: int = 2,
    test_size: float = 0.2,
    random_state: int = 0,
    y_in_log_space: bool = True,
):
    """
    Split data, fit pipeline, evaluate.
    Returns (pipeline, train_metrics, test_metrics, importance_df, X_test, y_test,
             y_train, y_train_pred, y_test_pred).
    importance_df uses model feature_importances_ or |coef_| when available; else equal weights.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pipeline = build_regression_pipeline(
        numeric_cols, low_card_cat_cols, high_card_text_cols,
        model_name=model_name,
        model_params=model_params or {},
        text_mode=text_mode,
        tfidf_max_features=tfidf_max_features,
        tfidf_min_df=tfidf_min_df,
        tfidf_ngram_range=tfidf_ngram_range,
        tfidf_ngram_max=tfidf_ngram_max,
        random_state=random_state,
    )
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_metrics = evaluate_regression(y_train.values, y_train_pred)
    test_metrics = evaluate_regression(y_test.values, y_test_pred)

    def evaluate_regression_dollars(y_true: np.ndarray, y_pred: np.ndarray, in_log: bool) -> dict[str, float]:
        if in_log:
            y_true = np.expm1(np.clip(y_true, None, 700))
            y_pred = np.expm1(np.clip(y_pred, None, 700))
        return {
            "MAE_$": mean_absolute_error(y_true, y_pred),
            "RMSE_$": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2_$": r2_score(y_true, y_pred),
        }

    train_metrics.update(evaluate_regression_dollars(y_train.values, y_train_pred, y_in_log_space))
    test_metrics.update(evaluate_regression_dollars(y_test.values, y_test_pred, y_in_log_space))

    feature_names = get_feature_names_from_pipeline(pipeline)
    imp = _get_model_importance(pipeline)
    if imp is not None and len(imp) == len(feature_names):
        importance_df = pd.DataFrame({"feature": feature_names, "importance": imp})
    else:
        importance_df = pd.DataFrame({"feature": feature_names, "importance": np.ones(len(feature_names)) / max(len(feature_names), 1)})
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

    return (
        pipeline,
        train_metrics,
        test_metrics,
        importance_df,
        X_test,
        y_test,
        y_train,
        y_train_pred,
        y_test_pred,
    )
