"""Model factories. Preprocessing belongs in each training fold's pipeline."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def logistic_pipeline(settings: dict[str, Any], seed: int) -> Pipeline:
    """Build regularized logistic regression with fold-fitted median imputation/scaling."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(settings.get("C", 1.0)),
                    class_weight=settings.get("class_weight", "balanced"),
                    max_iter=int(settings.get("max_iter", 2000)),
                    random_state=seed,
                ),
            ),
        ]
    )


def random_forest_pipeline(settings: dict[str, Any], seed: int) -> Pipeline:
    """Build an intentionally constrained forest; it is a comparison, not a default."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=int(settings.get("n_estimators", 400)),
                    max_depth=int(settings.get("max_depth", 4)),
                    min_samples_leaf=int(settings.get("min_samples_leaf", 5)),
                    class_weight=settings.get("class_weight", "balanced"),
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
