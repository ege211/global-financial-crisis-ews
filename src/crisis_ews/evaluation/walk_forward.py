"""Expanding-window annual validation with strict train-only preprocessing."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pandas as pd

from crisis_ews.evaluation.metrics import classification_metrics


@dataclass(frozen=True)
class TemporalFold:
    train_end_year: int
    test_year: int


def expanding_window_folds(
    years: pd.Series,
    minimum_train_years: int,
    test_start_year: int,
    test_end_year: int | None = None,
) -> Iterator[TemporalFold]:
    """Yield one-year test folds with every training year strictly before the test year."""
    unique_years = sorted(pd.Series(years).dropna().astype(int).unique())
    for test_year in unique_years:
        if test_end_year is not None and test_year > test_end_year:
            continue
        available_train_years = [year for year in unique_years if year < test_year]
        if test_year >= test_start_year and len(available_train_years) >= minimum_train_years:
            yield TemporalFold(train_end_year=max(available_train_years), test_year=test_year)


def evaluate_walk_forward(
    data: pd.DataFrame,
    features: list[str],
    model_factory: Callable[[], object],
    minimum_train_years: int,
    test_start_year: int,
    threshold: float,
    model_name: str,
    feature_version: str = "core-v1",
    test_end_year: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a fresh pipeline for each fold and return auditable predictions and metrics."""
    required = {"country_code", "year", "crisis_within_horizon", *features}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Modeling data lacks columns: {sorted(missing)}")
    predictions: list[pd.DataFrame] = []
    for fold in expanding_window_folds(data["year"], minimum_train_years, test_start_year, test_end_year):
        train = data.loc[data["year"] <= fold.train_end_year]
        test = data.loc[data["year"] == fold.test_year]
        y_train = train["crisis_within_horizon"].astype(int)
        if y_train.nunique() < 2:
            continue  # A classifier cannot be fit until both classes have occurred historically.
        model = model_factory()
        model.fit(train[features], y_train)
        probability = model.predict_proba(test[features])[:, 1]
        output = test[["country_code", "year", "crisis_within_horizon"]].copy()
        output["test_year"] = fold.test_year
        output["y_true"] = output["crisis_within_horizon"]
        output["predicted_probability"] = probability
        output["predicted_class"] = (probability >= threshold).astype(int)
        output["model"] = model_name
        output["model_name"] = model_name
        output["training_start_year"] = int(train["year"].min())
        output["training_end_year"] = fold.train_end_year
        output["feature_version"] = feature_version
        predictions.append(output)
    if not predictions:
        raise ValueError("No evaluable folds: check chronology coverage, labels, and training settings")
    prediction_frame = pd.concat(predictions, ignore_index=True)
    summary = classification_metrics(
        prediction_frame["crisis_within_horizon"].to_numpy(),
        prediction_frame["predicted_probability"].to_numpy(),
        threshold,
    )
    summary.update({
        "model": model_name,
        "feature_version": feature_version,
        "threshold": threshold,
        "negative_observations": summary["observations"] - summary["crises"],
    })
    return prediction_frame, pd.DataFrame([summary])


def compute_per_fold_metrics(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Calculate classification metrics for each individual test year fold."""
    records = []
    for test_year, group in predictions.groupby("test_year"):
        y_true = group["crisis_within_horizon"].to_numpy()
        prob = group["predicted_probability"].to_numpy()
        fold_summary = classification_metrics(y_true, prob, threshold)
        fold_summary["test_year"] = int(test_year)
        fold_summary["training_start_year"] = int(group["training_start_year"].iloc[0])
        fold_summary["training_end_year"] = int(group["training_end_year"].iloc[0])
        fold_summary["model"] = str(group["model"].iloc[0])
        fold_summary["threshold"] = threshold
        fold_summary["negative_observations"] = fold_summary["observations"] - fold_summary["crises"]
        records.append(fold_summary)
    return pd.DataFrame(records).sort_values("test_year").reset_index(drop=True)
