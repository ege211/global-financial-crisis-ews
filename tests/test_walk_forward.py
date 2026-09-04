import numpy as np
import pandas as pd

from crisis_ews.evaluation.walk_forward import evaluate_walk_forward, expanding_window_folds
from crisis_ews.models.models import logistic_pipeline, random_forest_pipeline


def test_expanding_folds_train_strictly_before_test() -> None:
    folds = list(expanding_window_folds(pd.Series([2000, 2001, 2002, 2003]), 2, 2002))
    assert [(fold.train_end_year, fold.test_year) for fold in folds] == [(2001, 2002), (2002, 2003)]


def test_scaler_is_fit_on_training_rows_not_future_rows() -> None:
    train = pd.DataFrame(
        {
            "feature": [1.0, 3.0, 1_000_000.0],
            "crisis_within_horizon": [0, 1, 0],
        }
    )
    model = logistic_pipeline({"class_weight": "balanced"}, seed=1)
    model.fit(train.iloc[:2][["feature"]], train.iloc[:2]["crisis_within_horizon"])
    assert model.named_steps["scaler"].mean_[0] == 2.0


def test_random_forest_imputer_is_fit_on_training_rows_not_future_rows() -> None:
    train = pd.DataFrame({"feature": [1.0, 3.0, 1_000_000.0], "crisis_within_horizon": [0, 1, 0]})
    model = random_forest_pipeline({"n_estimators": 5, "class_weight": "balanced"}, seed=1)
    model.fit(train.iloc[:2][["feature"]], train.iloc[:2]["crisis_within_horizon"])
    assert model.named_steps["imputer"].statistics_[0] == 2.0


def test_walk_forward_predictions_never_use_current_test_year_for_training() -> None:
    data = pd.DataFrame(
        {
            "country_code": ["AAA"] * 5 + ["BBB"] * 5,
            "year": list(range(2000, 2005)) * 2,
            "feature": np.arange(10, dtype=float),
            "crisis_within_horizon": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        }
    )
    predictions, _ = evaluate_walk_forward(
        data=data,
        features=["feature"],
        model_factory=lambda: logistic_pipeline({"class_weight": "balanced"}, seed=1),
        minimum_train_years=2,
        test_start_year=2002,
        threshold=0.5,
        model_name="logistic_regression",
    )
    assert (predictions["training_end_year"] < predictions["year"]).all()
