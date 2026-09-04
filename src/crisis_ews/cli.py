"""Small CLI exposing independent, reproducible pipeline stages."""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from crisis_ews.config import load_yaml
from crisis_ews.data.panel import build_panel
from crisis_ews.data.validation import country_coverage_report, coverage_report, validate_long_data
from crisis_ews.data.world_bank import WorldBankCollector
from crisis_ews.evaluation.walk_forward import evaluate_walk_forward
from crisis_ews.features.engineering import add_configured_transformations
from crisis_ews.labels.crisis_labels import construct_forward_label, read_crisis_chronology
from crisis_ews.labels.imf_2026 import (
    chronology_summary,
    import_validation_chronology,
    read_imf_2026_crises,
)

LOGGER = logging.getLogger(__name__)


def _project_path(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _run_path(root: Path, relative: str, namespace: str | None) -> Path:
    """Return a run-isolated artifact path while retaining legacy default locations."""
    return root / relative if not namespace else root / relative / namespace


def _inputs(root: Path, profile_path: Path | None) -> tuple[list[str], dict, dict, str | None]:
    """Load standard configuration, optionally applying a small audited run profile."""
    countries = load_yaml(root / "config/countries.yaml")["countries"]
    all_variables = load_yaml(root / "config/variables.yaml")["variables"]
    settings = deepcopy(load_yaml(root / "config/model_config.yaml"))
    if profile_path is None:
        return countries, all_variables, settings, None
    profile = load_yaml(profile_path if profile_path.is_absolute() else root / profile_path)
    countries = list(profile.get("countries", countries))
    selected_names = list(profile.get("variables", all_variables))
    unknown = set(selected_names) - set(all_variables)
    if unknown:
        raise ValueError(f"Profile contains unknown variables: {sorted(unknown)}")
    variables = {name: all_variables[name] for name in selected_names}
    if "features" in profile:
        settings["feature_sets"]["core"] = list(profile["features"])
    settings.update(profile.get("model_overrides", {}))
    return countries, variables, settings, profile.get("run_namespace")


def import_imf_2026_chronology(root: Path, dataset_zip: Path, profile_path: Path) -> None:
    """Extract and map the official IMF WP/26/94 workbook for a configured profile."""
    profile = load_yaml(profile_path if profile_path.is_absolute() else root / profile_path)
    countries, _, _, namespace = _inputs(root, profile_path)
    crosswalk_setting = profile.get("crisis_crosswalk")
    if not crosswalk_setting:
        raise ValueError("The profile must specify crisis_crosswalk for IMF chronology import")
    raw_directory = _run_path(root, "data/raw", namespace)
    interim_directory = _run_path(root, "data/interim", namespace)
    raw_directory.mkdir(parents=True, exist_ok=True)
    interim_directory.mkdir(parents=True, exist_ok=True)
    with ZipFile(dataset_zip) as archive:
        names = archive.namelist()
        workbooks = [name for name in names if name.lower().endswith(".xlsx")]
        if len(workbooks) != 1:
            raise ValueError("Expected exactly one workbook in the official IMF ZIP")
        workbook_name = workbooks[0]
        if Path(workbook_name).name != workbook_name:
            raise ValueError("The official IMF ZIP contains an unsafe workbook path")
        archive.extract(workbook_name, raw_directory)
    extract = read_imf_2026_crises(raw_directory / workbook_name)
    extract.to_csv(interim_directory / "imf_2026_crisis_source_extract.csv", index=False)
    crosswalk = Path(crosswalk_setting)
    chronology = import_validation_chronology(
        extract, crosswalk if crosswalk.is_absolute() else root / crosswalk
    )
    chronology.to_csv(raw_directory / "crisis_chronology.csv", index=False)
    table_directory = _run_path(root, "results", namespace) / "tables"
    table_directory.mkdir(parents=True, exist_ok=True)
    summary = chronology_summary(chronology, countries)
    summary.to_csv(table_directory / "crisis_chronology_summary.csv", index=False)
    LOGGER.info("Imported %s profile chronology episodes from official IMF workbook", len(chronology))


def download(root: Path, start_year: int, end_year: int, profile_path: Path | None = None) -> None:
    countries, variables, _, namespace = _inputs(root, profile_path)
    collector = WorldBankCollector()
    raw_directory = _run_path(root, "data/raw", namespace)
    data = collector.download_dataset(countries, variables, start_year, end_year, raw_directory)
    interim_path = _run_path(root, "data/interim", namespace) / "world_bank_long.csv"
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(interim_path, index=False)
    LOGGER.info("Saved %s raw observations", len(data))


def prepare(root: Path, chronology_path: Path, profile_path: Path | None = None) -> None:
    """Build annual features and supervised targets from the raw staging dataset."""
    countries, variables, settings, namespace = _inputs(root, profile_path)
    raw_path = _run_path(root, "data/interim", namespace) / "world_bank_long.csv"
    if not raw_path.exists():
        raise FileNotFoundError("Run `ews download` before `ews prepare`.")
    long_data = pd.read_csv(raw_path)
    quality = validate_long_data(long_data, variables)
    panel = build_panel(long_data, variables)
    features = add_configured_transformations(panel, variables)
    chronology = read_crisis_chronology(chronology_path)
    modeling = construct_forward_label(
        features,
        chronology,
        int(settings["prediction_horizon_years"]),
        str(settings["crisis_type"]),
    )
    processed_directory = _run_path(root, "data/processed", namespace)
    processed_directory.mkdir(parents=True, exist_ok=True)
    modeling.to_csv(processed_directory / "modeling_panel.csv", index=False)
    report = coverage_report(long_data, countries, list(variables))
    metadata_directory = _run_path(root, "data/metadata", namespace)
    metadata_directory.mkdir(parents=True, exist_ok=True)
    report.to_csv(metadata_directory / "coverage_report.csv", index=False)
    with (_run_path(root, "data/raw", namespace) / "world_bank_provenance.json").open(encoding="utf-8") as handle:
        provenance = json.load(handle)
    series = provenance["series"]
    start_year = int(next(iter(series.values()))["requested_start_year"])
    end_year = int(next(iter(series.values()))["requested_end_year"])
    country_coverage_report(
        long_data, modeling, countries, list(variables), start_year, end_year
    ).to_csv(metadata_directory / "country_coverage_report.csv", index=False)
    (metadata_directory / "data_quality_report.json").write_text(
        json.dumps(asdict(quality), indent=2), encoding="utf-8"
    )
    LOGGER.info("Saved %s country-year rows to modeling panel", len(modeling))


def evaluate(root: Path, model_name: str, profile_path: Path | None = None) -> None:
    """Run expanding-window validation; raw/model inputs remain unchanged."""
    _, _, settings, namespace = _inputs(root, profile_path)
    data_path = _run_path(root, "data/processed", namespace) / "modeling_panel.csv"
    if not data_path.exists():
        raise FileNotFoundError("Run `ews prepare --chronology ...` before `ews evaluate`.")
    data = pd.read_csv(data_path)
    features = settings["feature_sets"]["core"]
    seed = int(settings["seed"])
    if model_name == "logistic_regression":
        from crisis_ews.models.models import logistic_pipeline

        factory = lambda: logistic_pipeline(settings["models"][model_name], seed)
    elif model_name == "random_forest":
        from crisis_ews.models.models import random_forest_pipeline

        factory = lambda: random_forest_pipeline(settings["models"][model_name], seed)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    test_end_year = settings.get("test_end_year")
    predictions, metrics = evaluate_walk_forward(
        data=data,
        features=features,
        model_factory=factory,
        minimum_train_years=int(settings["minimum_train_years"]),
        test_start_year=int(settings["test_start_year"]),
        threshold=float(settings["threshold"]),
        model_name=model_name,
        test_end_year=int(test_end_year) if test_end_year is not None else None,
    )
    output_directory = _run_path(root, "results", namespace)
    (output_directory / "model_outputs").mkdir(parents=True, exist_ok=True)
    (output_directory / "tables").mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_directory / "model_outputs" / f"{model_name}_predictions.csv", index=False)
    metrics.to_csv(output_directory / "tables" / f"{model_name}_metrics.csv", index=False)
    LOGGER.info("Saved %s out-of-sample predictions", len(predictions))


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial crisis early-warning research pipeline")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root (default: current directory)")
    parser.add_argument("--profile", type=Path, help="Optional run profile, e.g. config/pipeline_validation.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    download_parser = subparsers.add_parser("download", help="Download unmodified World Bank data")
    download_parser.add_argument("--start-year", type=int)
    download_parser.add_argument("--end-year", type=int)
    import_parser = subparsers.add_parser("import-imf-2026", help="Import official IMF WP/26/94 chronology")
    import_parser.add_argument("--dataset-zip", type=Path, required=True)
    prepare_parser = subparsers.add_parser("prepare", help="Build modeled panel and labels")
    prepare_parser.add_argument("--chronology", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate", help="Run expanding-window evaluation")
    evaluate_parser.add_argument("--model", choices=["logistic_regression", "random_forest"], default="logistic_regression")
    expanded_parser = subparsers.add_parser(
        "run-expanded-research",
        aliases=["expanded-research"],
        help="Execute Phase 4B expanded universe pipeline across qualifying economies",
    )
    expanded_parser.add_argument("--profile", type=Path, help="Run profile path")
    pre_fit_parser = subparsers.add_parser(
        "pre-fit-audit",
        aliases=["audit-pre-fit"],
        help="Execute Phase 5A pre-fit audit across the 76-country panel without model fitting",
    )
    pre_fit_parser.add_argument("--profile", type=Path, help="Run profile path")
    initial_models_parser = subparsers.add_parser(
        "run-initial-models",
        aliases=["initial-models"],
        help="Execute Phase 5B initial model estimation across the locked expanding window (2000-2008)",
    )
    initial_models_parser.add_argument("--profile", type=Path, help="Run profile path")
    robustness_parser = subparsers.add_parser(
        "run-phase-5c",
        aliases=["robustness", "robustness-analysis"],
        help="Execute Phase 5C robustness and sensitivity analysis across the 76-country panel",
    )
    robustness_parser.add_argument("--profile", type=Path, help="Run profile path")
    predictor_audit_parser = subparsers.add_parser(
        "run-phase-5d",
        aliases=["predictor-audit", "audit-predictors"],
        help="Execute Phase 5D predictor selection and data coverage audit across candidate indicators",
    )
    predictor_audit_parser.add_argument("--profile", type=Path, help="Run profile path")
    extended_models_parser = subparsers.add_parser(
        "run-extended-models",
        aliases=["extended-models", "run-phase-5e"],
        help="Execute Phase 5E 6-variable Extended Model estimation across the locked expanding window",
    )
    extended_models_parser.add_argument("--profile", type=Path, help="Run profile path")
    generalization_parser = subparsers.add_parser(
        "temporal-generalization",
        aliases=["run-temporal-generalization", "run-phase-5e-generalization"],
        help="Execute Phase 5E temporal generalization and GFC dependence analysis",
    )
    generalization_parser.add_argument("--threshold", type=float, default=0.50, help="Classification decision threshold")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = args.root.resolve()
    profile = args.profile
    if profile is not None:
        profile_data = load_yaml(profile if profile.is_absolute() else root / profile)
    else:
        profile_data = {}
    if args.command == "download":
        download(root, args.start_year or int(profile_data.get("start_year", 1990)), args.end_year or int(profile_data.get("end_year", 2025)), profile)
    elif args.command == "prepare":
        prepare(root, args.chronology, profile)
    elif args.command == "import-imf-2026":
        if profile is None:
            raise ValueError("--profile is required for import-imf-2026")
        import_imf_2026_chronology(root, args.dataset_zip, profile)
    elif args.command == "evaluate":
        evaluate(root, args.model, profile)
    elif args.command in ("run-expanded-research", "expanded-research"):
        from crisis_ews.data.expanded_pipeline import run_phase_4b_pipeline

        profile_path = profile if profile is not None else (root / "config/expanded_research.yaml")
        summary = run_phase_4b_pipeline(root, profile_path)
        LOGGER.info(
            "Phase 4B expanded research completed: %s candidate economies, %s included, "
            "%s country-year rows, %s positive forward labels",
            summary["candidate_countries"],
            summary["included_countries"],
            summary["total_rows"],
            summary["positive_labels"],
        )
    elif args.command in ("pre-fit-audit", "audit-pre-fit"):
        from crisis_ews.evaluation.pre_fit_audit import run_pre_fit_audit

        profile_path = profile if profile is not None else (root / "config/expanded_research.yaml")
        audit_summary = run_pre_fit_audit(root, profile_path)
        LOGGER.info(
            "Phase 5A pre-fit audit completed: %s folds, %s cumulative test rows, %s test positives, zero models fitted",
            audit_summary["folds_count"],
            audit_summary["cumulative_test_rows"],
            audit_summary["cumulative_test_positives"],
        )
    elif args.command in ("run-initial-models", "initial-models"):
        from crisis_ews.evaluation.initial_models import run_phase_5b_initial_models

        profile_path = profile if profile is not None else (root / "config/expanded_research.yaml")
        model_summary = run_phase_5b_initial_models(root, profile_path)
        LOGGER.info(
            "Phase 5B model estimation completed: %s test observations, %s positive events evaluated across 9 folds",
            model_summary["total_test_observations"],
            model_summary["test_positive_events"],
        )
    elif args.command in ("run-phase-5c", "robustness", "robustness-analysis"):
        from crisis_ews.evaluation.robustness_analysis import run_phase_5c_robustness

        profile_path = profile if profile is not None else (root / "config/expanded_research.yaml")
        summary = run_phase_5c_robustness(root, profile_path)
        LOGGER.info(
            "Phase 5C robustness analysis completed: 4 analyses evaluated across expanding folds"
        )
    elif args.command in ("run-phase-5d", "predictor-audit", "audit-predictors"):
        from crisis_ews.evaluation.predictor_audit import run_phase_5d_predictor_audit

        profile_path = profile if profile is not None else (root / "config/expanded_research.yaml")
        summary = run_phase_5d_predictor_audit(root, profile_path=profile_path)
        LOGGER.info(
            "Phase 5D predictor audit completed: %s candidates audited, %s eligible: %s",
            summary["candidate_count"],
            len(summary["eligible_candidates"]),
            summary["eligible_candidates"],
        )
    elif args.command in ("run-extended-models", "extended-models", "run-phase-5e"):
        from crisis_ews.evaluation.extended_models import run_phase_5e_extended_models

        profile_path = profile if profile is not None else (root / "config/expanded_research.yaml")
        summary = run_phase_5e_extended_models(root, profile_path=profile_path)
        LOGGER.info(
            "Phase 5E Extended Model estimation completed across %s test observations (%s events)",
            summary["total_test_observations"],
            summary["test_positive_events"],
        )
    elif args.command in ("temporal-generalization", "run-temporal-generalization", "run-phase-5e-generalization"):
        from crisis_ews.evaluation.temporal_generalization import (
            run_phase_5e_temporal_generalization,
        )

        df = run_phase_5e_temporal_generalization(root, threshold=args.threshold)
        LOGGER.info(
            "Phase 5E Temporal Generalization analysis completed across %s evaluation subsets",
            len(df["subset_key"].unique()),
        )


if __name__ == "__main__":
    main()
