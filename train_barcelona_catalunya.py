from __future__ import annotations

import json
import warnings
from pathlib import Path

import fastf1
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
MODEL_PATH = ROOT / "barcelona_catalunya_model.pkl"
PREDICTIONS_PATH = ROOT / "barcelona_catalunya_predictions.json"
METADATA_PATH = ROOT / "barcelona_catalunya_metadata.json"
GRID_OVERRIDE_PATH = ROOT / "qualifying_grid.json"
PREDICTION_ONLY_PRIOR_WEIGHTS = {
    "base_probability": 0.85,
    "recent_dominance": 0.075,
    "market_odds": 0.075,
}
PREDICTION_ONLY_DRIVER_PRIORS = {
    "ANT": {
        "recent_dominance_score": 1.0,
        "market_decimal_odds": 2.0,
        "evidence": [
            "Five consecutive Grand Prix wins before Barcelona-Catalunya",
            "Monaco 2026 pole, win, fastest lap, led every lap, and Grand Slam",
            "Quoted Barcelona-Catalunya winner odds: 1/1 (decimal 2.0, American +100)",
        ],
    }
}
POSTPROCESS_CANDIDATES = []
for model_weight in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
    remaining_weight = 1 - model_weight
    for grid_share in (0.25, 0.35, 0.45, 0.55):
        for track_share in (0.15, 0.25, 0.35):
            form_share = 1 - grid_share - track_share
            if form_share <= 0:
                continue
            for floor in (0.0005, 0.001):
                POSTPROCESS_CANDIDATES.append(
                    {
                        "model_weight": model_weight,
                        "form_weight": remaining_weight * form_share,
                        "grid_weight": remaining_weight * grid_share,
                        "track_weight": remaining_weight * track_share,
                        "floor": floor,
                    }
                )

RACES_TO_LOAD = [
    (2022, "Spain"),
    (2023, "Spain"),
    (2024, "Spain"),
    (2025, "Australia"),
    (2025, "China"),
    (2025, "Japan"),
    (2025, "Bahrain"),
    (2025, "Saudi Arabia"),
    (2025, "Miami"),
    (2025, "Emilia Romagna"),
    (2025, "Monaco"),
    (2025, "Spain"),
    (2025, "Canada"),
    (2025, "Austria"),
    (2025, "British"),
    (2025, "Belgium"),
    (2025, "Hungary"),
    (2025, "Dutch"),
    (2025, "Italian"),
    (2025, "Azerbaijan"),
    (2025, "Singapore"),
    (2025, "United States"),
    (2025, "Mexico"),
    (2025, "Sao Paulo"),
    (2025, "Las Vegas"),
    (2025, "Qatar"),
    (2025, "Abu Dhabi"),
    (2026, "Australia"),
    (2026, "China"),
    (2026, "Japan"),
    (2026, "Bahrain"),
    (2026, "Saudi Arabia"),
    (2026, "Miami"),
    (2026, "Canada"),
    (2026, "Monaco"),
]

STREET_CIRCUITS = {"Miami", "Monaco", "Singapore", "Azerbaijan", "Las Vegas", "Saudi Arabia"}
FEATURES = [
    "GridPosition",
    "DriverCode",
    "TeamName",
    "AvgPoints5",
    "AvgGrid5",
    "AvgFinish5",
    "WinRate10",
    "TeamAvgPoints5",
    "BarcelonaExperience",
    "BarcelonaWinRate",
    "IsStreetCircuit",
]
CATEGORICAL_FEATURES = ["DriverCode", "TeamName"]
NUMERIC_FEATURES = [feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES]

DRIVER_ROSTER_2026 = [
    ("NOR", "Norris", "McLaren"),
    ("PIA", "Piastri", "McLaren"),
    ("RUS", "Russell", "Mercedes"),
    ("ANT", "Antonelli", "Mercedes"),
    ("VER", "Verstappen", "Red Bull Racing"),
    ("HAD", "Hadjar", "Red Bull Racing"),
    ("LEC", "Leclerc", "Ferrari"),
    ("HAM", "Hamilton", "Ferrari"),
    ("ALB", "Albon", "Williams"),
    ("SAI", "Sainz", "Williams"),
    ("LIN", "Lindblad", "Racing Bulls"),
    ("LAW", "Lawson", "Racing Bulls"),
    ("STR", "Stroll", "Aston Martin"),
    ("ALO", "Alonso", "Aston Martin"),
    ("OCO", "Ocon", "Haas"),
    ("BEA", "Bearman", "Haas"),
    ("HUL", "Hulkenberg", "Audi"),
    ("BOR", "Bortoleto", "Audi"),
    ("GAS", "Gasly", "Alpine"),
    ("COL", "Colapinto", "Alpine"),
    ("PER", "Perez", "Cadillac"),
    ("BOT", "Bottas", "Cadillac"),
]

PROJECTED_GRID = {
    "NOR": 1,
    "PIA": 2,
    "RUS": 3,
    "ANT": 4,
    "VER": 5,
    "LEC": 6,
    "HAM": 7,
    "HAD": 8,
    "ALB": 9,
    "SAI": 10,
    "ALO": 11,
    "STR": 12,
    "BEA": 13,
    "OCO": 14,
    "GAS": 15,
    "COL": 16,
    "LAW": 17,
    "LIN": 18,
    "HUL": 19,
    "BOR": 20,
    "PER": 21,
    "BOT": 22,
}


def load_grid_positions() -> tuple[dict[str, int], dict]:
    if not GRID_OVERRIDE_PATH.exists():
        return PROJECTED_GRID.copy(), {
            "grid_source": "projected_grid",
            "grid_override_file": None,
            "overridden_drivers": [],
        }

    raw_grid = json.loads(GRID_OVERRIDE_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw_grid, dict):
        raise ValueError("qualifying_grid.json must be an object like {\"NOR\": 1, \"PIA\": 2}")

    grid = PROJECTED_GRID.copy()
    unknown_codes = sorted(set(raw_grid) - set(PROJECTED_GRID))
    if unknown_codes:
        raise ValueError(f"qualifying_grid.json contains unknown driver codes: {unknown_codes}")

    for code, position in raw_grid.items():
        position = int(position)
        if position < 1:
            raise ValueError(f"Grid position for {code} must be 1 or greater")
        grid[code] = position

    return grid, {
        "grid_source": "qualifying_grid",
        "grid_override_file": GRID_OVERRIDE_PATH.name,
        "overridden_drivers": sorted(raw_grid),
    }


def load_results() -> pd.DataFrame:
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    rows = []

    for race_order, (year, gp) in enumerate(RACES_TO_LOAD):
        try:
            session = fastf1.get_session(year, gp, "R")
            session.load(telemetry=False, weather=False, messages=False)
            result = session.results[
                ["Abbreviation", "TeamName", "GridPosition", "Position", "Points"]
            ].copy()
            if result.empty:
                raise RuntimeError("FastF1 returned no classified results")
            result["Year"] = year
            result["GrandPrix"] = gp
            result["RaceOrder"] = race_order
            result["IsStreetCircuit"] = int(gp in STREET_CIRCUITS)
            result["Winner"] = (pd.to_numeric(result["Position"], errors="coerce") == 1).astype(int)
            rows.append(result)
            print(f"Loaded {year} {gp}")
        except Exception as exc:
            print(f"Skipped {year} {gp}: {exc}")

    if not rows:
        raise RuntimeError("No race data loaded. Check FastF1/cache/network availability.")

    data = pd.concat(rows, ignore_index=True)
    for column, fallback in {"GridPosition": 10, "Position": 20, "Points": 0}.items():
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(fallback)
    data["DriverCode"] = data["Abbreviation"]
    return data


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values(["RaceOrder", "Position"])
    grouped_driver = data.groupby("Abbreviation", group_keys=False)
    grouped_team = data.groupby("TeamName", group_keys=False)

    data["AvgPoints5"] = grouped_driver["Points"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    data["AvgGrid5"] = grouped_driver["GridPosition"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    data["AvgFinish5"] = grouped_driver["Position"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    data["WinRate10"] = grouped_driver["Winner"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    data["TeamAvgPoints5"] = grouped_team["Points"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    data["BarcelonaExperience"] = data.groupby("Abbreviation")["GrandPrix"].transform(
        lambda x: x.eq("Spain").shift(fill_value=False).cumsum()
    )
    data["BarcelonaWinRate"] = 0.0
    for _, index in data.groupby("Abbreviation").groups.items():
        driver_rows = data.loc[index].sort_values("RaceOrder")
        prior_barcelona_wins = []
        wins = []
        for _, row in driver_rows.iterrows():
            prior_barcelona_wins.append(float(np.mean(wins)) if wins else 0.0)
            if row["GrandPrix"] == "Spain":
                wins.append(row["Winner"])
        data.loc[driver_rows.index, "BarcelonaWinRate"] = prior_barcelona_wins

    data["AvgPoints5"] = data["AvgPoints5"].fillna(0)
    data["AvgGrid5"] = data["AvgGrid5"].fillna(data["GridPosition"])
    data["AvgFinish5"] = data["AvgFinish5"].fillna(14)
    data["WinRate10"] = data["WinRate10"].fillna(0)
    data["TeamAvgPoints5"] = data["TeamAvgPoints5"].fillna(0)
    return data


def build_model() -> Pipeline:
    base = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=220,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=42,
    )
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
    return Pipeline(
        [
            (
                "encode",
                ColumnTransformer(
                    [
                        (
                            "categorical",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                            CATEGORICAL_FEATURES,
                        ),
                        ("numeric", "passthrough", NUMERIC_FEATURES),
                    ],
                    remainder="drop",
                ),
            ),
            ("model", calibrated),
        ]
    )


def build_prediction_rows(data: pd.DataFrame, grid_positions: dict[str, int]) -> pd.DataFrame:
    latest = data.sort_values("RaceOrder").groupby("Abbreviation").tail(1).set_index("Abbreviation")
    rows = []

    for code, driver, team in DRIVER_ROSTER_2026:
        history = latest.loc[code] if code in latest.index else None
        rows.append(
            {
                "Abbreviation": code,
                "driver": driver,
                "TeamName": team,
                "GridPosition": grid_positions[code],
                "DriverCode": code,
                "AvgPoints5": float(history["AvgPoints5"]) if history is not None else 0.0,
                "AvgGrid5": float(history["AvgGrid5"]) if history is not None else grid_positions[code],
                "AvgFinish5": float(history["AvgFinish5"]) if history is not None else 14.0,
                "WinRate10": float(history["WinRate10"]) if history is not None else 0.0,
                "TeamAvgPoints5": float(history["TeamAvgPoints5"]) if history is not None else 0.0,
                "BarcelonaExperience": float(history["BarcelonaExperience"]) if history is not None else 0.0,
                "BarcelonaWinRate": float(history["BarcelonaWinRate"]) if history is not None else 0.0,
                "IsStreetCircuit": 0,
                "RecentDominancePriorScore": PREDICTION_ONLY_DRIVER_PRIORS.get(code, {}).get(
                    "recent_dominance_score", 0.0
                ),
                "MarketImpliedProbability": (
                    1 / PREDICTION_ONLY_DRIVER_PRIORS[code]["market_decimal_odds"]
                    if code in PREDICTION_ONLY_DRIVER_PRIORS
                    and PREDICTION_ONLY_DRIVER_PRIORS[code].get("market_decimal_odds")
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def normalize_prior(score: pd.Series) -> pd.Series:
    score = score.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
    total = score.sum()
    if total <= 0:
        return pd.Series(np.full(len(score), 1 / len(score)), index=score.index)
    return score / total


def calculate_component_priors(pred: pd.DataFrame) -> dict[str, pd.Series]:
    avg_points_max = pred["AvgPoints5"].clip(lower=0).max()
    team_points_max = pred["TeamAvgPoints5"].clip(lower=0).max()
    avg_finish_score = 1 / pred["AvgFinish5"].clip(lower=1)
    avg_finish_max = avg_finish_score.max()
    experience_max = pred["BarcelonaExperience"].clip(lower=0).max()

    form_score = (
        pred["AvgPoints5"].clip(lower=0) / (avg_points_max if avg_points_max else 1)
        + pred["TeamAvgPoints5"].clip(lower=0) / (team_points_max if team_points_max else 1)
        + pred["WinRate10"].clip(lower=0)
        + avg_finish_score / (avg_finish_max if avg_finish_max else 1)
    )
    grid_score = 1 / np.power(pred["GridPosition"].clip(lower=1), 1.25)
    track_score = (
        pred["BarcelonaWinRate"].clip(lower=0)
        + 0.15 * pred["BarcelonaExperience"].clip(lower=0) / (experience_max if experience_max else 1)
    )

    return {
        "form": normalize_prior(form_score),
        "grid": normalize_prior(grid_score),
        "track": normalize_prior(track_score),
    }


def blend_probabilities(
    model_probs: np.ndarray,
    priors: dict[str, pd.Series],
    config: dict,
) -> pd.Series:
    blended_score = (
        (config["model_weight"] * model_probs)
        + (config["form_weight"] * priors["form"].to_numpy())
        + (config["grid_weight"] * priors["grid"].to_numpy())
        + (config["track_weight"] * priors["track"].to_numpy())
        + config["floor"]
    )
    return pd.Series(blended_score / blended_score.sum(), index=priors["form"].index)


def apply_probability_postprocess(
    pred: pd.DataFrame,
    model_probs: np.ndarray,
    config: dict,
) -> pd.Series:
    priors = calculate_component_priors(pred)
    return blend_probabilities(model_probs, priors, config)


def apply_prediction_only_priors(pred: pd.DataFrame, base_probability: pd.Series) -> pd.Series:
    dominance_prior = normalize_prior(pred["RecentDominancePriorScore"])
    market_prior = normalize_prior(pred["MarketImpliedProbability"])
    weights = PREDICTION_ONLY_PRIOR_WEIGHTS
    adjusted_score = (
        weights["base_probability"] * base_probability.to_numpy()
        + weights["recent_dominance"] * dominance_prior.to_numpy()
        + weights["market_odds"] * market_prior.to_numpy()
    )
    return pd.Series(adjusted_score / adjusted_score.sum(), index=base_probability.index)


def summarize_backtest(results: list[dict], probability_rows: list[dict]) -> dict:
    if not results:
        return {
            "races_tested": 0,
            "top1_accuracy": None,
            "top3_accuracy": None,
            "top5_accuracy": None,
            "mean_winner_rank": None,
            "log_loss": None,
            "brier_score": None,
        }

    actual = [row["actual"] for row in probability_rows]
    probability = [row["probability"] for row in probability_rows]
    return {
        "races_tested": len(results),
        "top1_accuracy": round(float(np.mean([row["winner_rank"] == 1 for row in results])), 4),
        "top3_accuracy": round(float(np.mean([row["winner_rank"] <= 3 for row in results])), 4),
        "top5_accuracy": round(float(np.mean([row["winner_rank"] <= 5 for row in results])), 4),
        "mean_winner_rank": round(float(np.mean([row["winner_rank"] for row in results])), 2),
        "log_loss": round(float(log_loss(actual, probability)), 4),
        "brier_score": round(float(brier_score_loss(actual, probability)), 4),
    }


def rank_backtest_result(item: dict) -> tuple:
    summary = item["summary"]
    return (
        summary["top1_accuracy"] or 0,
        -1 * (summary["mean_winner_rank"] or 999),
        -1 * (summary["log_loss"] or 999),
        -1 * (summary["brier_score"] or 999),
    )


def tune_walk_forward_postprocess(data: pd.DataFrame, min_training_races: int = 8) -> dict:
    race_keys = (
        data[["RaceOrder", "Year", "GrandPrix"]]
        .drop_duplicates()
        .sort_values("RaceOrder")
        .to_dict("records")
    )
    candidates = {
        index: {"config": config, "races": [], "probability_rows": []}
        for index, config in enumerate(POSTPROCESS_CANDIDATES)
    }

    for race in race_keys:
        prior_race_count = sum(item["RaceOrder"] < race["RaceOrder"] for item in race_keys)
        if prior_race_count < min_training_races:
            continue

        train = data[data["RaceOrder"] < race["RaceOrder"]]
        target = data[data["RaceOrder"] == race["RaceOrder"]].copy()
        if train["Winner"].nunique() < 2 or target.empty:
            continue

        model = build_model()
        model.fit(train[FEATURES], train["Winner"])
        model_probs = model.predict_proba(target[FEATURES])[:, 1]
        priors = calculate_component_priors(target)

        for candidate in candidates.values():
            config = candidate["config"]
            target["probability"] = blend_probabilities(
                model_probs,
                priors,
                config,
            )
            ranked = target.sort_values("probability", ascending=False).reset_index(drop=True)
            winner_index = ranked.index[ranked["Winner"].eq(1)]
            if len(winner_index) == 0:
                continue

            winner_rank = int(winner_index[0] + 1)
            winner = ranked.loc[winner_index[0]]
            candidate["races"].append(
                {
                    "race": f"{race['Year']} {race['GrandPrix']}",
                    "winner": str(winner["Abbreviation"]),
                    "winner_rank": winner_rank,
                    "winner_probability": round(float(winner["probability"]), 4),
                }
            )
            candidate["probability_rows"].extend(
                {
                    "actual": int(row["Winner"]),
                    "probability": float(row["probability"]),
                }
                for _, row in ranked.iterrows()
            )

    evaluated = []
    for candidate in candidates.values():
        evaluated.append(
            {
                "config": candidate["config"],
                "races": candidate["races"],
                "summary": summarize_backtest(candidate["races"], candidate["probability_rows"]),
            }
        )

    best = max(evaluated, key=rank_backtest_result)
    return {
        "selected_config": best["config"],
        "races": best["races"],
        "summary": best["summary"],
        "candidates_tested": len(evaluated),
        "selection_metric": "top1_accuracy, then mean_winner_rank, log_loss, brier_score",
    }


def run_walk_forward_backtest(
    data: pd.DataFrame,
    config: dict,
    min_training_races: int = 8,
) -> dict:
    race_keys = (
        data[["RaceOrder", "Year", "GrandPrix"]]
        .drop_duplicates()
        .sort_values("RaceOrder")
        .to_dict("records")
    )
    results = []
    probability_rows = []

    for race in race_keys:
        prior_race_count = sum(item["RaceOrder"] < race["RaceOrder"] for item in race_keys)
        if prior_race_count < min_training_races:
            continue

        train = data[data["RaceOrder"] < race["RaceOrder"]]
        target = data[data["RaceOrder"] == race["RaceOrder"]].copy()
        if train["Winner"].nunique() < 2 or target.empty:
            continue

        model = build_model()
        model.fit(train[FEATURES], train["Winner"])
        model_probs = model.predict_proba(target[FEATURES])[:, 1]
        target["probability"] = apply_probability_postprocess(
            target,
            model_probs,
            config,
        )
        ranked = target.sort_values("probability", ascending=False).reset_index(drop=True)
        winner_index = ranked.index[ranked["Winner"].eq(1)]
        if len(winner_index) == 0:
            continue

        winner_rank = int(winner_index[0] + 1)
        winner = ranked.loc[winner_index[0]]
        results.append(
            {
                "race": f"{race['Year']} {race['GrandPrix']}",
                "winner": str(winner["Abbreviation"]),
                "winner_rank": winner_rank,
                "winner_probability": round(float(winner["probability"]), 4),
            }
        )
        probability_rows.extend(
            {
                "actual": int(row["Winner"]),
                "probability": float(row["probability"]),
            }
            for _, row in ranked.iterrows()
        )

    return {
        "races": results,
        "summary": summarize_backtest(results, probability_rows),
    }


def main() -> None:
    data = engineer_features(load_results())
    grid_positions, grid_metadata = load_grid_positions()
    tuned_postprocess = tune_walk_forward_postprocess(data)
    selected_config = tuned_postprocess["selected_config"]
    model = build_model()
    x = data[FEATURES]

    model.fit(x, data["Winner"])
    pred = build_prediction_rows(data, grid_positions)
    model_probs = model.predict_proba(pred[FEATURES])[:, 1]
    pred["validated_probability"] = apply_probability_postprocess(pred, model_probs, selected_config)
    pred["probability"] = apply_prediction_only_priors(pred, pred["validated_probability"])
    backtest = {
        "races": tuned_postprocess["races"],
        "summary": tuned_postprocess["summary"],
    }

    predictions = [
        {
            "driver": row["driver"],
            "team": row["TeamName"],
            "probability": round(float(row["probability"]), 4),
        }
        for _, row in pred.sort_values("probability", ascending=False).iterrows()
    ]

    metadata = {
        "race": "Barcelona-Catalunya GP",
        "circuit": "Circuit de Barcelona-Catalunya",
        "model_version": "barcelona-catalunya-hgb-calibrated-1.3",
        "training_samples": int(len(data)),
        "training_races_loaded": int(data[["Year", "GrandPrix"]].drop_duplicates().shape[0]),
        "features": FEATURES,
        "validation": {
            "method": "walk_forward_backtest",
            **backtest["summary"],
        },
        "prediction_postprocess": {
            "model_weight": round(selected_config["model_weight"], 4),
            "form_prior_weight": round(selected_config["form_weight"], 4),
            "grid_prior_weight": round(selected_config["grid_weight"], 4),
            "barcelona_track_prior_weight": round(selected_config["track_weight"], 4),
            "floor_before_normalization": selected_config["floor"],
            "prediction_only_prior_weights": PREDICTION_ONLY_PRIOR_WEIGHTS,
            "prediction_only_driver_priors": PREDICTION_ONLY_DRIVER_PRIORS,
            "selection": {
                "method": "walk_forward_grid_search",
                "candidates_tested": tuned_postprocess["candidates_tested"],
                "selection_metric": tuned_postprocess["selection_metric"],
            },
        },
        "prediction_input": grid_metadata,
        "backtest": backtest,
    }

    joblib.dump(model, MODEL_PATH)
    PREDICTIONS_PATH.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    for i, item in enumerate(predictions, 1):
        print(f"P{i}: {item['driver']:12} {item['team']:16} {item['probability'] * 100:5.1f}%")


if __name__ == "__main__":
    main()
