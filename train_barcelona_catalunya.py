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
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
MODEL_PATH = ROOT / "barcelona_catalunya_model.pkl"
PREDICTIONS_PATH = ROOT / "barcelona_catalunya_predictions.json"
METADATA_PATH = ROOT / "barcelona_catalunya_metadata.json"

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


def load_results() -> pd.DataFrame:
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


def build_prediction_rows(data: pd.DataFrame) -> pd.DataFrame:
    latest = data.sort_values("RaceOrder").groupby("Abbreviation").tail(1).set_index("Abbreviation")
    rows = []

    for code, driver, team in DRIVER_ROSTER_2026:
        history = latest.loc[code] if code in latest.index else None
        rows.append(
            {
                "Abbreviation": code,
                "driver": driver,
                "TeamName": team,
                "GridPosition": PROJECTED_GRID[code],
                "DriverCode": code,
                "AvgPoints5": float(history["AvgPoints5"]) if history is not None else 0.0,
                "AvgGrid5": float(history["AvgGrid5"]) if history is not None else PROJECTED_GRID[code],
                "AvgFinish5": float(history["AvgFinish5"]) if history is not None else 14.0,
                "WinRate10": float(history["WinRate10"]) if history is not None else 0.0,
                "TeamAvgPoints5": float(history["TeamAvgPoints5"]) if history is not None else 0.0,
                "BarcelonaExperience": float(history["BarcelonaExperience"]) if history is not None else 0.0,
                "BarcelonaWinRate": float(history["BarcelonaWinRate"]) if history is not None else 0.0,
                "IsStreetCircuit": 0,
            }
        )

    return pd.DataFrame(rows)


def apply_probability_postprocess(pred: pd.DataFrame, model_probs: np.ndarray) -> pd.Series:
    avg_points_max = pred["AvgPoints5"].clip(lower=0).max()
    team_points_max = pred["TeamAvgPoints5"].clip(lower=0).max()
    form_score = (
        (1 / pred["GridPosition"].clip(lower=1))
        + pred["AvgPoints5"].clip(lower=0) / (avg_points_max if avg_points_max else 1)
        + pred["TeamAvgPoints5"].clip(lower=0) / (team_points_max if team_points_max else 1)
        + pred["WinRate10"].clip(lower=0)
        + pred["BarcelonaWinRate"].clip(lower=0)
    )
    form_score = form_score / form_score.sum()
    blended_score = (0.75 * model_probs) + (0.25 * form_score.to_numpy()) + 0.001
    return pd.Series(blended_score / blended_score.sum(), index=pred.index)


def run_walk_forward_backtest(data: pd.DataFrame, min_training_races: int = 8) -> dict:
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
        target["probability"] = apply_probability_postprocess(target, model_probs)
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

    if not results:
        return {
            "races": [],
            "summary": {
                "races_tested": 0,
                "top1_accuracy": None,
                "top3_accuracy": None,
                "top5_accuracy": None,
                "mean_winner_rank": None,
                "log_loss": None,
                "brier_score": None,
            },
        }

    actual = [row["actual"] for row in probability_rows]
    probability = [row["probability"] for row in probability_rows]
    return {
        "races": results,
        "summary": {
            "races_tested": len(results),
            "top1_accuracy": round(float(np.mean([row["winner_rank"] == 1 for row in results])), 4),
            "top3_accuracy": round(float(np.mean([row["winner_rank"] <= 3 for row in results])), 4),
            "top5_accuracy": round(float(np.mean([row["winner_rank"] <= 5 for row in results])), 4),
            "mean_winner_rank": round(float(np.mean([row["winner_rank"] for row in results])), 2),
            "log_loss": round(float(log_loss(actual, probability)), 4),
            "brier_score": round(float(brier_score_loss(actual, probability)), 4),
        },
    }


def main() -> None:
    data = engineer_features(load_results())
    model = build_model()
    x = data[FEATURES]
    y = data["Winner"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probs = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]

    model.fit(x, y)
    pred = build_prediction_rows(data)
    model_probs = model.predict_proba(pred[FEATURES])[:, 1]
    pred["probability"] = apply_probability_postprocess(pred, model_probs)
    backtest = run_walk_forward_backtest(data)

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
        "model_version": "barcelona-catalunya-hgb-calibrated-1.0",
        "training_samples": int(len(data)),
        "training_races_loaded": int(data[["Year", "GrandPrix"]].drop_duplicates().shape[0]),
        "features": FEATURES,
        "validation": {
            "roc_auc": round(float(roc_auc_score(y, cv_probs)), 4),
            "log_loss": round(float(log_loss(y, cv_probs)), 4),
            "brier_score": round(float(brier_score_loss(y, cv_probs)), 4),
        },
        "prediction_postprocess": {
            "model_weight": 0.75,
            "form_prior_weight": 0.25,
            "floor_before_normalization": 0.001,
        },
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
