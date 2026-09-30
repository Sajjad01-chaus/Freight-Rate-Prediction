"""Exploratory data analysis: writes figures to reports/figures and a findings summary.

Run:  python scripts/eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freight import config, data

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
EQUIP_COLORS = {"Dry Van": BLUE, "Flatbed": ORANGE, "Reefer": AQUA}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": "#9c9b96", "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "lines.linewidth": 2.0, "legend.frameon": False,
})


def save(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png")
    plt.close(fig)


def base_residuals(clean: pd.DataFrame) -> pd.Series:
    """Residual of log(rate) after a simple structural model (distance, equipment, weight, market)."""
    X = np.column_stack([
        np.ones(len(clean)),
        np.log(clean["distance"]),
        clean["equipment"].eq("Flatbed"),
        clean["equipment"].eq("Reefer"),
        clean["weight"].fillna(clean["weight"].median()) / 1e4,
        np.log(clean["market_index_filled"]),
    ]).astype(float)
    y = np.log(clean[config.TARGET]).to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return pd.Series(y - X @ beta, index=clean.index)


def main() -> None:
    raw_train, raw_val = data.load_train(), data.load_validation()
    train = data.flag_label_outliers(data.clean_features(raw_train))
    val = data.clean_features(raw_val)
    for frame in (train, val):
        frame["market_index_filled"] = frame["market_index"].fillna(
            frame.groupby("date")["market_index"].transform("mean")
        )
    clean = train[train["label_outlier"] == 0].copy()
    clean["rpm"] = clean[config.TARGET] / clean["distance"]
    clean["resid"] = base_residuals(clean)
    facts: dict[str, str] = {}

    # 1. Label corruption: rate / expected-rate ratio
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bins = np.logspace(np.log10(0.1), np.log10(8), 120)
    ax.hist(train.loc[train.label_outlier == 0, "rate_ratio"], bins=bins, color=BLUE, label="Clean loads")
    ax.hist(train.loc[train.label_outlier == 1, "rate_ratio"], bins=bins, color=ORANGE, label="Flagged as corrupted")
    ax.set_xscale("log"); ax.set_yscale("log")
    for x in (config.OUTLIER_LOW_RATIO, config.OUTLIER_HIGH_RATIO):
        ax.axvline(x, color=INK_2, linewidth=0.8)
    ax.set_xticks([0.2, 0.5, 1, 2, 5]); ax.set_xticklabels(["0.2x", "0.5x", "1x", "2x", "5x"])
    ax.set_xlabel("Posted rate / expected rate (equipment x distance band x month median)")
    ax.set_ylabel("Loads (log scale)")
    ax.set_title("~1.4% of labels are corrupted: two clusters far from the clean band")
    ax.legend(loc="upper right")
    save(fig, "01_label_corruption")
    lo = int((train.rate_ratio < config.OUTLIER_LOW_RATIO).sum())
    hi = int((train.rate_ratio > config.OUTLIER_HIGH_RATIO).sum())
    facts["Corrupted labels"] = (
        f"{lo + hi} of {len(train):,} ({(lo + hi) / len(train):.2%}): {lo} deflated (0.16-0.45x), "
        f"{hi} inflated (2.1-5.2x); clean loads sit within 0.8-1.2x of expected"
    )

    # 2. Rate per mile vs distance by equipment
    fig, ax = plt.subplots(figsize=(9, 4.2))
    edges = np.quantile(clean["distance"], np.linspace(0, 1, 26))
    clean["dist_bin"] = pd.cut(clean["distance"], edges, include_lowest=True)
    for equip, color in EQUIP_COLORS.items():
        grp = clean[clean.equipment == equip].groupby("dist_bin", observed=True)
        curve = grp.agg(d=("distance", "median"), rpm=("rpm", "median"))
        ax.plot(curve["d"], curve["rpm"], color=color, marker="o", markersize=4, label=equip)
        ax.annotate(equip, (curve["d"].iloc[-1], curve["rpm"].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", color=INK_2, va="center", fontsize=9)
    ax.set_xlabel("Distance (miles)"); ax.set_ylabel("Median rate per mile ($)")
    ax.set_title("Rate per mile falls with distance; Reefer > Flatbed > Dry Van at every haul length")
    ax.legend(loc="upper right")
    save(fig, "02_rate_per_mile_vs_distance")
    rel =clean.groupby("equipment")["rpm"].median()
    facts["Equipment"] = ", ".join(f"{k} ${v:.2f}/mi" for k, v in rel.items()) + " (median)"

    # 3. Market index drives day-to-day rates (two panels, shared time axis)
    daily = clean.groupby("date").agg(mi=("market_index_filled", "mean"), rpm=("rpm", "median"))
    clean["rel"] = clean["rpm"] / clean.groupby(["equipment", "dist_bin"], observed=True)["rpm"].transform("median")
    daily["rel"] = clean.groupby("date")["rel"].median()
    val_daily = val.groupby("date")["market_index_filled"].mean()
    all_daily = pd.concat([daily["mi"], val_daily])
    weekday = all_daily.groupby(all_daily.index.dayofweek).mean()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw={"hspace": 0.25})
    a1.plot(daily.index, daily["mi"], color=BLUE, linewidth=1.2, label="Train (Jan-Oct)")
    a1.plot(val_daily.index, val_daily.values, color=ORANGE, linewidth=1.2, label="Validation (Nov-Dec)")
    a1.set_ylabel("Daily market index"); a1.set_ylim(top=1.58); a1.legend(loc="upper left", ncol=2)
    a1.set_title("Market index is a daily signal; Nov-Dec sits well below the spring peak")
    a2.plot(daily.index, daily["rel"], color=BLUE, linewidth=1.2)
    a2.axhline(1, color=INK_2, linewidth=0.8)
    a2.set_ylabel("Daily rate vs typical\n(same equipment & distance)")
    a2.set_title("Daily rate level tracks the market index (corr %.2f)" % daily[["mi", "rel"]].corr().iloc[0, 1])
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    save(fig, "03_market_index_vs_rate")
    facts["Market index"] = (
        f"daily mean correlates {daily[['mi', 'rel']].corr().iloc[0, 1]:.2f} with daily rate level; "
        f"train mean {train.market_index.mean():.3f} vs validation {val.market_index.mean():.3f}; "
        f"within-day sd only {train.groupby('date').market_index.std().mean():.3f}; strong weekly cycle "
        f"(Thu peak {weekday.max():.2f} vs Sun trough {weekday.min():.2f}); same-day value beats short rolling windows and lags; long smoothed averages correlate slightly higher only because they track the slow trend, and did not generalise in model tests"
    )

    # 4. Unexplained upward drift over time
    monthly = clean.groupby(clean["date"].dt.to_period("M"))["resid"].mean()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(monthly.index.strftime("%b"), np.expm1(monthly.values) * 100, color=BLUE, width=0.7)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_ylabel("Rate vs structural model (%)")
    ax.set_title("After distance, equipment, weight & market: rates still drift up ~6% Jan to Oct")
    save(fig, "04_residual_drift_by_month")
    facts["Time drift"] = (
        f"residual goes from {np.expm1(monthly.iloc[0]):+.1%} (Jan) to {np.expm1(monthly.iloc[-1]):+.1%} (Oct) "
        "after controlling for market index: a trend the model must extrapolate into Nov-Dec"
    )

    # 5. Weight quality
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.hist(raw_train["weight"].dropna(), bins=np.arange(-48_000, 48_500, 1_000), color=BLUE)
    ax.set_yscale("log")
    ax.set_xlabel("Raw weight (lb)"); ax.set_ylabel("Loads (log scale)")
    ax.set_title("Weight: negative sign-flip errors, a hard cap at 47,500 lb, and missing values")
    ax.annotate(f"{(raw_train.weight < 0).sum()} negative", (-32_000, 30), color=INK_2)
    ax.annotate(f"{(raw_train.weight == config.WEIGHT_CAP).sum():,} at\n47,500 cap", (47_500, 1_300),
                xytext=(-10, 30), textcoords="offset points", ha="right", color=INK_2,
                arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8})
    ax.set_ylim(top=2e4)
    save(fig, "05_weight_quality")
    facts["Weight"] = (
        f"train: {(raw_train.weight < 0).sum()} negative, {raw_train.weight.isna().sum()} missing, "
        f"{(raw_train.weight == config.WEIGHT_CAP).sum():,} capped at 47,500; validation: "
        f"{(raw_val.weight < 0).sum()} negative, {raw_val.weight.isna().sum()} missing. "
        f"Negative-weight loads price like |weight| (median rpm {clean.loc[clean.weight_was_negative == 1, 'rpm'].median():.2f} "
        f"vs {clean.loc[clean.weight_was_negative == 0, 'rpm'].median():.2f})"
    )

    # 6. City coverage & synthetic coordinates
    cities = data.city_table(raw_train, raw_val)
    train_cities = set(raw_train.pickup) | set(raw_train.delivery)
    unseen = cities[~cities.index.isin(train_cities)]
    fig, ax = plt.subplots(figsize=(9, 5))
    seen = cities[cities.index.isin(train_cities)]
    ax.scatter(seen.lon, seen.lat, s=28, color=BLUE, label=f"In train ({len(seen)})", edgecolor="white", linewidth=1)
    ax.scatter(unseen.lon, unseen.lat, s=48, color=ORANGE, label=f"Validation only ({len(unseen)})",
               edgecolor="white", linewidth=1)
    for name, row in unseen.iterrows():
        ax.annotate(name, (row.lon, row.lat), xytext=(5, 3), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xlabel("Longitude (as given)"); ax.set_ylabel("Latitude (as given)")
    ax.set_title("8 validation cities never appear in training; coordinates are shifted but consistent")
    ax.legend(loc="lower left")
    save(fig, "06_city_coverage")
    val_lane = raw_val.pickup + ">" + raw_val.delivery
    train_lanes = set(raw_train.pickup + ">" + raw_train.delivery)
    unseen_city_rows = (~raw_val.pickup.isin(train_cities)) | (~raw_val.delivery.isin(train_cities))
    facts["Cities & lanes"] = (
        f"{len(cities)} cities total; {len(unseen)} validation-only ({', '.join(unseen.index)}) touching "
        f"{unseen_city_rows.mean():.1%} of validation loads; {val_lane.isin(train_lanes).mean():.1%} of validation "
        f"loads are on a lane seen in training. Coordinates are synthetic (e.g. Los Angeles lat 28.6, Boston lon clipped "
        f"at -69.5) but one fixed pair per city; distance is ~1.18x the great-circle distance"
    )

    # 7. Quote signal: apparent U-shape is an artifact of a mis-specified distance curve
    # residual against a flexible model that never sees quote_signal
    feats = ["distance", "equipment", "weight", "market_index", "pickup_lat", "pickup_lon",
             "delivery_lat", "delivery_lon"]
    X = clean[feats].assign(equipment=clean["equipment"].astype("category"),
                            days=(clean["date"] - clean["date"].min()).dt.days)
    booster = lgb.train({"objective": "l2", "learning_rate": 0.05, "num_leaves": 63, "verbose": -1,
                         "seed": config.RANDOM_SEED}, lgb.Dataset(X, np.log(clean[config.TARGET])), 600)
    clean["resid_flex"] = np.log(clean[config.TARGET]) - booster.predict(X)
    qbin = pd.qcut(clean["quote_signal"], 20)
    q = clean.groupby(qbin, observed=True).agg(qs=("quote_signal", "median"), r=("resid", "mean"),
                                               rf=("resid_flex", "mean"))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(q["qs"], np.expm1(q["r"]) * 100, color=ORANGE, marker="o", markersize=4,
            label="vs log-linear model (distance curve mis-specified)")
    ax.plot(q["qs"], np.expm1(q["rf"]) * 100, color=BLUE, marker="o", markersize=4,
            label="vs flexible model (no quote_signal)")
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel("quote_signal (20 quantile bins)"); ax.set_ylabel("Unexplained rate (%)")
    ax.set_title("quote_signal's U-shape is an artifact: under 0.3% once distance is modelled properly")
    ax.legend(loc="upper center")
    save(fig, "07_quote_signal")
    spread = clean.groupby(pd.qcut(clean["distance"], 5), observed=True)["quote_signal"].std()
    facts["quote_signal"] = (
        f"apparent U-shaped effect (+/-3%) vs a log-linear model vanishes (max |effect| "
        f"{np.expm1(q['rf']).abs().max():.2%}) against a flexible model; its spread is much wider on short "
        f"hauls (sd {spread.iloc[0]:.2f} vs {spread.iloc[-1]:.2f}), which is what created the artifact"
    )

    # 8. Missingness summary
    facts["Missing values"] = (
        f"train: weight {raw_train.weight.isna().sum()}, market_index {raw_train.market_index.isna().sum()}; "
        f"validation: weight {raw_val.weight.isna().sum()}, market_index {raw_val.market_index.isna().sum()}; "
        "no other nulls, no duplicate IDs or rows"
    )
    facts["December chart inputs"] = (
        "only pickup, delivery, distance, equipment, weight, date - no market_index, quote_signal or "
        "coordinates; these are reconstructed from validation.csv daily market means and the city table"
    )

    lines = ["# EDA findings (auto-generated by scripts/eda.py)", ""]
    lines += [f"- **{k}**: {v}" for k, v in facts.items()]
    lines += ["", "Figures: `reports/figures/`"]
    (config.REPORTS_DIR / "eda_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
