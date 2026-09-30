"""Build reports/report.pdf from the generated figures and results.

Run after train/predict/score/insights:  python scripts/build_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from freight import config

TEAL = colors.HexColor("#064A56")
GRID = colors.HexColor("#D9E2E4")
SHADE = colors.HexColor("#F2F6F7")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, textColor=TEAL, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, textColor=TEAL, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.5, leading=13.5, alignment=TA_LEFT)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.5, leading=11, textColor=colors.HexColor("#455A60"))
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8.5, leading=11)
WIDTH = A4[0] - 3.6 * cm


def p(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [p(f"&bull;&nbsp;&nbsp;{t}") for t in items]


HEADER = ParagraphStyle("Header", parent=CELL, textColor=colors.white, fontName="Helvetica-Bold")
BOLD = ParagraphStyle("Bold", parent=CELL, fontName="Helvetica-Bold")


def table(rows, widths, bold_last=False):
    data = [[p(str(c), HEADER if i == 0 else BOLD if bold_last and i == len(rows) - 1 else CELL) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), TEAL), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SHADE]),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    t.setStyle(TableStyle(style))
    return t


def figure(name, width=WIDTH, caption=None):
    path = config.FIGURES_DIR / name if not Path(name).is_absolute() else Path(name)
    img = Image(str(path))
    img.drawWidth, img.drawHeight = width, width * img.imageHeight / img.imageWidth
    parts = [img]
    if caption:
        parts.append(p(caption, SMALL))
    return KeepTogether(parts + [Spacer(1, 6)])


def results_rows():
    r = pd.read_csv(config.REPORTS_DIR / "experiments.csv")
    r = r[r.test_rows == "clean"]
    keep = ["Baseline: rate-card median", "Log-linear OLS + trend", "LightGBM (trend=none)",
            "LightGBM (trend=feature)", "LightGBM (trend=detrend) [no label cleaning]", "LightGBM (trend=detrend)",
            "LightGBM (trend=hybrid)", "Hybrid, linear trees (single model)"]
    final = [m for m in r.model.unique() if m.startswith("FINAL")]
    labels = {
        "Baseline: rate-card median": "Rate-card baseline (median $/mi by equipment x distance)",
        "Log-linear OLS + trend": "Log-linear regression + trend",
        "LightGBM (trend=none)": "LightGBM, no time feature",
        "LightGBM (trend=feature)": "LightGBM, time as a feature",
        "LightGBM (trend=detrend) [no label cleaning]": "LightGBM, detrended, without label cleaning",
        "LightGBM (trend=detrend)": "LightGBM, detrended",
        "LightGBM (trend=hybrid)": "Hybrid: linear market part + standard trees",
        "Hybrid, linear trees (single model)": "Hybrid with linear trees",
    }
    folds = list(dict.fromkeys(r.fold))
    rows = [["Model"] + [f.split(":")[0] for f in folds] + ["Mean MAPE", "Mean MAE"]]
    for m in keep + final:
        g = r[r.model == m].set_index("fold")
        name = "FINAL: hybrid, linear trees, 5 seeds, lane premium" if m in final else labels[m]
        rows.append([name] + [f"{g.loc[f, 'MAPE']:.2%}" for f in folds]
                    + [f"{g.MAPE.mean():.2%}", f"${g.MAE.mean():.2f}"])
    return rows


def main():
    doc = SimpleDocTemplate(str(config.REPORTS_DIR / "report.pdf"), pagesize=A4, leftMargin=1.8 * cm,
                            rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title="Freight Rate Prediction - Assessment Report")
    s = []
    s.append(p("Freight Rate Prediction", ParagraphStyle("T", parent=H1, fontSize=20, spaceAfter=2)))
    s.append(p("Machine Learning Engineer assessment - approach, validation and results", SMALL))
    s.append(Spacer(1, 8))

    summary = table([
        ["Question", "Answer"],
        ["Task", "Predict posted_rate for 12,000 loads in Nov-Dec 2025 from 48,000 labelled loads (Jan-Oct 2025)"],
        ["Split", "Time-based, expanding window; each fold predicts the next two months (no random split)"],
        ["Model", "log(rate) = linear market/calendar part + LightGBM (linear trees) on load features "
                  "+ shrunk lane premium"],
        ["Accuracy on unseen months", "1.50% MAPE, $35.6 MAE (rate-card baseline: 4.15%, $96.0); "
                                      "89% of loads within 3%"],
        ["Key insight", "Rates rise 2-8% over the last month of every quarter (strongest for Flatbed); "
                        "December is a quarter end"],
    ], [3.6 * cm, WIDTH - 3.6 * cm])
    s += [summary, Spacer(1, 6)]

    # 1. data
    s.append(p("1. Data and key findings", H1))
    s += bullets([
        "<b>Scoring data is the future.</b> train_test.csv covers 2025-01-01 to 2025-10-31; validation.csv covers "
        "2025-11-01 to 2025-12-31. This is a forecasting problem, which drives the whole validation design.",
        "<b>Pricing structure.</b> Rate per mile falls with distance (rate ~ distance<super>0.87</super>). "
        "Reefer is ~13% and Flatbed ~8% above Dry Van at every distance; ~+3% per 10,000 lb.",
        "<b>Market index is a daily signal</b> (within-day std 0.025) with a strong weekly cycle (Thursday peak, "
        "Sunday trough). Its daily mean correlates 0.71 with the daily rate level; a 10% higher index adds ~1.5%.",
        "<b>Quarter-end ramp.</b> After removing distance, equipment, weight and market effects, rates are flat for "
        "two months of each quarter, then climb ~5% over the last month and reset when the next quarter opens. "
        "Seen in Q1, Q2 and Q3 alike.",
        "<b>quote_signal is a trap.</b> Against a log-linear model it shows a +/-3% U-shape; against a flexible model "
        "the effect is under 0.3%. The shape came from short hauls, where quote_signal is much more spread out.",
    ])
    s.append(figure("02_rate_per_mile_vs_distance.png", WIDTH * 0.82))
    s.append(figure("03_market_index_vs_rate.png", WIDTH * 0.9))

    s.append(p("2. Data-quality issues", H1))
    s.append(table([
        ["Issue", "Size", "How it was handled"],
        ["Corrupted posted_rate labels", "677 (1.41%)",
         "Ratio to a robust expected rate (median $/mi per equipment x distance band x month). Clean loads sit in "
         "0.8-1.2x; corrupted ones in 0.16-0.45x and 2.1-5.2x with nothing in between. Cut at 0.6x / 1.6x, "
         "removed from training."],
        ["Negative weights", "292 train, 145 val", "Sign-flip errors (they price like the absolute value): abs() + flag"],
        ["Weight capped at 47,500 lb", "1,191", "Kept, flagged"],
        ["Missing weight", "300 / 165", "Training median + flag"],
        ["Missing market_index", "374 / 249", "Mean of all loads on that date (daily signal)"],
        ["Synthetic coordinates", "all 72 cities", "Shifted but consistent per city; used as relative location only"],
        ["Cities only in validation", "8 cities, 12% of val loads",
         "No city or lane IDs in the trees; tested by hiding 8 cities: 1.9% vs 1.7% error"],
        ["December inputs lack market_index and coordinates", "31 rows",
         "Coordinates from the city table; market index = that day's mean in validation.csv (features only)"],
    ], [4.2 * cm, 2.9 * cm, WIDTH - 7.1 * cm]))
    s.append(Spacer(1, 6))
    s.append(figure("01_label_corruption.png", WIDTH * 0.8))

    # 3. validation (required)
    s.append(PageBreak())
    s.append(p("3. Validation and train/test split", H1))
    s.append(p("<b>Why not a random split.</b> The loads to predict come from the two months after the labelled "
               "data. A random split would put November-like days (and their market states) on both sides and reward "
               "models that memorise days, overstating accuracy. Every fold therefore trains on an expanding window "
               "and tests on the next two months only."))
    s.append(Spacer(1, 4))
    s.append(table([
        ["Fold", "Train", "Test", "Purpose"],
        ["Q2", "Jan - Apr", "May - Jun", "Same shape as the real task: train through month 1 of a quarter, "
                                         "predict months 2-3 (Oct -> Nov-Dec)"],
        ["Q3", "Jan - Jul", "Aug - Sep", "Same shape as the real task"],
        ["X1", "Jan - Jun", "Jul - Aug", "Stress test: predict into a quarter never seen"],
        ["X2", "Jan - Aug", "Sep - Oct", "Most recent two months"],
    ], [1.3 * cm, 2.2 * cm, 2.2 * cm, WIDTH - 5.7 * cm]))
    s.append(Spacer(1, 6))
    s += bullets([
        "<b>Label cleaning inside the split.</b> Corrupted labels are removed from training rows only. Metrics are "
        "reported on clean test labels (true model accuracy) and on all test labels (what a noisy scoring set shows).",
        "<b>Number of trees.</b> Early stopping on the latest 20% of each training window (time-ordered, never the "
        "test months), then refit on the full window with 15% more rounds.",
        "<b>Model selection</b> used the mean over all four folds and required gains to hold on the two "
        "production-like folds; changes that only moved error between folds were rejected (e.g. splitting the "
        "market index into smooth and weekly parts).",
        "<b>Unseen cities.</b> 8 random cities hidden from training (3 draws): error on loads touching them "
        "1.85-2.05% vs 1.68% for known cities.",
        "<b>Uncertainty.</b> 90% range of -2.7% / +3.0% around the prediction, calibrated on Q2+Q3 and checked on "
        "the other folds (89.8% coverage on Sep-Oct; 77% when predicting into an unseen quarter).",
        "<b>Leakage checks.</b> No feature uses the target; lane premiums use out-of-fold residuals inside the "
        "training window; validation.csv contributes only features (cities, daily market index), never labels. "
        "Unit tests assert that no fold trains on dates after its test window.",
    ])

    s.append(PageBreak())
    s.append(p("4. Model", H1))
    s.append(p("log(rate) = <b>market part</b> (linear: market index elasticity, quarter-end ramp by equipment, "
               "monthly drift) + <b>load part</b> (LightGBM with linear trees on distance, equipment, weight, "
               "coordinates, circuity and weight flags) + <b>lane premium</b> (out-of-fold residual per lane, shrunk "
               "by n/(n+10)). Five seeds are averaged."))
    s.append(Spacer(1, 4))
    s += bullets([
        "Plain LightGBM was unstable across folds: the daily market value is unique per day, so the trees used it as "
        "a date ID and memorised day-level prices; with a time feature they held the last level flat. Moving all "
        "time-varying effects into a small linear part lets them extrapolate, and the trees only learn the stable "
        "'what does this load cost' structure.",
        "Linear trees fit the smooth distance/weight curves better than flat leaves (1.62% -> 1.53%).",
        "Remaining error: a second model trained to predict the final model's errors from all load features "
        "explains only ~3% of them, so what is left is close to the noise in negotiated spot prices.",
    ])
    s.append(Spacer(1, 4))
    rows = results_rows()
    s.append(table(rows, [6.3 * cm] + [1.45 * cm] * 4 + [1.75 * cm, 1.75 * cm], bold_last=True))
    s.append(p("MAPE on clean test labels per fold. Including the 1.4% corrupted test labels, MAPE is ~3.9% and RMSE "
               "~$620 for every reasonable model: those rates are 2-5x off and cannot be predicted.", SMALL))
    s.append(Spacer(1, 4))
    s.append(figure("08_error_distribution.png", WIDTH * 0.75))

    # 5. december (required)
    s.append(PageBreak())
    s.append(p("5. Fixed December prediction chart (score.py)", H1))
    s.append(figure(config.ROOT / "scorer_results" / "candidate_december.png", WIDTH))
    s += bullets([
        "Inputs are fixed (Lexington -> Fort Wayne, 360 mi, Dry Van, 32,000 lb); only the date changes. The file has no "
        "market index or coordinates, so the FeatureBuilder fills coordinates from the city table and the market "
        "index from that day's mean across validation.csv loads.",
        "<b>Weekly zig-zag:</b> the market index's weekly cycle (Thursday peaks).",
        "<b>Upward drift ($831 -> $882, +6%):</b> the quarter-end / year-end ramp plus the underlying monthly drift.",
        "<b>Sanity check:</b> real Dry Van loads on this lane in training ranged $758-$934 (e.g. $864 on 30 Oct at "
        "market index 1.01); the December predictions are consistent with that history.",
    ])
    s.append(figure("09_quarter_end_uplift.png", WIDTH * 0.72))

    s.append(p("6. Business use and limitations", H1))
    s += bullets([
        "<b>Quarter-end pricing:</b> on the last day of a quarter the model adds ~$164 to a median Flatbed load, "
        "~$110 to Reefer and ~$46 to Dry Van.",
        "<b>Quote ranges:</b> the 90% band (-2.7% / +3.0%) gives a floor and ceiling for negotiation.",
        "<b>Data guardrail:</b> the ratio check that found the corrupted labels can flag suspicious rates at entry.",
        "<b>Limitations:</b> no holiday effect can be learned from Jan-Oct, so Christmas week is priced at its market "
        "level (no holiday effect was visible around Memorial Day, July 4 or Labor Day either); the year-end ramp is "
        "assumed to behave like Q1-Q3, and the last 1-3 days of past quarters were under-predicted by ~0.5-1%; in production the daily market index would come "
        "from the market feed rather than from the scoring file.",
    ])
    s.append(Spacer(1, 6))
    s.append(p("Code, run instructions and a step-by-step analysis log: see the repository README and "
               "docs/ANALYSIS_LOG.md. The whole submission is reproduced with <b>python run_all.py</b>.", SMALL))
    doc.build(s)
    print(f"Wrote {config.REPORTS_DIR / 'report.pdf'}")


if __name__ == "__main__":
    main()
