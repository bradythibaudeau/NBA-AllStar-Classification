# NBA All-Star Classification

A binary classification project that predicts whether a player-season is labeled `ALL_STAR` using NBA season-average stats.

## Project Goal
Build and evaluate machine learning models that distinguish All-Star and non-All-Star player seasons, with a temporal holdout split for forward-looking evaluation.

## Repository Contents
- `all_star_classification.ipynb`: Main end-to-end analysis notebook (EDA, modeling, evaluation, interpretation)
- `all_star_classification_notes.md`: Implementation notes and latest summary metrics
- `nba_season_averages.py`: Data collection script using `nba_api`
- `requirements.txt`: Python dependencies
- `data/nba_player_season_averages_all_seasons.csv`: Main dataset used by the notebook
- `SEASONS/nba_player_season_averages_all_seasons.csv`: Additional copy of season averages

## Method Summary
- Target: `ALL_STAR`
- Features: Numeric season-average stats (excluding identifiers and leakage-prone metadata)
- Temporal split: Train on seasons before `2018-19`, test on `2018-19` and later
- Models compared:
  - DummyClassifier (most frequent)
  - LogisticRegression (class-weighted)
  - RandomForestClassifier (class-weighted)
  - GradientBoostingClassifier
- Model selection: 5-fold stratified CV
- Threshold tuning: Chosen from training-only cross-validated scores (no holdout label peeking)

## Setup
1. Create and activate a Python environment.

```bash
python -m venv venv
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run The Project
1. Open `all_star_classification.ipynb`.
2. Run all cells from top to bottom.
3. Review:
- Cross-validation summary table
- Holdout ROC-AUC / average precision
- Precision-recall, ROC, and confusion-matrix plots
- Per-season holdout analysis and error analysis tables

## Optional: Refresh Dataset From NBA API
If you want to regenerate the dataset:

```bash
python nba_season_averages.py
```

This script fetches regular-season and All-Star splits by season, labels players, and writes:
- `data/nba_player_season_averages_all_seasons.csv`

## Latest Reported Results
From the latest project notes:
- Best model by CV ROC-AUC: Logistic Regression
- Holdout ROC-AUC: 0.9764
- Holdout average precision: 0.7199
- Optimized threshold (training-CV selected): 0.9215
- Holdout class-1 metrics at optimized threshold:
  - Precision: 0.610
  - Recall: 0.869
  - F1: 0.717

## Important Caveat
The model uses full-season averages, while All-Star selections occur during the season. Results should be interpreted as post-hoc classification quality, not strict pre-event forecasting.
