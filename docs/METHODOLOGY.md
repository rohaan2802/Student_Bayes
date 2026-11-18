# Methodology and reproducibility

The supplied Bayesian notebook and the enhanced baseline answer related questions with different models. Saved posterior results are not presented as freshly generated baseline estimates.

## Data boundaries

The CSV contains 150 synthetic records. The three habit variables are inputs. Quiz score is the continuous regression target; student type is the categorical target and is never an input to classification. All original values are retained. `source_row` is a zero-based data-row position excluding the CSV header.

## Bayesian study

The notebook defines eight models: joint bivariate Normal correlation; simple Gaussian, Student-t, quadratic and multiple regression; unpooled and hierarchical group means; and a reference-coded Softmax/Categorical classifier. Predictors are centered or standardized as stated in the notebook. Full-sample regression standardization serves a descriptive analysis. Classification standardization uses its training split only.

All reported Bayesian intervals are 94% highest-density intervals. Mean-curve intervals describe uncertainty about the fitted expectation, not uncertainty about a new observed score. The saved run uses seed 42, four chains, 1,500 tuning and 1,000 retained draws per chain. Its recorded environment is Python 3.12, PyMC 5.28.5 and ArviZ 0.23.4. An optional Nutpie sampler can accelerate compatible installations.

## Enhanced baseline

`src/analyze.py` is the maintained command-line application. A stratified seed-42 split reserves 30 rows and fits on 120. A scikit-learn pipeline keeps StandardScaler and logistic regression together, including inside cross-validation. Class order is exported explicitly rather than inferred from CSV column order.

Repeated stratified cross-validation uses 5 folds and 3 repeats on the training partition only. Fifteen fold scores are reported. These folds overlap across repeats; their SD summarizes variability and is not a confidence interval. No parameter selection uses the reserved test partition.

Classification reports accuracy, class-wise precision/recall/F1, log loss and multiclass Brier score. Brier is defined here as the mean sum of squared errors across all three class probabilities, with theoretical range 0–2. A class-prior dummy model supplies a simple reference. The 95% Wilson score interval gives binomial context for the small held-out sample; independence and representativeness cannot be guaranteed for synthetic data.

Permutation importance measures the validation log-loss increase after shuffling a feature. Five training-only validation folds each use 10 permutations. The CSV reports mean and population SD across 50 values per feature. Overlapping predictors may share information, so importance is not an independent or causal contribution.

Four fixed regression models share the same reserved rows: mean-only, study-only OLS, multiple OLS and standardized Ridge with alpha=1. RMSE, MAE and R² are held-out scores. The separate full-sample OLS fit and residual chart are descriptive; they must not be confused with the held-out comparison.

## Portable model and browser explorer

`reports/demo_model.json` contains training means, population-SD scales, class order, linear coefficients, intercepts and observed training ranges. Both Python prediction and the browser evaluate standardized inputs and stable Softmax (subtracting the largest score before exponentiation). Tests compare exported probabilities to scikit-learn on every test row.

The explorer runs locally without sending input values to a service. It flags inputs outside training ranges; it does not estimate out-of-distribution reliability. The fixed model is a frequentist baseline, not a posterior predictive calculator. Presets are observed category medians used to demonstrate the interface, not validation cases.

## Evidence and presentation

Figures 01–15 are extracted from saved notebook PNG outputs. Figure 16 onward is generated from the original CSV, the maintained application or its exported reports. Blank margins are trimmed, chart aspect ratios preserved, and plots fitted to a high-contrast white surface inside a 1440 × 900 frame. The same 960 × 600 previews, full files, captions and order are generated for README and website from one manifest. Figure regeneration does not resample Bayesian models.

The CSV fingerprint and baseline package versions are saved in `reports/baseline_environment.json`. Re-run the application before rebuilding the portfolio after analysis changes.
