# Verification

## Reproducible application

The maintained application was executed against the original 150-row synthetic CSV. It reports zero missing values and zero exact duplicate rows. Seed 42 yields a stratified 120/30 split with 10 held-out examples per category.

| Check / metric | Result |
|---|---|
| Logistic holdout accuracy | 30/30 correct |
| 95% Wilson interval | approximately 88.6%–100% |
| Training-only repeated CV | 15 folds, mean accuracy 100%, SD 0% |
| Prior-only dummy accuracy | 33.3% |
| Logistic holdout log loss | approximately 0.0626 |
| Multiple OLS holdout RMSE | approximately 0.655 score points |
| Multiple OLS holdout R² | approximately 0.904 |
| Full-sample descriptive OLS R² | approximately 0.911 |

Exact results, data fingerprint and package versions are stored in `reports/`. Training-fold permutation importance and four held-out regression comparisons are exported as CSV files.

Seven correctness tests cover deterministic outputs, split disjointness, malformed-data rejection, train-only scaler/coefficient fitting, CV/importance isolation from test values, portable-model agreement with scikit-learn, and small-sample uncertainty. Run `python -m unittest discover -s tests`.

The `pyproject.toml` package built successfully, and prediction was smoke-tested from the built package using the original CSV with explicit `--data`. Temporary package artifacts are not maintained as duplicate release assets. The IDE workspace and direct source commands remain the review entry points.

## Portfolio checks

The self-contained browser checker validates 24 full images at 1440 × 900 and 24 previews at 960 × 600. It checks exact README/site image order, captions and paths, internal anchors, relative file links, the three-row Live Links table, and seven clickable opening badges. The README opening block is centered, its contents use numbered lines, and all three Live Links display full HTTPS URLs. About the Author is its final section, after figure 24.

Browser scenarios use Microsoft Edge at 1440 × 1000, 820 × 1180 and 375 × 812 in both themes. Checks cover overflow, 3/2/1 gallery columns, theme controls, visible SVG icons, mobile navigation, explorer/Python probability parity, sample presets, extrapolation warnings, invalid inputs with stale predictions hidden, search, empty results, filter reset, filtered gallery wrapping, Previous/Next, arrow keys, Escape, zoom, simulated horizontal swipe, focus restoration, local downloads and Back to Top. Additional checks cover selectable Slate/Warm/Graphite gallery backgrounds, increased navigation type, the larger Back to Top control, and the final author section. Results are recorded in `reports/site_checks.json`.

## Saved Bayesian evidence

The Bayesian notebook preserves its supplied executed outputs, including the reported posterior intervals and sampler diagnostics. A fresh full resampling remains unverified: the previous Jupyter execution attempt failed when creating its protected kernel connection file, and the system PyMC/ArviZ major versions differ from the recorded notebook environment. Use the isolated dependencies in `requirements-bayesian.txt` for a new run. The enhanced baseline and figure build do not generate or validate new posterior samples.

## Scope limits

External hosting, real-device touch behavior, screen-reader behavior and a clean installation of the Bayesian environment have not been tested. Simulated swipe checks verify the event handling, not physical-device usability. Classification probabilities are not claimed to be calibrated for real cohorts. These limitations do not affect the reproducible local baseline or the preserved saved notebook evidence.

## Final consistency audit

The final audit checked 20 Markdown file links, local webpage resources and anchors, project configuration, source syntax, and notebook schema/code syntax. The maintained notebook contains eight Bayesian model definitions, 15 saved figure outputs and no saved error outputs. These are structural checks, not fresh Bayesian execution. The application and seven correctness tests were rerun successfully. See `reports/verification_summary.json` for the recorded audit scope.
