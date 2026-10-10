# Survival-Aware Fuzzy Virtual Sensing

Reproducible experiments for sparse Takagi–Sugeno portfolios under simultaneous sensor-group failures. The release contains the implementation, public-data archive snapshots, fixed partitions, protocol configuration, mathematical checks, numerical reference results, and trained neural weights for replay.

The method guarantees an unaffected expert after any two flagged group outages. Predictive accuracy still depends on the information in the surviving sensors. All six tasks and all baseline results are retained, including settings where another method or the training mean is more accurate.

## Quick start

Use Python 3.12 on a 64-bit Unix-like system. Install the recorded dependency versions in a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/src/download_data.py
python experiments/tests/test_invariants.py
```

`download_data.py` verifies and extracts the bundled source archives. An absent archive is retrieved from its recorded UCI URL. It then regenerates the deterministic partitions and metadata.

Run the complete experimental protocol, including fresh neural training:

```bash
python experiments/run_all.py --workers 6 --gpu-workers 2
```

The neural trainer uses the Metal backend when available and otherwise runs on CPU. Each CPU worker uses one numerical thread. The recorded system had 12 CPU cores, 18 GPU cores, and 18 GB unified memory. Worker counts can be reduced for smaller systems. Runtime and temporary storage depend on the platform; the generated model and prediction records can occupy several gigabytes.

For a numerical replay that keeps the published neural weights while refitting the CPU models:

```bash
python experiments/run_all.py --workers 6 --gpu-workers 2 --reference-neural
python experiments/src/check_results.py
```

Training is restartable: existing model records are reused. Use a fresh extraction for an independent full retraining run. Seeded GPU retraining can vary across accelerators and numerical libraries. The reference-weight option reproduces the reported neural functions on CPU. `check_results.py` compares all 28,220 outage scores by their task, seed, method, and mask identifiers at a default relative tolerance of 1e-5; timings are measured anew. `reference/results/` preserves the exact numerical release.

## Contents

```text
survival-aware-fuzzy-sensing/
├── README.md
├── LICENSE
├── DATA_LICENSES.md
├── CITATION.cff
├── requirements.txt
├── environment.json
├── manifest.sha256
├── data/
│   ├── download_manifest.json
│   ├── raw/                    # Original UCI ZIP snapshots
│   └── processed/              # Group definitions and fixed row partitions
├── experiments/
│   ├── run_all.py
│   ├── configs/final_protocol.json
│   ├── tests/test_invariants.py
│   └── src/
└── reference/
    ├── neural_weights/         # Portable network arrays and model metadata
    └── results/                # Recorded scores, diagnostics, and summaries
```

## Experimental modules

| File in `experiments/src/` | Purpose |
|---|---|
| `download_data.py` | Verify source checksums, extract data, and record partitions. |
| `data.py` | Define targets, predictors, sensor groups, and ordered splits. |
| `fuzzy.py` | Fit local TSK experts; select and gate sparse portfolios; calculate survival certificates. |
| `baselines.py` | Fit tree, imputation, ridge, and full/group-dropout TSK comparators. |
| `neural.py` | Train the group-dropout MLP with validation stopping. |
| `train.py` | Fit the fixed task–seed–model grid with parallel workers. |
| `extra_train.py` | Fit the group-dropout TSK comparator. |
| `evaluate.py` | Enumerate all zero- to three-group test masks and save predictions and errors. |
| `monitor.py` | Evaluate sustained bias with imperfect fault flags. |
| `synthetic.py` | Generate and evaluate the controlled-redundancy simulation. |
| `analyze.py` | Calculate per-mask, per-seed, and task summaries and paired block intervals. |
| `benchmark.py` | Audit coverage and measure serial batch inference time. |
| `neural_reference.py` | Restore the released neural arrays as executable model records. |
| `check_results.py` | Compare regenerated outage metrics with the release. |

The invariant checks verify failure-input invariance, the survival certificate, the residual-moment risk identity, the all-failed fallback, disjoint ordered partitions, target exclusion, and complete assignment of predictors to groups.

## Data and protocol

The six targets are air CO, air benzene, air NO2, turbine CO, turbine NOx, and appliance energy. The source datasets remain separate. Air tasks use eight inputs in six groups; turbine tasks use nine inputs in four functional groups; appliance energy uses 24 inputs in ten groups. Original dataset attributions and notices are in `DATA_LICENSES.md`.

Air and appliance records use 50% fitting, 15% calibration, 15% validation, and 20% final testing. Turbines use 2011–2012 for fitting, the two halves of 2013 for calibration and validation, and 2014–2015 for testing. Scaling uses fitting data only. Seeds are 11, 23, 37, 51, and 71.

Selection sees only clean inputs and single-group outages. The primary endpoint averages each mask's RMSE over every two-group outage and all fitting seeds. Triple-group outages provide an additional severity test. The block analysis uses 2,000 paired non-overlapping-block replicates at lengths 12, 24, and 48. Its intervals are conditional on the fitted models and observed test sequence.

Artificial failures delete every channel in a selected group simultaneously. The imperfect-monitor experiment adds a three-standard-deviation positive bias to two groups for 64-row episodes and varies sensitivity and false alarms. The synthetic experiment varies the information carried by redundant sensor groups independently of structural coverage.

## Results files

- `all_metrics.csv`: one row per task, seed, method, and outage mask, with physical-unit RMSE, MAE, normalized RMSE, and recorded fit time.
- `analysis/summary.csv`: task/method/severity averages, seed standard deviations, and worst-mask errors.
- `analysis/per_seed.csv` and `analysis/per_mask.csv`: the corresponding individual-seed and individual-mask summaries.
- `analysis/bootstrap.csv`: paired percentage improvements and conditional 95% intervals for each baseline and block length.
- `analysis/portfolios.csv`: selected supports, weights, rule counts, parameter counts, and exact certificates.
- `analysis/baseline_configuration.csv`: selected baseline hyperparameters and validation scores.
- `analysis/validation_selected_baseline.csv` and `analysis/selected_baseline_test.csv`: baseline choices made from validation scores and their subsequent test results.
- `coverage.csv`: counts of masks with no surviving expert and numbers of surviving experts.
- `monitor.csv`: errors for each prescribed monitor sensitivity and false-alarm setting.
- `synthetic.csv`: all controlled-redundancy errors by seed and mask.
- `timing.csv`: nine-repetition median and interquartile batch inference times for the fixed benchmark case.

Fuzzy fit-time records include fitting the candidate bank and all portfolio ablations. They are not the cost of deploying only the selected portfolio. The serial timing experiment evaluates only surviving selected experts for SFR and uses the same 1,024-row batch and first-two-group outage for each comparator.

## License and citation

The implementation is MIT licensed. Dataset attribution and retained source terms are given in `DATA_LICENSES.md`. `CITATION.cff` provides machine-readable software citation metadata.
