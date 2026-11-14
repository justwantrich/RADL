# RADL: code for editorial and peer review

This package is supplied for confidential editorial and peer-review assessment. It is not a public release. Access controls depend on the submission platform; this statement does not assert that access is technically restricted. The authors retain rights in their original work. The included CUTS+ components are subject to the separate third-party MIT license in `dependency/LICENSE-CUTS-PLUS.txt`.

## Scope

`forecast/` contains the supplied RADL forecasting implementation, including input preparation, model, training, evaluation and component switches. `dependency/` contains the separate CUTS+ implementation and market-specific runner underlying the predictive-dependency analysis in Section 4.2. Neural Granger dependencies are predictive, not structural or intervention-based causal effects. CUTS+ is separate from the RADL forecasting model and does not supply its graph.

The RADL architecture processes historical prices, calendar gaps and price changes. Price-channel reversible normalization uses a last-observation anchor. A historical-state encoder adjusts variable gates and per-layer graph propagation strengths; temporal convolutions and window-specific and static graph supports feed the forecasting head. Training uses target Huber loss and a gate penalty, with validation MSE used for model selection. See the implementation for precise details and switches.

## Data and access

The Wind source data are proprietary and are not distributed here. The input CSV or Excel file must contain a date/index column followed by numeric columns: the target market first, then the 15 predictors. Use chronological, unique dates, aligned to the forecast origin using only observations available by that time. The files in this package do not independently establish that source timestamps were aligned correctly. Data acquisition, vendor identifiers, licensing and any external pre-alignment remain outside this package. Researchers must obtain authorized data access.

## Implementation and evaluation

The forecasting script uses train-split scaling, a rolling history, and the configurable price, calendar-gap and change channels. The supplied defaults enable training-only spike repair and a horizon-dependent Huber threshold; consult `forecast/config.py` and `forecast/load_data.py` for the actual implementation. These details, the split boundaries, seed, number of epochs and all overrides must be reconciled with the final manuscript and the original experiment records before asserting exact reproduction. In particular, the manuscript's stated adjacent-record Huber threshold and the supplied code's H-step threshold differ for horizons greater than one. The source configuration also sets early-stopping patience to 60, whereas the supplied supplementary table reports 40. This package does not determine which configuration produced the reported tables.

The Section 4.2 runner standardizes the chronological training segment and uses a 21-record input, one-step prediction, 200 epochs and 16 nodes under its defaults. It includes seed and regularization sweeps and a matched-length option. Gate probabilities and thresholded density are descriptive predictive-dependency summaries. The retained CUTS+ sampler passes probability pairs as logits and uses an effective sampling temperature of 1; these implementation semantics are retained. The included configuration file is used as a configuration template and the runner overrides the market settings.

## Dependencies

Install a compatible Python environment with the packages listed in `requirements.txt`. A CUDA-enabled PyTorch installation and supported GPU are required by the dependency-analysis runner. No environment or numerical execution was performed while assembling this package.

From `forecast/`, an authorized user can supply an input with `python RADL.py -data_path PATH_TO_INPUT.csv`. From `dependency/`, use `python run_markets.py --data-dir PATH_TO_DATA --output PATH_TO_OUTPUT`, where the data directory contains the four named market CSV files. These commands train models and write outputs when run; they are examples, not evidence that this package has been executed or validated on the manuscript data. Inspect configuration and data access before running.

## Omitted materials and limits

The package contains no raw data, trained weights, precomputed results, experimental logs or figures. It does not include implementations for every manuscript benchmark, the complete 72-comparison adjusted DM analysis, or the state-conditional bootstrap and its figure-data pipeline. Consequently, this package is suitable for inspecting the two included implementations but is not a complete reproduction package for every reported finding. Independent verification of all tables and figures requires the omitted analysis scripts, genuine aligned forecast records, exact experiment configurations and authorized data access. The authors should reconcile the reported state-conditional analysis with its actual forecast sources before submission.
