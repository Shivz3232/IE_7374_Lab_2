# MLflow Use Cases

Two small scikit-learn projects that use MLflow for experiment tracking, model registry, model serving and nested runs.

| Notebook | Dataset | What it does |
|---|---|---|
| `default/starter.ipynb` | Wine Quality | Part 0: complete run of the provided starter code |
| `salary_prediction.ipynb` | `data/Position_Salaries.csv` | Part 1: train, register and serve a regression model |
| `social_network_ads.ipynb` | `data/Social_Network_Ads.csv` | Part 2: train/register a classifier, then compare k-fold CV settings with nested runs |

## Setup

Requires **Python 3.12** (older MLflow/Python combinations caused dependency conflicts).

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install mlflow==3.17.1 scikit-learn==1.9.1 pandas==3.0.6 numpy==2.5.4 skops==0.16.0 requests ipykernel jupyter
```

Run notebooks from the project root (paths like `data/...` are relative). MLflow stores runs in `mlflow.db` and `mlruns/` in the same directory; both are git-ignored and created on first run.

To browse runs and registered models:

```bash
mlflow ui --port 5000             # open http://localhost:5000
```

## Part 0 – Starter code

`default/starter.ipynb` contains the full run of the provided wine-quality lab. The rest of `default/` (data, requirements, docs) is git-ignored, so only the executed notebook is included for reference.

## Part 1 – Salary prediction (`salary_prediction.ipynb`)

Predicts salary from position level using a `DecisionTreeRegressor` (`max_depth=4`).

1. Run all cells. The notebook:
   - logs a run `untuned_decision_tree` with params (`max_depth`, `tree_depth`, `n_leaves`) and the R² `score`;
   - logs the model as a `pyfunc` and registers it as **`position_salaries`**;
   - reloads the latest registered version and prints its R² as a sanity check.
2. In a separate terminal (venv active, project root), serve the model:
   ```bash
   mlflow models serve --env-manager=local -m models:/position_salaries/None -h 0.0.0.0 -p 5001
   ```
3. Run the last cell, which POSTs levels `6.5`, `4.0`, `10` to `http://localhost:5001/invocations`.

**Expected output**

```
Score: 0.9998553639817056
{'predictions': [150000.0, 80000.0, 1000000.0]}
```

Registering again creates a new version (e.g. `Created version '2' of model 'position_salaries'`); this is normal.

## Part 2 – Social network ads (`social_network_ads.ipynb`)

Predicts whether a user purchased (`Purchased`) from `Age` and `EstimatedSalary`.

1. Splits data 75/25 and standardises features with `StandardScaler`.
2. Trains a `RandomForestClassifier` (`n_estimators=10`, `max_depth=4`), logs accuracy, precision, recall and F1 under run `simple_random_forest`, and registers it as **`random_forest`** (`mlflow.sklearn`, with `skops` trusted types).
3. Loads the registered model and, in a new experiment **`kfold_experiment`**, creates a parent run `kfold_comparison` with one nested child run per `k ∈ {3, 5, 7, 10, 15, 20}`. Each child logs `cv_folds`, the mean/std/min/max accuracy and each fold's accuracy (`fold_N_accuracy`).

**Expected result:** test accuracy of `0.94`, and mean CV accuracy of roughly 0.88–0.91 across all `k`, with spread growing as `k` increases. In the MLflow UI, `kfold_experiment` shows `kfold_comparison` with six expandable child runs (`k_3` … `k_20`). Select them and use **Compare** to see how mean accuracy and variance change with `k`.
