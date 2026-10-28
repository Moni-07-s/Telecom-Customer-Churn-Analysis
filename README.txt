# Telecom Customer Churn Dashboard Starter

## 1. Export dashboard files from Jupyter
Run the `integration_export_cell.txt` code in the same notebook after:
- `pipelines`, `best_name`, `X`, `y`, `df` exist
- `cv_df`, `test_results`, `importance_df` exist
- `recommend_retention_action` exists (optional)

The cell creates:
- dashboard_data.csv
- model_comparison.csv
- final_test_metrics.csv
- feature_importance.csv

Copy these CSVs into this folder beside `app.py`.

Note: the dashboard data uses a model refitted on all labelled rows for retrospective exploration.
The reported final test metrics still come from the held-out test set.

## 2. Install packages on Windows
Open Command Prompt in this folder and run:

    py -m pip install -r requirements.txt

## 3. Launch dashboard

    py -m streamlit run app.py

A browser tab should open automatically. If it does not, copy the local URL printed in the terminal.

## 4. Common problems
- `streamlit is not recognized`: use `py -m streamlit run app.py`.
- Missing CSV error: run the export cell and copy the generated CSVs into this folder.
- Missing column: ensure the notebook generated dashboard_data.csv using the supplied export cell.
- Do not report demo risk bands as validated thresholds; they are illustrative.
