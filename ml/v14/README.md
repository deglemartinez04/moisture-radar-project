# V14 feature tables

`dataset_A_all_useful.csv` and `dataset_B_dry_plus_new.csv` are copied unchanged from
`App/ml/` inside `Newest_App_V14_A_all_data.zip` / `Newest_App_V14_B_dry_plus_new.zip`
on the `AppV10` branch. They hold the 28-feature rows (real recordings) that the AppV10
logistic-regression models were trained on.

`comparison_report.txt` is the output of `ml/compare_models.py`, which compares that
model with this repo's Random Forest on these tables.
