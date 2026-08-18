# SDNFlow Results

This directory contains the final experimental results of the proposed SSL-TabNet-HCS framework on the SDNFlow dataset.

## Final Open-Set Results

- Classification Accuracy (CA): 89.01%
- Macro Precision: 91.20%
- Macro Recall: 91.10%
- Macro-F1: 90.91%
- MCC: 84.97%
- New Attack Precision: 89.23%
- New Attack Recall: 75.28%
- New Attack F1-score: 81.66%

## Files

- `metrics_summary.json` – overall validation and open-set evaluation metrics and calibrated thresholds.
- `final_report_percent.txt` – final metrics, per-class performance, and confusion matrix.
- `confusion_matrix.csv` – final open-set confusion matrix.
- `per_class_report_percent.csv` – precision, recall, F1-score, and support for each class.

These results were generated using `sdnflow_ssl_tabnet_hcs.py`.
