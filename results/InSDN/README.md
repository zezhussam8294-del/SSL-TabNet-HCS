
# InSDN Results

This directory contains the final experimental results of the proposed SSL-TabNet-HCS framework on the InSDN dataset.

## Final Open-Set Results

- Classification Accuracy (CA): 98.67%
- Macro Precision: 93.91%
- Macro Recall: 96.16%
- Macro-F1: 94.95%
- MCC: 98.07%
- New Attack Precision: 77.07%
- New Attack Recall: 87.44%
- New Attack F1-score: 81.93%

## Files

- `metrics_summary.json` – overall validation and open-set evaluation metrics and calibrated thresholds.
- `final_report_percent.txt` – final metrics, per-class performance, and confusion matrix.
- `confusion_matrix.csv` – final open-set confusion matrix.
- `per_class_report_percent.csv` – precision, recall, F1-score, and support for each class.

These results were generated using `insdn_ssl_tabnet_hcs.py`.
