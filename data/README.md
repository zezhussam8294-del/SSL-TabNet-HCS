# Data and Open-Set Split Preparation

This directory documents the datasets and data splits used to reproduce the experiments reported in the SSL-TabNet-HCS study.

Two Software-Defined Networking (SDN) intrusion-detection datasets were used:

1. **InSDN**
2. **SDNFlow**

The experiments follow an open-set evaluation protocol in which only known traffic classes are used during supervised training and validation, while unseen attack classes appear only during the open-set test stage.

---

## 1. InSDN

### Source

The InSDN dataset was introduced in:

M. S. Elsayed, N.-A. Le-Khac, and A. D. Jurcut,
**"InSDN: A Novel SDN Intrusion Dataset,"**
IEEE Access, vol. 8, pp. 165263–165284, 2020.
DOI: https://doi.org/10.1109/ACCESS.2020.3022633

Users should obtain the original InSDN dataset from its original/public source.

### Split-generation script

The repository provides the script:

`prepare_insdn_splits.py`

which generates the four data subsets used by the proposed method.

The split uses:

* Random seed: `42`
* Known classes: `Normal`, `DDoS`, `Probe`
* Unseen attack classes: `DoS`, `BFA`, `U2R`

For the known classes, the data are divided into approximately:

* 60% training
* 20% validation
* 20% testing

All samples belonging to the unseen attack classes are reserved exclusively for the open-set test set.

### InSDN split composition

| Split                       | Class distribution                                                     |
| --------------------------- | ---------------------------------------------------------------------- |
| Self-supervised pretraining | Normal = 41,053                                                        |
| Fine-tuning training        | DDoS = 44,117; Normal = 41,053; Probe = 37,054                         |
| Validation                  | DDoS = 14,706; Normal = 13,685; Probe = 12,351                         |
| Open-set test               | DDoS = 14,706; Normal = 13,685; Probe = 12,352; unseen attacks = 1,457 |

The unseen InSDN attacks consist of:

* DoS
* BFA
* U2R

In the generated open-set data file, unseen attacks are represented using the `OpenSetLabel` field. During final model evaluation, these samples are reported as **New Attack**.

The generated files are:

```text
pretrain_normal_train.csv
finetune_train.csv
finetune_val.csv
open_test.csv
```

---

## 2. SDNFlow

### Source

The SDNFlow dataset is described in:

J. Buzzio-García et al.,
**"Exploring Traffic Patterns Through Network Programmability: Introducing SDNFLow, a Comprehensive OpenFlow-Based Statistics Dataset for Attack Detection,"**
IEEE Access, vol. 12, pp. 42163–42180, 2024.
DOI: https://doi.org/10.1109/ACCESS.2024.3378271

The dataset used in this study was obtained from its original dataset source.

The SDNFlow data files are **not redistributed in this repository**. Researchers who wish to reproduce the SDNFlow experiments should obtain the dataset from its original provider and prepare the required data subsets according to the composition documented below.

### Open-set configuration

Known classes:

* Normal
* DDoS
* Probe

Unseen attack classes:

* DoS
* Password-Guessing
* SQL Injection
* U2R

The unseen attack categories are grouped together as **New Attack** during open-set evaluation.

### SDNFlow split composition

| Split                       | Class distribution                                                  |
| --------------------------- | ------------------------------------------------------------------- |
| Self-supervised pretraining | Normal = 177,251                                                    |
| Fine-tuning training        | Normal = 132,938; Probe = 90,874; DDoS = 44,786                     |
| Validation                  | Normal = 44,313; Probe = 30,291; DDoS = 14,929                      |
| Open-set test               | Normal = 44,313; Probe = 30,291; DDoS = 14,929; New Attack = 43,033 |

The four files expected by the SDNFlow experiment script are:

```text
pretrain_normal_train.csv
finetune_train.csv
finetune_val.csv
open_test.csv
```

Because the exact SDNFlow source data are subject to the access conditions of the original provider, users should obtain the source dataset directly from the original repository/provider rather than from this GitHub repository.

---

## 3. Open-Set Evaluation Protocol

For both datasets, the experimental protocol follows the same principle:

* Self-supervised pretraining uses only Normal traffic.
* Supervised fine-tuning uses only known classes.
* Validation contains only known classes and is used for rejection-threshold calibration.
* Unseen attack classes are not used during training or threshold calibration.
* Unseen attacks are introduced only in the final open-set test set.
* All unseen attack categories are reported collectively as **New Attack** during final evaluation.

This separation ensures that the reported New Attack detection performance is evaluated on attack categories that were not used during model training.

---

## 4. Expected Data Structure

After preparing the datasets, the experiment scripts expect the following structure:

```text
data/
├── InSDN/
│   ├── pretrain_normal_train.csv
│   ├── finetune_train.csv
│   ├── finetune_val.csv
│   └── open_test.csv
│
└── SDNFlow/
    ├── pretrain_normal_train.csv
    ├── finetune_train.csv
    ├── finetune_val.csv
    └── open_test.csv
```

The InSDN split-generation script is provided in this repository. SDNFlow data must be obtained separately from the original data source.
