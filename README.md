# Cloud‑Workload‑Forecasting‑Baselines
This repository contains the implementation of my master's research for **short‑term & long‑term cloud workload forecasting**, including two proposed models **TFC** and **FDFormer**, as well as multiple mainstream time‑series SOTA baselines.



> 🔗 Related Papers (Under Review)
> 1. *Harnessing Time‑Frequency Collaboration for Short‑Term Cloud Workload Forecasting* (TFC, lightweight short‑term forecast)
> 2. *FDFormer: A Frequency‑Decoupled Dual‑Path Learning Framework for Long‑Term Cloud Workload Forecasting* (FDFormer, long‑term forecast)

## 📋 Project Overview
Cloud workload forecasting is the core foundation for cloud AIOps, supporting **predictive auto‑scaling (short‑term)** and **cluster long‑term capacity planning**.

- **TFC**: Lightweight time‑frequency collaboration model for short‑term cloud workload forecasting. Designed for online real‑time inference scenario. Only 3.43K parameters with sub‑millisecond inference latency. Validated on Alibaba Cluster Trace, Google Cluster Trace.
- **FDFormer**: Frequency‑decoupled dual‑path Transformer for long‑term cloud workload forecasting. Solves frequency interference in long‑horizon prediction. Validated on 4 real‑world ByteDance cloud datasets(FaaS/IaaS/PaaS/RDS).

This repo provides a unified experimental pipeline: data preprocessing, baseline models, model training, evaluation metrics and visualization scripts.

## 📂 Repository Structure
```
├── data_provider        # Dataset loading, data preprocessing pipeline
├── dataset_norm         # Time‑series data normalization & cleaning
├── exp                  # Training & evaluation experiment entry
├── layers               # Custom network layers (Frequency‑domain decomposition layers for FDFormer)
├── models               # Implemented models: TFC, FDFormer & SOTA baseline models
├── scripts             # Shell running scripts for reproduction
├── utils                # Metrics, loss function, visualization tools
├── run.py               # Main entry for model training
└── README.md
```


## 🛠️ Environment
python >=3.9
pytorch >=2.0
numpy, pandas, scipy, matplotlib

## 🚀 Quick Start
1. Prepare cloud trace datasets (Alibaba / Google / ByteDance cloud workload datasets)
2. Modify dataset path in config
3. Run training:
```bash
python run.py
```

## 📊 Datasets
- Public cluster traces: Alibaba Cluster Trace, Google Cluster Trace
- Industrial private dataset: ByteDance multi‑service cloud workload datasets(FaaS/IaaS/PaaS/RDS)


## 📝 Statement

This code corresponds to my master research work. Two papers are currently under review.






