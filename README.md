# ChainCare: Clinical Prediction via Bidirectional Monitoring Event Chains and Multi-Event Time Series Modeling

This repository contains the official implementation of our paper **"ChainCare: Clinical Prediction via Bidirectional Monitoring Event Chains and Multi-Event Time Series Modeling"**.  
ChainCare constructs bidirectional monitoring-level event chains and performs multi-event time series modeling to identify critical clinical indicators over time, achieving robust and interpretable clinical prediction performance.

---

## 🗂️ Repository Structure

```
├── data/                     # Dataset folder
│   └── raw_data/             # Place the original datasets here
│
├── models/                   # Model architectures and checkpoints
│
├── preprocess/               # Data preprocessing scripts and logic
│
├── main.py                   # Main entry point for training and evaluation
├── trainer.py                # Model training and evaluation pipeline
├── utils.py                  # Common utilities and helper functions
├── Task.py                   # Task configuration and management
│
├── OverWrite_base_ehr_dataset.py
├── OverWrite_data.py
├── OverWrite_mimic3.py
├── OverWrite_mimic4.py
├── OverWrite_sample_dataset.py
│
├── .gitignore
└── README.md
```

---

## 📊 Dataset Preparation

1. Download the required clinical datasets (e.g., MIMIC-III, MIMIC-IV).  
2. Place all raw data files in:

```
data/raw_data/
```

3. The preprocessing scripts in `preprocess/` will automatically handle data formatting, filtering, and feature extraction.

---

## 🚀 Run the Model

You can start training and evaluation with the following command:

```bash
python main.py --device_id 0
```

Other optional arguments (e.g., learning rate, epochs, batch size) can be adjusted inside `main.py` or passed via command line.

---

## 📁 Output

- Trained model checkpoints and logs will be saved under `logs/`.
- Processed datasets will be stored in `data/` for reuse.

---

⭐ **If you find this project helpful, please consider giving it a star!**