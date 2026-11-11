# Blueberry SSC Prediction

PyTorch implementation for nondestructive blueberry soluble solids content (SSC) prediction using a self-supervised pretrained spectral Transformer (Mixed-Mask MAE + FruitConsistency). This repository provides fine-tuning and evaluation scripts for L25+F6 and L11.

## Requirements

Python 3.11.15 · PyTorch 2.9.1 (CUDA 12.6). Other package versions are listed in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

## Usage

Place the pretrained encoder at `weight/encoder.pt` and the train/test CSV files under `data/`.

**Train:**

```bash
python L25F6.py
python L11.py
```

**Evaluate:**

```bash
python L25F6_test.py
python L11_test.py
```

Fine-tuned weights are saved to `weight/L25F6.pt` and `weight/L11.pt`. Evaluation reports **R², RMSE, and RPD** on both training and test sets, with outputs saved under `results/`.
