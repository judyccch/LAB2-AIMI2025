# LAB2 - EEGNet

This script implements EEGNet and DeepConvNet in PyTorch for EEG-based brain–computer interface (BCI) classification.
It supports flexible configuration of activation functions, dropout rates, and learning rates, as well as an automatic hyperparameter sweep for large-scale experiments.

## Requirements

- Python 3.8+
- PyTorch、NumPy、Pandas、Matplotlib、tqdm

## Usage Examples

```bash
python3 ./lab2_EEG_classification/main.py --model eegnet --sweep -num_epochs 10
```
```bash
python3 ./lab2_EEG_classification/main.py --model deepconvnet --sweep
```
Run all combinations of activations × dropouts × learning rates.
Each run automatically saves:
```bash
runs/sweep_YYYYMMDD_HHMMSS/
├─ act-relu_d-0p30_lr-0p003/
│  ├─ train_accuracy.png
│  ├─ train_loss.png
│  ├─ test_accuracy.png
│  ├─ metrics.csv
│  └─ config.json
└─ summary.csv   # Overview of all runs
```


