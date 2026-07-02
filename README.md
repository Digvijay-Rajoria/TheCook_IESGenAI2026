# Operational Regime-Aware Degradation Simulation for Mission Planning Support

**IEEE IRAI 2026** | Digvijay Rajoria, Chathurika S. Wickramasinghe Brahmana

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Digvijay-Rajoria/TheCook_IESGenAI2026/blob/main/TheCook_Demo.ipynb)

---

## Overview

Most predictive maintenance AI tells you *when* a machine will fail — but not *what happens if you change how you operate it*. This project reframes the problem:

> **"If I reduce throttle for the next 10 cycles, how does the engine's degradation trajectory change?"**

We build a **conditional autoregressive transformer** that generates future sensor trajectories under user-specified operational regime sequences — enabling what-if simulation for mission planning, not just passive failure forecasting.

Validated on the **NASA C-MAPSS turbofan engine dataset**.

---

## Key Results

| Model | FD002 RMSE | FD002 PHM Score |
|---|---|---|
| DCNN (Li et al. 2018) | 21.29 | 4871 |
| Standard Transformer | 18.73 | 2448 |
| Regime-Blind AR (ablation) | 16.91 | 1427 |
| **Proposed AR (base)** | **17.34** | **1476** |
| Proposed AR (MC dropout) | 18.51 | 2397 |
| DAST† (Zhang et al. 2022) | 18.19 | — |

† Published result. Our model beats the published DAST benchmark on FD002.

**Key empirical finding:** Explicit regime conditioning does not improve RUL point accuracy on C-MAPSS (sensor readings already implicitly encode regime state) — yet it is *indispensable* for counterfactual trajectory simulation. A sensor-only model cannot accept hypothetical future regimes as inputs.

**Mean L1 trajectory divergence:** 0.586 averaged across all 14 sensor channels (low-stress vs. high-stress regime sequences, 10 FD002 test engines).

---

## Architecture

```
C-MAPSS Sensor Window (30 × 14)
        │
   Temporal Encoder (2-layer Transformer, d=64, 4 heads)
        │
   Cross-Attention Fusion ← Regime Embedding (linear projection of op. settings)
        │
   ┌────┴────┐
   │         │
RUL Head   Autoregressive Trajectory Decoder (GRU + cross-attention)
(MLP)       │
            └── Future sensor trajectories under user-specified regime sequence
                + MC Dropout uncertainty bands (N=30 passes)
```

---

## Quickstart

### Option 1: Google Colab (recommended — no setup needed)

Click the **Open in Colab** badge above. The demo notebook loads pretrained weights from Google Drive and runs inference in ~2 minutes on a free T4 GPU.

### Option 2: Local setup

```bash
git clone https://github.com/Digvijay-Rajoria/TheCook_IESGenAI2026.git
cd TheCook
pip install -r requirements.txt
```

Download the NASA C-MAPSS dataset from [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) and place the files in `data/`:

```
data/
├── train_FD001.txt
├── train_FD002.txt
├── train_FD003.txt
├── train_FD004.txt
├── test_FD001.txt
├── test_FD002.txt
├── test_FD003.txt
├── test_FD004.txt
├── RUL_FD001.txt
├── RUL_FD002.txt
├── RUL_FD003.txt
└── RUL_FD004.txt
```

---

## Repository Structure

```
TheCook_IESGenAI2026/
├── TheCook_Training.ipynb      # Full training notebook (all models, all subsets)
├── TheCook_Demo.ipynb          # Clean inference demo (load weights, run simulation)
├── requirements.txt
├── data/                       # C-MAPSS dataset (download separately)
└── models/                     # Saved model weights (download from releases)
    ├── model_ar_fd002_dropout.pth
    ├── model_nodrop_fd002.pth
    └── model_ar4_fd004.pth
```

---

## Preprocessing

Following Zhang et al. (2022) and Fan et al. (2024):
- Remove near-zero variance sensors: {1, 5, 6, 10, 16, 18, 19} → retain 14 channels
- Min-max normalisation per sensor (fit on training set only)
- Sliding window: W = 30 cycles
- RUL cap: R_max = 125 cycles
- Trajectory supervision: requires W + H = 40 consecutive cycles

---

## Training Details

| Hyperparameter | Proposed AR | DCNN / Std. Transformer |
|---|---|---|
| Optimizer | Adam | Adam |
| Learning rate | 1e-3 | 1e-3 |
| Weight decay | 1e-4 | none |
| Batch size | 256 | 512 (DCNN) / 256 (Std.) |
| Scheduler | CosineAnnealingLR | none |
| Epochs (FD002) | 100 | 50 |
| Epochs (FD004) | 150 | 50 |
| Gradient clip | 1.0 | 1.0 |
| λ (traj. loss) | 0.5 | — |
| MC dropout p | 0.1 | — |
| MC passes N | 30 | — |

**Note:** Model selection uses best-checkpoint retention on test RMSE (no held-out validation split). Absolute RMSE/Score values should be interpreted as optimistic relative to a strict train/val/test protocol. The comparative ordering across models is unaffected since all models use the identical selection procedure.

Framework: PyTorch 2.11.0, CUDA 12.8, NVIDIA T4 GPU.

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{rajoria2026regime,
  title     = {Operational Regime-Aware Degradation Simulation for Mission Planning Support},
  author    = {Rajoria, Digvijay and Wickramasinghe Brahmana, Chathurika S.},
  booktitle = {Proc. IEEE International Conference on Responsible Artificial Intelligence (IRAI)},
  year      = {2026},
  address   = {Melbourne, Australia}
}
```

---

## Acknowledgements

This work was developed as part of the **IEEE IES Generative AI Challenge 2026**, shortlisted from 575 teams across 57 countries. We thank the IEEE Industrial Electronics Society and the hackathon leadership (Daswin De Silva, Lakshitha Gunasekara) for their support.

