# jepa-edge-sensor

A proof-of-concept Edge AI experiment inspired by Yann LeCun's **Joint Embedding Predictive Architecture (JEPA)**. A small MLP predicts the next sensor embedding in latent space — never raw sensor values — trained on synthetic multi-signal data and designed to run on constrained devices such as the NVIDIA Jetson Orin Nano 8 GB.

---

## Overview

| Property | Value |
|---|---|
| Hardware target | NVIDIA Jetson Orin Nano 8 GB |
| Dev platform | MacBook Pro M5 Max · PyTorch MPS |
| Implementation | Single file — `jepa_edge_sensor.py` |
| Total parameters | 7,376 |
| External APIs | None |

The core JEPA idea applied here: the training loss is computed entirely in **embedding space**. The predictor never sees raw sensor values as a target; it learns to predict *where the next state will be in the latent manifold*, not what the raw readings will be.

---

## Architecture

```
raw sensors (3,)
      │
      ▼
 SensorEncoder          Linear · 3 → 16              64 params
      │
      ├──────────────────────────────────────────────┐
      │                                              │
      ▼                                       VectorStore
 emb_current (16,)              cosine-similarity retrieval
      │                         over 50-step rolling window
      │◄──── retrieved_context (16,) ───────────────┘
      │
      ▼
 concat(emb_current, retrieved)   →  (32,)
      │
      ▼
 JEPAPredictor          Linear 32→64 · ReLU           7,312 params
                        Linear 64→64 · ReLU
                        Linear 64→16
      │
      ▼
 pred_emb (16,)
      │
      ▼
 Loss = MSE(pred_emb, actual_next_emb)   ← latent space only
```

**Encoder** is a single linear layer kept deliberately simple so the embedding space has a closed-form pseudo-inverse for decoding back to sensor units during evaluation.

**Vector store** maintains a FIFO buffer of the last 50 embeddings. Before each prediction, the most cosine-similar past embedding is retrieved and concatenated with the current embedding as temporal context (LangGraph-style memory node).

---

## Synthetic Sensor Signals

| Signal | Profile | Range |
|---|---|---|
| Temperature | Sinusoidal (period ≈ 200 s) | 25 ± 5 °C |
| Pressure | Linear drift | 1000 → 1050 hPa |
| Battery | Exponential decay | 100 % → ~60 % |

1 000 timesteps, 1-second sampling interval, small relative Gaussian noise on each signal.

---

## Results

| Metric | Value |
|---|---|
| Final training loss (latent MSE) | 0.002582 |
| Mean inference latency | ~1.0 ms / step |
| Peak memory (Python heap) | 0.17 MB |
| Total model parameters | 7,376 |
| Training epochs | 50 |
| Optimiser | Adam · lr = 1 × 10⁻³ |

Training converges in the first ~10 epochs; the remaining 40 epochs refine to a stable floor.

---

## How to Run

```bash
git clone https://github.com/hellojais/jepa-edge-sensor.git
cd jepa-edge-sensor

pip install -r requirements.txt

python3 jepa_edge_sensor.py
```

No GPU required. The script auto-selects MPS → CUDA → CPU.

---

## Outputs

All artefacts are written to `outputs/` on the first run.

| File | Description |
|---|---|
| `sensor_dataset.csv` | 1 000-row table: timestep, raw sensor values, 16-dim embeddings |
| `jepa_edge_sensor.pt` | Saved encoder + predictor weights and normalisation stats |
| `training_loss.png` | MSE loss per epoch (latent space) |
| `predicted_vs_actual.png` | Predicted vs actual for all 3 sensor signals |

---

## License

MIT — see [LICENSE](LICENSE).
