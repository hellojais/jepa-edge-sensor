"""
jepa_edge_sensor.py
===================
Edge AI experiment inspired by Yann LeCun's Joint Embedding Predictive
Architecture (JEPA).

The system learns to predict the *next sensor embedding* from the *current
sensor embedding* — predictions live entirely in latent space, never in raw
sensor space. A small rolling vector store provides retrieved context for each
prediction (LangGraph-style memory).

Hardware target : NVIDIA Jetson Orin Nano 8 GB
Dev platform    : MacBook Pro M5 Max (128 GB)
Backend         : PyTorch — MPS (Apple Silicon) → CUDA → CPU
"""

import os
import time
import math
import tracemalloc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")   # headless — safe for Jetson / CI
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# ══════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

SEED              = 42
NUM_TIMESTEPS     = 1000        # total sensor readings
SENSOR_DIM        = 3           # temperature, pressure, battery
EMBED_DIM         = 16          # latent-space dimension
HIDDEN_DIM        = 64          # MLP hidden units per layer
NUM_EPOCHS        = 50
LR                = 1e-3
BATCH_SIZE        = 32
VECTOR_STORE_SIZE = 50          # rolling window of recent embeddings kept in memory
OUTPUT_DIR        = "outputs"

torch.manual_seed(SEED)
np.random.seed(SEED)

# Device: prefer Apple Silicon MPS, then CUDA, then CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"[Config]  Device            : {DEVICE}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# 1.  SYNTHETIC SENSOR DATA GENERATION
# ══════════════════════════════════════════════════════════════════

def generate_sensor_data(n: int = NUM_TIMESTEPS,
                         noise_std: float = 0.02) -> np.ndarray:
    """
    Synthesise 3 sensor signals sampled every 1 second for n steps:

      temperature : sinusoidal oscillation  (25 ± 5 °C,  period ≈ 200 s)
      pressure    : slow linear drift        (1000 → 1050 hPa)
      battery     : exponential decay        (100 % → ~60 % at t=1000)

    Small relative Gaussian noise is added to each signal.
    Returns ndarray of shape (n, 3), dtype float32.
    """
    t = np.arange(n, dtype=np.float64)

    temperature = 25.0 + 5.0 * np.sin(2.0 * math.pi * t / 200.0)
    pressure    = 1000.0 + 0.05 * t
    battery     = 100.0 * np.exp(-5e-4 * t)

    signals = np.stack([temperature, pressure, battery], axis=1)   # (n, 3)

    # Relative Gaussian noise (noise_std fraction of the signal value)
    signals += np.random.normal(0.0, noise_std, signals.shape) * signals

    return signals.astype(np.float32)


raw_data = generate_sensor_data()
print(f"[Data]    {raw_data.shape[0]} timesteps  ×  {raw_data.shape[1]} sensors")
print(f"          temperature  [{raw_data[:,0].min():.2f}, {raw_data[:,0].max():.2f}] °C")
print(f"          pressure     [{raw_data[:,1].min():.2f}, {raw_data[:,1].max():.2f}] hPa")
print(f"          battery      [{raw_data[:,2].min():.2f}, {raw_data[:,2].max():.2f}] %")


# ══════════════════════════════════════════════════════════════════
# 2.  PER-SIGNAL MIN-MAX NORMALISATION  →  values ∈ [0, 1]
# ══════════════════════════════════════════════════════════════════

data_min   = raw_data.min(axis=0)          # (3,)
data_max   = raw_data.max(axis=0)          # (3,)
data_range = data_max - data_min           # (3,)


def normalize(x: np.ndarray) -> np.ndarray:
    """Map raw sensor values to [0, 1] using training-set statistics."""
    return (x - data_min) / (data_range + 1e-8)


def denormalize(x: np.ndarray) -> np.ndarray:
    """Inverse of normalize — recover physical units."""
    return x * data_range + data_min


norm_data = normalize(raw_data)            # (1000, 3)


# ══════════════════════════════════════════════════════════════════
# 3.  SENSOR ENCODER   raw sensor values  →  latent embedding
#     A single linear layer — deliberately tiny for edge deployment.
# ══════════════════════════════════════════════════════════════════

class SensorEncoder(nn.Module):
    """
    Linear projection: R^{SENSOR_DIM} → R^{EMBED_DIM}.

    Kept linear so the embedding space has a simple geometric interpretation
    and so we can invert it analytically (via pseudo-inverse) during eval.
    """
    def __init__(self, in_dim: int = SENSOR_DIM, out_dim: int = EMBED_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)   # (batch, EMBED_DIM)


# ══════════════════════════════════════════════════════════════════
# 4.  JEPA PREDICTOR MLP
#     Predicts embedding[t+1] from (embedding[t] ⊕ retrieved context)
# ══════════════════════════════════════════════════════════════════

class JEPAPredictor(nn.Module):
    """
    Three-layer feedforward MLP (the "predictor" in JEPA terminology).

    Input  : concat(current_embedding, retrieved_embedding) — size 2·EMBED_DIM
    Output : predicted next embedding                       — size EMBED_DIM

    Loss during training = MSE( predicted_emb , actual_next_emb )
    All loss signals are computed in latent space; raw sensor values are
    never part of the training objective (core JEPA philosophy).
    """
    def __init__(self, embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (batch, EMBED_DIM)


# ══════════════════════════════════════════════════════════════════
# 5.  IN-MEMORY VECTOR STORE   (cosine-similarity retrieval)
#     Inspired by LangGraph-style memory nodes.
# ══════════════════════════════════════════════════════════════════

class VectorStore:
    """
    A rolling window buffer of recent embeddings.

    Before each prediction the store is queried for the most similar past
    embedding (cosine similarity).  That retrieved vector is concatenated with
    the current embedding to give the predictor additional temporal context.

    All tensors are stored on CPU to minimise device-memory pressure.
    """

    def __init__(self, max_size: int = VECTOR_STORE_SIZE):
        self.max_size = max_size
        self.store: list = []           # list of 1-D float32 CPU tensors

    def add(self, embedding: torch.Tensor) -> None:
        """Add embedding to the store, evicting the oldest if at capacity."""
        vec = embedding.detach().cpu().float()
        self.store.append(vec)
        if len(self.store) > self.max_size:
            self.store.pop(0)           # FIFO eviction

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """
        Return the stored embedding most cosine-similar to query.
        Returns a zero vector on cold start (empty store).
        """
        if not self.store:
            return torch.zeros(query.shape[-1])

        q     = query.detach().cpu().float()
        stack = torch.stack(self.store, dim=0)              # (k, embed_dim)

        # Cosine similarity: normalise both sides, then dot-product
        q_norm = q     / (q.norm()                      + 1e-8)
        s_norm = stack / (stack.norm(dim=1, keepdim=True) + 1e-8)
        sims   = s_norm @ q_norm                            # (k,)

        return self.store[sims.argmax().item()]              # (embed_dim,)

    def __len__(self) -> int:
        return len(self.store)


# ══════════════════════════════════════════════════════════════════
# 6.  PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════

class SensorDataset(Dataset):
    """Pairs of (normalised_x_t, normalised_x_{t+1}) for supervised training."""

    def __init__(self, data: np.ndarray):
        self.x = torch.tensor(data[:-1], dtype=torch.float32)   # (n-1, 3)
        self.y = torch.tensor(data[1:],  dtype=torch.float32)   # (n-1, 3)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


dataset = SensorDataset(norm_data)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)


# ══════════════════════════════════════════════════════════════════
# 7.  MODEL INSTANTIATION + OPTIMISER
# ══════════════════════════════════════════════════════════════════

encoder      = SensorEncoder().to(DEVICE)
predictor    = JEPAPredictor().to(DEVICE)
vector_store = VectorStore()

# Joint optimisation of encoder + predictor
optimizer = optim.Adam(
    list(encoder.parameters()) + list(predictor.parameters()),
    lr=LR,
)
criterion = nn.MSELoss()

total_params = (
    sum(p.numel() for p in encoder.parameters()) +
    sum(p.numel() for p in predictor.parameters())
)
print(f"[Model]   Encoder params    : {sum(p.numel() for p in encoder.parameters()):,}")
print(f"[Model]   Predictor params  : {sum(p.numel() for p in predictor.parameters()):,}")
print(f"[Model]   Total params      : {total_params:,}")


# ══════════════════════════════════════════════════════════════════
# 8.  HELPER — encode entire dataset without gradients
# ══════════════════════════════════════════════════════════════════

def encode_all(data: np.ndarray) -> torch.Tensor:
    """
    Run the encoder over the full dataset in one pass.
    Returns a CPU tensor of shape (n, EMBED_DIM).
    Sets encoder back to train mode before returning.
    """
    encoder.eval()
    with torch.no_grad():
        x   = torch.tensor(data, dtype=torch.float32).to(DEVICE)
        emb = encoder(x)
    encoder.train()
    return emb.cpu()


# ══════════════════════════════════════════════════════════════════
# 9.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

print(f"\n[Train]   {NUM_EPOCHS} epochs  ·  batch={BATCH_SIZE}  ·  lr={LR}")
epoch_losses: list = []

for epoch in range(1, NUM_EPOCHS + 1):

    # Re-encode the full dataset with the current encoder weights so the
    # vector store context stays consistent with the evolving latent space.
    all_embs = encode_all(norm_data)                    # (1000, EMBED_DIM)

    # Seed vector store with the first VECTOR_STORE_SIZE embeddings.
    # During training this acts as a fixed-window context approximation.
    vector_store = VectorStore()
    for emb_vec in all_embs[:VECTOR_STORE_SIZE]:
        vector_store.add(emb_vec)

    epoch_loss  = 0.0
    num_batches = 0

    encoder.train()
    predictor.train()

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(DEVICE)               # (B, SENSOR_DIM)
        batch_y = batch_y.to(DEVICE)               # (B, SENSOR_DIM)

        # Encode current timestep and the target (next) timestep
        emb_current = encoder(batch_x)             # (B, EMBED_DIM)
        emb_target  = encoder(batch_y)             # (B, EMBED_DIM)

        # Retrieve context for each sample (uses the epoch-start snapshots)
        retrieved = torch.stack(
            [vector_store.retrieve(emb_current[i]) for i in range(len(emb_current))],
            dim=0,
        ).to(DEVICE)                               # (B, EMBED_DIM)

        # Predict next embedding from (current ⊕ context)
        pred_input = torch.cat([emb_current, retrieved], dim=1)   # (B, 2·EMBED_DIM)
        pred_emb   = predictor(pred_input)                         # (B, EMBED_DIM)

        # JEPA-style loss: compare predicted embedding to actual next embedding.
        # We detach the target to avoid collapsing representations through
        # the target encoder branch.
        loss = criterion(pred_emb, emb_target.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss  += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / max(num_batches, 1)
    epoch_losses.append(avg_loss)

    if epoch == 1 or epoch % 10 == 0:
        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}  │  loss = {avg_loss:.6f}")

print("[Train]   Done.\n")


# ══════════════════════════════════════════════════════════════════
# 10. SAVE MODEL WEIGHTS
# ══════════════════════════════════════════════════════════════════

model_path = os.path.join(OUTPUT_DIR, "jepa_edge_sensor.pt")
torch.save(
    {
        "encoder_state":   encoder.state_dict(),
        "predictor_state": predictor.state_dict(),
        # Save normalisation statistics so the model can be reloaded standalone
        "data_min":        data_min.tolist(),
        "data_max":        data_max.tolist(),
        "embed_dim":       EMBED_DIM,
        "hidden_dim":      HIDDEN_DIM,
        "sensor_dim":      SENSOR_DIM,
    },
    model_path,
)
print(f"[Save]    Model weights  → {model_path}")


# ══════════════════════════════════════════════════════════════════
# 11. SAVE FULL DATASET AS CSV  (timestep + raw values + embeddings)
# ══════════════════════════════════════════════════════════════════

encoder.eval()
with torch.no_grad():
    final_embs = encoder(
        torch.tensor(norm_data, dtype=torch.float32).to(DEVICE)
    ).cpu().numpy()                                # (1000, EMBED_DIM)

csv_dict = {
    "timestep":    np.arange(NUM_TIMESTEPS),
    "temperature": raw_data[:, 0],
    "pressure":    raw_data[:, 1],
    "battery":     raw_data[:, 2],
}
for d in range(EMBED_DIM):
    csv_dict[f"emb_{d}"] = final_embs[:, d]

csv_path = os.path.join(OUTPUT_DIR, "sensor_dataset.csv")
pd.DataFrame(csv_dict).to_csv(csv_path, index=False)
print(f"[Save]    Dataset CSV    → {csv_path}")


# ══════════════════════════════════════════════════════════════════
# 12. STREAMING INFERENCE EVALUATION
#     Simulates step-by-step operation as on an edge device.
#     Measures per-step latency and peak Python heap memory.
# ══════════════════════════════════════════════════════════════════

print("\n[Eval]    Running streaming inference …")

# Pre-compute the pseudo-inverse of the encoder's weight matrix.
# Encoder  :  emb = W @ x_norm + b       (W: EMBED_DIM × SENSOR_DIM)
# Decode   :  x_norm ≈ pinv(W) @ (emb − b)
# This is a least-squares inversion — exact when EMBED_DIM == SENSOR_DIM,
# approximate (but useful for visualisation) when EMBED_DIM > SENSOR_DIM.
W_np   = encoder.fc.weight.data.cpu().numpy()   # (EMBED_DIM, SENSOR_DIM)
b_np   = encoder.fc.bias.data.cpu().numpy()     # (EMBED_DIM,)
W_pinv = np.linalg.pinv(W_np)                   # (SENSOR_DIM, EMBED_DIM)


def decode_embedding(emb: np.ndarray) -> np.ndarray:
    """Map a latent embedding back to normalised sensor space via pinv(W)."""
    return W_pinv @ (emb - b_np)                # (SENSOR_DIM,)


encoder.eval()
predictor.eval()

eval_store    = VectorStore()   # starts empty — cold start
predicted_raw = []              # decoded physical-unit predictions
latencies_ms  = []              # wall-clock time per inference step

tracemalloc.start()             # track Python heap allocations

with torch.no_grad():
    for t in range(NUM_TIMESTEPS - 1):
        t0 = time.perf_counter()

        # --- Encode current step ---
        x_t   = torch.tensor(norm_data[t], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        emb_t = encoder(x_t)                                            # (1, EMBED_DIM)

        # --- Retrieve most similar past embedding as context ---
        ctx  = eval_store.retrieve(emb_t.squeeze(0)).unsqueeze(0).to(DEVICE)

        # --- Predict next embedding ---
        pred = predictor(torch.cat([emb_t, ctx], dim=1))               # (1, EMBED_DIM)

        # --- Decode to sensor space for visualisation ---
        pred_norm = np.clip(
            decode_embedding(pred.squeeze(0).cpu().numpy()), 0.0, 1.0
        )
        predicted_raw.append(denormalize(pred_norm))

        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1e3)

        # Add current embedding to rolling store (do NOT add predicted,
        # only observed — mirrors real edge-device behaviour)
        eval_store.add(emb_t.squeeze(0))

_, peak_mem_bytes = tracemalloc.get_traced_memory()
tracemalloc.stop()

predicted_raw = np.array(predicted_raw)         # (999, 3) — physical units
actual_raw    = raw_data[1:]                    # (999, 3) — ground truth t+1

mean_lat_ms  = float(np.mean(latencies_ms))
peak_mem_mb  = peak_mem_bytes / 1024 ** 2
final_loss   = epoch_losses[-1]

print(f"[Eval]    {NUM_TIMESTEPS - 1} inference steps completed.")


# ══════════════════════════════════════════════════════════════════
# 13. PLOT — Training Loss Curve
# ══════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, NUM_EPOCHS + 1), epoch_losses, color="#2563EB", linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss (latent space)")
ax.set_title("JEPA Edge Sensor — Training Loss (latent MSE)")
ax.grid(True, alpha=0.3)
plt.tight_layout()

loss_png = os.path.join(OUTPUT_DIR, "training_loss.png")
plt.savefig(loss_png, dpi=150)
plt.close()
print(f"[Plot]    Training loss      → {loss_png}")


# ══════════════════════════════════════════════════════════════════
# 14. PLOT — Predicted vs Actual for all 3 signals
# ══════════════════════════════════════════════════════════════════

signal_labels  = ["Temperature (°C)", "Pressure (hPa)", "Battery (%)"]
signal_colours = ["#DC2626", "#16A34A", "#2563EB"]
t_axis         = np.arange(1, NUM_TIMESTEPS)        # timesteps 1 … 999

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

for i, (label, colour) in enumerate(zip(signal_labels, signal_colours)):
    axes[i].plot(t_axis, actual_raw[:, i],
                 lw=1.2, color="gray", alpha=0.8, label="Actual")
    axes[i].plot(t_axis, predicted_raw[:, i],
                 lw=1.2, color=colour, linestyle="--", label="Predicted")
    axes[i].set_ylabel(label, fontsize=9)
    axes[i].legend(fontsize=8, loc="upper right")
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel("Timestep")
fig.suptitle("JEPA Edge Sensor — Predicted vs Actual Signals", fontsize=12)
plt.tight_layout()

pred_png = os.path.join(OUTPUT_DIR, "predicted_vs_actual.png")
plt.savefig(pred_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"[Plot]    Predicted vs actual → {pred_png}")


# ══════════════════════════════════════════════════════════════════
# 15. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════

print()
print("═" * 56)
print("  JEPA EDGE SENSOR EXPERIMENT — SUMMARY")
print("═" * 56)
print(f"  Device                  : {DEVICE}")
print(f"  Total model parameters  : {total_params:,}")
print(f"  Final training loss     : {final_loss:.6f}  (latent MSE)")
print(f"  Mean inference latency  : {mean_lat_ms:.3f} ms / step")
print(f"  Peak memory (Python)    : {peak_mem_mb:.2f} MB")
print(f"  Outputs directory       : {os.path.abspath(OUTPUT_DIR)}/")
print("═" * 56)
print()
print("  Files saved:")
print(f"    {os.path.join(OUTPUT_DIR, 'sensor_dataset.csv')}")
print(f"    {os.path.join(OUTPUT_DIR, 'jepa_edge_sensor.pt')}")
print(f"    {os.path.join(OUTPUT_DIR, 'training_loss.png')}")
print(f"    {os.path.join(OUTPUT_DIR, 'predicted_vs_actual.png')}")
print("═" * 56)
