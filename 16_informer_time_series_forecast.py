"""Stage 2-16: compact Informer with ProbSparse encoding and one-pass decoding."""

import csv
import math
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# config
SEED = 42
LOOKBACK = 96
LABEL_LEN = 48
HORIZON = 24
BATCH_SIZE = 16
EPOCHS = int(os.getenv("EPOCHS", "10"))
D_MODEL = 64
N_HEADS = 4
FACTOR = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_NAMES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
TARGET_NAMES = ["HUFL", "OT"]
TARGET_INDICES = [FEATURE_NAMES.index(name) for name in TARGET_NAMES]
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "transformer_etth1" / "ETTh1.csv"
CHECKPOINT = ROOT / "checkpoints" / "17_informer.pt"


# dataloader
def load_etth1():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Run: python download_data/download_16_informer_etth1.py"
        )
    rows, time_features, timestamps = [], [], []
    with DATA_PATH.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            stamp = datetime.fromisoformat(row["date"])
            timestamps.append(row["date"])
            rows.append([float(row[name]) for name in FEATURE_NAMES])
            cycles = [
                (stamp.month - 1, 12),
                (stamp.day - 1, 31),
                (stamp.weekday(), 7),
                (stamp.hour, 24),
            ]
            time_features.append(
                [
                    value
                    for position, period in cycles
                    for value in (
                        math.sin(2 * math.pi * position / period),
                        math.cos(2 * math.pi * position / period),
                    )
                ]
            )
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(time_features, dtype=np.float32),
        timestamps,
    )


class WindowDataset(Dataset):
    def __init__(self, values, time_features):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.time_features = torch.as_tensor(time_features, dtype=torch.float32)
        if len(values) < LOOKBACK + HORIZON:
            raise ValueError("Not enough rows for one forecast window.")
        if not 0 < LABEL_LEN <= LOOKBACK:
            raise ValueError("LABEL_LEN must be in (0, LOOKBACK].")

    def __len__(self):
        return len(self.values) - LOOKBACK - HORIZON + 1

    def __getitem__(self, index):
        start = index + LOOKBACK
        end = start + HORIZON
        encoder_values = self.values[index:start]  # [96,7]
        encoder_time = self.time_features[index:start]  # [96,8]
        decoder_history = self.values[start - LABEL_LEN : start]  # [48,7]
        future_zeros = torch.zeros(HORIZON, len(FEATURE_NAMES))  # [24,7]
        decoder_values = torch.cat([decoder_history, future_zeros], dim=0)  # [72,7]
        decoder_time = self.time_features[start - LABEL_LEN : end]  # [72,8]
        targets = self.values[start:end, TARGET_INDICES]  # [24,2]
        return encoder_values, encoder_time, decoder_values, decoder_time, targets


def build_dataloaders():
    raw, time_features, timestamps = load_etth1()
    train_end = int(len(raw) * 0.70)
    val_end = int(len(raw) * 0.85)
    mean = raw[:train_end].mean(axis=0)
    std = raw[:train_end].std(axis=0)
    std[std == 0] = 1.0
    values = (raw - mean) / std
    train_set = WindowDataset(values[:train_end], time_features[:train_end])
    val_set = WindowDataset(
        values[train_end - LOOKBACK : val_end],
        time_features[train_end - LOOKBACK : val_end],
    )
    test_set = WindowDataset(
        values[val_end - LOOKBACK :], time_features[val_end - LOOKBACK :]
    )
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, BATCH_SIZE)
    test_loader = DataLoader(test_set, BATCH_SIZE)
    return train_loader, val_loader, test_loader, mean, std, timestamps[val_end:]


# model
class DataEmbedding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        self.value = nn.Linear(len(FEATURE_NAMES), d_model)
        self.temporal = nn.Linear(8, d_model, bias=False)
        position = torch.arange(max_len).float().unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("position", encoding.unsqueeze(0))
        self.dropout = nn.Dropout(0.1)

    def forward(self, values, time_features):
        value = self.value(values)  # [B,L,64]
        temporal = self.temporal(time_features)  # [B,L,64]
        position = self.position[:, : values.size(1)]  # [1,L,64]
        hidden = value + temporal + position  # [B,L,64]
        hidden = self.dropout(hidden)  # [B,L,64]
        return hidden


class ProbSparseAttention(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, factor=FACTOR):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.factor = factor
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, features):
        batch, length, width = features.shape
        queries = self.q(features)  # [B,L,D]
        keys = self.k(features)  # [B,L,D]
        values = self.v(features)  # [B,L,D]
        queries = queries.reshape(batch, length, self.n_heads, self.head_dim)
        queries = queries.transpose(1, 2)  # [B,H,L,d]
        keys = keys.reshape(batch, length, self.n_heads, self.head_dim)
        keys = keys.transpose(1, 2)  # [B,H,L,d]
        values = values.reshape(batch, length, self.n_heads, self.head_dim)
        values = values.transpose(1, 2)  # [B,H,L,d]
        sample_count = min(
            length, max(1, self.factor * math.ceil(math.log(length + 1)))
        )
        top_count = sample_count
        if self.training:
            sampled_indices = torch.randint(
                length, (length, sample_count), device=features.device
            )
        else:
            generator = torch.Generator(device=features.device).manual_seed(SEED)
            sampled_indices = torch.randint(
                length,
                (length, sample_count),
                device=features.device,
                generator=generator,
            )
        sampled_keys = keys[:, :, sampled_indices, :]  # [B,H,L,S,d]
        sampled_scores = torch.einsum(
            "bhld,bhlsd->bhls", queries, sampled_keys
        )  # [B,H,L,S]
        sparsity = (
            sampled_scores.max(dim=-1).values - sampled_scores.sum(dim=-1) / length
        )
        top_indices = sparsity.topk(top_count, dim=-1, sorted=False).indices  # [B,H,U]
        query_indices = top_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        selected_queries = queries.gather(2, query_indices)  # [B,H,U,d]
        transposed_keys = keys.transpose(-2, -1)  # [B,H,d,L]
        scores = selected_queries @ transposed_keys  # [B,H,U,L]
        scores = scores / math.sqrt(self.head_dim)  # [B,H,U,L]
        weights = scores.softmax(dim=-1)  # [B,H,U,L]
        weights = self.dropout(weights)  # [B,H,U,L]
        selected_context = weights @ values  # [B,H,U,d]
        mean_context = values.mean(dim=2, keepdim=True)  # [B,H,1,d]
        context = mean_context.expand(-1, -1, length, -1).clone()  # [B,H,L,d]
        context = context.scatter(2, query_indices, selected_context)  # [B,H,L,d]
        context = context.transpose(1, 2).contiguous()  # [B,L,H,d]
        context = context.reshape(batch, length, width)  # [B,L,D]
        output = self.output(context)  # [B,L,D]
        return output


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = ProbSparseAttention()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.feedforward = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * D_MODEL, D_MODEL),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, features):
        attended = self.attention(features)  # [B,L,64]
        attended = self.dropout(attended)  # [B,L,64]
        hidden = self.norm1(features + attended)  # [B,L,64]
        projected = self.feedforward(hidden)  # [B,L,64]
        projected = self.dropout(projected)  # [B,L,64]
        output = self.norm2(hidden + projected)  # [B,L,64]
        return output


class DistillingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(D_MODEL, D_MODEL, 3, padding=1, padding_mode="circular")
        self.norm = nn.BatchNorm1d(D_MODEL)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)

    def forward(self, features):
        hidden = features.transpose(1, 2)  # [B,64,L]
        hidden = self.conv(hidden)  # [B,64,L]
        hidden = self.norm(hidden)  # [B,64,L]
        hidden = self.activation(hidden)  # [B,64,L]
        hidden = self.pool(hidden)  # [B,64,ceil(L/2)]
        output = hidden.transpose(1, 2)  # [B,ceil(L/2),64]
        return output


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            D_MODEL, N_HEADS, dropout=0.1, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            D_MODEL, N_HEADS, dropout=0.1, batch_first=True
        )
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.norm3 = nn.LayerNorm(D_MODEL)
        self.feedforward = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * D_MODEL, D_MODEL),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, features, memory):
        length = features.size(1)
        causal_mask = torch.ones(
            length, length, device=features.device, dtype=torch.bool
        ).triu(1)
        attended, _ = self.self_attention(
            features, features, features, attn_mask=causal_mask, need_weights=False
        )  # [B,72,64]
        attended = self.dropout(attended)  # [B,72,64]
        hidden = self.norm1(features + attended)  # [B,72,64]
        crossed, _ = self.cross_attention(
            hidden, memory, memory, need_weights=False
        )  # [B,72,64]
        crossed = self.dropout(crossed)  # [B,72,64]
        hidden = self.norm2(hidden + crossed)  # [B,72,64]
        projected = self.feedforward(hidden)  # [B,72,64]
        projected = self.dropout(projected)  # [B,72,64]
        output = self.norm3(hidden + projected)  # [B,72,64]
        return output


class Informer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_embedding = DataEmbedding(D_MODEL)
        self.decoder_embedding = DataEmbedding(D_MODEL)
        self.encoder1 = EncoderLayer()
        self.distilling = DistillingLayer()
        self.encoder2 = EncoderLayer()
        self.encoder_norm = nn.LayerNorm(D_MODEL)
        self.decoder = DecoderLayer()
        self.head = nn.Linear(D_MODEL, len(TARGET_NAMES))

    def forward(self, encoder_values, encoder_time, decoder_values, decoder_time):
        hidden = self.encoder_embedding(encoder_values, encoder_time)  # [B,96,64]
        hidden = self.encoder1(hidden)  # [B,96,64]
        hidden = self.distilling(hidden)  # [B,48,64]
        hidden = self.encoder2(hidden)  # [B,48,64]
        memory = self.encoder_norm(hidden)  # [B,48,64]
        hidden = self.decoder_embedding(decoder_values, decoder_time)  # [B,72,64]
        hidden = self.decoder(hidden, memory)  # [B,72,64]
        predictions = self.head(hidden)  # [B,72,2]
        predictions = predictions[:, -HORIZON:]  # [B,24,2]
        return predictions


# training and evaluation
def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_mse = total_mae = count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, batch in enumerate(loader, 1):
            encoder_values, encoder_time, decoder_values, decoder_time, targets = [
                item.to(DEVICE) for item in batch
            ]
            predictions = model(
                encoder_values, encoder_time, decoder_values, decoder_time
            )
            loss = loss_fn(predictions, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            size = targets.size(0)
            total_mse += loss.item() * size
            total_mae += (predictions - targets).abs().mean().item() * size
            count += size
            if training and (step == 1 or step % 100 == 0):
                print(
                    f"  step={step}/{len(loader)} mse={total_mse / count:.5f}",
                    flush=True,
                )
    return total_mse / count, total_mae / count


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train_loader, val_loader, test_loader, mean, std, test_times = build_dataloaders()
    model = Informer().to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    CHECKPOINT.parent.mkdir(exist_ok=True)
    best = float("inf")
    print(
        f"device={DEVICE} train={len(train_loader.dataset)} val={len(val_loader.dataset)}",
        flush=True,
    )
    for epoch in range(1, EPOCHS + 1):
        train_mse, train_mae = run_epoch(model, train_loader, loss_fn, optimizer)
        val_mse, val_mae = run_epoch(model, val_loader, loss_fn)
        if val_mse < best:
            best = val_mse
            torch.save(
                {
                    "model": model.state_dict(),
                    "mean": torch.tensor(mean),
                    "std": torch.tensor(std),
                    "lookback": LOOKBACK,
                    "label_len": LABEL_LEN,
                    "horizon": HORIZON,
                    "feature_names": FEATURE_NAMES,
                    "target_names": TARGET_NAMES,
                },
                CHECKPOINT,
            )
        print(
            f"epoch={epoch:02d} train_mse={train_mse:.5f} train_mae={train_mae:.5f} val_mse={val_mse:.5f} val_mae={val_mae:.5f}",
            flush=True,
        )
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test_mse, test_mae = run_epoch(model, test_loader, loss_fn)
    print(f"test_mse(normalized)={test_mse:.5f} test_mae(normalized)={test_mae:.5f}")
    model.eval()
    batch = next(iter(test_loader))
    inputs = [item[:1].to(DEVICE) for item in batch[:4]]
    with torch.no_grad():
        predictions = model(*inputs).cpu().numpy()  # [1,24,2]
    mean = checkpoint["mean"].cpu().numpy()[TARGET_INDICES]
    std = checkpoint["std"].cpu().numpy()[TARGET_INDICES]
    predictions = predictions * std + mean
    actual = batch[-1][:1].numpy() * std + mean
    for step, timestamp in enumerate(test_times[:HORIZON]):
        values = " ".join(
            f"pred_{name}={predictions[0, step, i]:.3f} true_{name}={actual[0, step, i]:.3f}"
            for i, name in enumerate(TARGET_NAMES)
        )
        print(f"time={timestamp} {values}")


if __name__ == "__main__":
    main()
