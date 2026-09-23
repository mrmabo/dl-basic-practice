"""Stage 2-10: tokenize SMS text, train an Embedding + LSTM spam classifier."""

import os
import random
import re
from collections import Counter
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

SEED = 42
BATCH_SIZE = 64
EPOCHS = int(os.getenv("EPOCHS", "10"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sms_spam" / "SMSSpamCollection"
CHECKPOINT = ROOT / "checkpoints" / "10_text_lstm.pt"
PAD, UNK = 0, 1


def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def load_rows():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Run: python download_data/download_10_sms_spam.py"
        )
    rows = []
    with DATA_PATH.open(encoding="utf-8") as file:
        for line in file:
            label, text = line.rstrip("\n").split("\t", 1)
            rows.append((text, 1 if label == "spam" else 0))
    return rows


def build_vocab(texts, min_frequency=2):
    counts = Counter(token for text in texts for token in tokenize(text))
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for token, count in sorted(counts.items()):
        if count >= min_frequency:
            vocab[token] = len(vocab)
    return vocab


class SMSDataset(Dataset):
    def __init__(self, rows, vocab):
        self.rows, self.vocab = rows, vocab

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        text, label = self.rows[index]
        tokens = [self.vocab.get(token, UNK) for token in tokenize(text)] or [UNK]
        return torch.tensor(tokens), torch.tensor(label, dtype=torch.float32)


def collate_batch(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    padded = pad_sequence(sequences, batch_first=True, padding_value=PAD)
    return padded, lengths, torch.stack(labels)


class TextClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=PAD)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.head = nn.Linear(64, 1)

    def forward(self, tokens, lengths):
        embedded = self.embedding(tokens)  # [B, L] -> [B, L, 64]
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.head(hidden[-1]).squeeze(1)


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for tokens, lengths, labels in loader:
            tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
            logits = model(tokens, lengths)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += ((logits >= 0) == labels.bool()).sum().item()
            count += len(labels)
    return loss_sum / count, correct / count


def predict_text(model, vocab, text):
    ids = torch.tensor([[vocab.get(token, UNK) for token in tokenize(text)] or [UNK]])
    lengths = torch.tensor([ids.size(1)])
    with torch.no_grad():
        probability = model(ids.to(DEVICE), lengths).sigmoid().item()
    return probability


def main():
    random.seed(SEED); torch.manual_seed(SEED); CHECKPOINT.parent.mkdir(exist_ok=True)
    rows = load_rows()
    train_rows, temp_rows = train_test_split(rows, test_size=0.3, random_state=SEED,
                                              stratify=[label for _, label in rows])
    val_rows, test_rows = train_test_split(temp_rows, test_size=0.5, random_state=SEED,
                                            stratify=[label for _, label in temp_rows])
    vocab = build_vocab([text for text, _ in train_rows])
    loaders = [DataLoader(SMSDataset(split, vocab), BATCH_SIZE, shuffle=(i == 0),
                          collate_fn=collate_batch)
               for i, split in enumerate([train_rows, val_rows, test_rows])]
    train_loader, val_loader, test_loader = loaders

    model = TextClassifier(len(vocab)).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn)
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "vocab": vocab}, CHECKPOINT)
        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    saved = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(saved["model"])
    test_loss, test_acc = run_epoch(model, test_loader, loss_fn)
    message = "Congratulations! You won a free prize. Call now."
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}")
    print(f"spam_probability={predict_text(model, saved['vocab'], message):.3f}")


if __name__ == "__main__":
    main()
