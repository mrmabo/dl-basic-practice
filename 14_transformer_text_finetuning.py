"""Flow 14: fine-tune DistilBERT for SMS spam classification."""

import os
import random
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = int(os.getenv("EPOCHS", "3"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "transformer_sms_spam" / "SMSSpamCollection"
CHECKPOINT = ROOT / "checkpoints" / "14_distilbert_sms.pt"


def load_rows():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Run: python download_data/download_14_transformer_sms.py"
        )
    rows = []
    with DATA_PATH.open(encoding="utf-8") as file:
        for line in file:
            label, text = line.rstrip("\n").split("\t", 1)
            rows.append((text, 1 if label == "spam" else 0))
    return rows


class TokenizedTextDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        text, label = self.rows[index]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        encoded["labels"] = label
        return encoded


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        labels = torch.tensor([item.pop("labels") for item in batch], dtype=torch.long)
        padded = tokenizer.pad(batch, padding=True, return_tensors="pt")
        padded["labels"] = labels
        return padded

    return collate_fn


def build_dataloaders(tokenizer):
    rows = load_rows()
    labels = [label for _, label in rows]
    train_rows, temporary_rows = train_test_split(
        rows, test_size=0.3, random_state=SEED, stratify=labels
    )
    temporary_labels = [label for _, label in temporary_rows]
    val_rows, test_rows = train_test_split(
        temporary_rows,
        test_size=0.5,
        random_state=SEED,
        stratify=temporary_labels,
    )
    collate_fn = make_collate_fn(tokenizer)
    loaders = [
        DataLoader(
            TokenizedTextDataset(split, tokenizer),
            batch_size=BATCH_SIZE,
            shuffle=index == 0,
            collate_fn=collate_fn,
        )
        for index, split in enumerate([train_rows, val_rows, test_rows])
    ]
    spam_count = sum(label for _, label in train_rows)
    class_weights = torch.tensor(
        [1.0, (len(train_rows) - spam_count) / spam_count], device=DEVICE
    )
    return *loaders, class_weights


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            labels = batch.pop("labels").to(DEVICE)
            inputs = {key: value.to(DEVICE) for key, value in batch.items()}
            logits = model(**inputs).logits
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
    return total_loss / count, correct / count


def predict_text(model, tokenizer, text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    model.eval()
    with torch.no_grad():
        probabilities = model(**inputs).logits.softmax(dim=1)[0]
    return probabilities.cpu()


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    CHECKPOINT.parent.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    )
    test_loss, test_acc = run_epoch(model, test_loader, loss_fn)
    text = "Congratulations! You won a free prize. Call now."
    probabilities = predict_text(model, tokenizer, text)
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}")
    print(
        f"ham_probability={probabilities[0]:.3f} spam_probability={probabilities[1]:.3f}"
    )


if __name__ == "__main__":
    main()
