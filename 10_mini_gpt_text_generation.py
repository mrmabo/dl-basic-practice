"""流程10：用DailyDialog短英文对话从零训练Mini GPT。

目标：手写 decoder-only Transformer 的完整问答训练闭环。
数据：公开DailyDialog（ConvLab镜像），默认5000组短问答，官方split独立保存。
输入：[BOS] + 问题字符 + [SEP] + 回答字符；标签右移一位。
只对回答和EOS计算CrossEntropyLoss，问题和padding标签为-100。
token/position embedding -> 因果多头Q/K/V注意力 -> 残差FFN
-> [B,T,V] logits -> loss -> backward -> optimizer -> 最佳checkpoint。
推理：输入[BOS]+问题+[SEP]，逐字符采样，遇EOS停止，只显示回答。
字符级教学模型约65万参数，不能保证通用问答质量；每轮聊天独立。
先运行 download_data/download_10_dailydialog.py；再训练；
--generate-only --interactive 加载DailyDialog权重聊天。
"""

import argparse
import math
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "dailydialog"
CHECKPOINT = ROOT / "checkpoints" / "10_mini_gpt_dailydialog.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BLOCK_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 10
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
DROPOUT = 0.1
PAD, UNK, BOS, SEP, EOS = range(5)
IGNORE_INDEX = -100
MAX_PROMPT_CHARS = 96


def encode(text, chars):
    stoi = {c: i + 5 for i, c in enumerate(chars)}
    return [stoi.get(c, UNK) for c in text]


class DialogueDataset(Dataset):
    def __init__(self, rows, chars, block_size):
        self.examples = []
        for row in rows:
            question = encode(row['question'], chars)[:MAX_PROMPT_CHARS]
            # Keep at least an answer token and EOS inside the context.
            answer = encode(row['answer'], chars)[:block_size - len(question) - 2]
            sequence = [BOS] + question + [SEP] + answer + [EOS]
            inputs, targets = sequence[:-1], sequence[1:]
            # SEP at input index len(question)+1 predicts first answer char.
            targets[:len(question) + 1] = [IGNORE_INDEX] * (len(question) + 1)
            pad = block_size - len(inputs)
            self.examples.append((torch.tensor(inputs + [PAD] * pad),
                                  torch.tensor(targets + [IGNORE_INDEX] * pad)))
        if not self.examples:
            raise ValueError('Empty split: run the DailyDialog download script.')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]  # [T] long -> DataLoader [B,T]


def build_dataloaders(splits, chars, block_size, batch_size):
    return tuple(DataLoader(DialogueDataset(splits[name], chars, block_size),
                           batch_size=batch_size, shuffle=name == 'train')
                 for name in ('train', 'validation', 'test'))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.projection = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask", torch.ones(block_size, block_size, dtype=torch.bool).tril()
        )

    def forward(self, x):
        b, t, d = x.shape  # [B, T, D]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [
            z.reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)
            for z in (q, k, v)
        ]  # [B, H, T, D/H]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:t, :t], float("-inf"))
        weights = self.attention_dropout(scores.softmax(dim=-1))
        hidden = (weights @ v).transpose(1, 2).contiguous().reshape(b, t, d)
        return self.output_dropout(self.projection(hidden))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, block_size, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))  # pre-norm residual, [B, T, D]
        return x + self.ffn(self.norm2(x))


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=BLOCK_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(d_model, n_heads, block_size, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        return self.head(self.norm(self.blocks(x)))  # [B, T, V], raw logits


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens = 0.0, 0
    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            logits = model(inputs)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            valid_tokens = (targets != IGNORE_INDEX).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
    return total_loss / total_tokens


@torch.no_grad()
def generate(model, chars, prompt, max_new_tokens=160, temperature=0.8, top_k=20):
    if temperature <= 0 or max_new_tokens < 1 or top_k < 1:
        raise ValueError('temperature, length and top_k must be positive')
    if not prompt.strip():
        raise ValueError('Please enter an English question.')
    model.eval()
    question = encode(' '.join(prompt.split()), chars)[:MAX_PROMPT_CHARS]
    tokens = torch.tensor([[BOS] + question + [SEP]], device=DEVICE)
    answer = []
    # Keep the complete prompt in context; leave one position per next token.
    for _ in range(min(max_new_tokens, model.block_size - tokens.size(1) + 1)):
        logits = model(tokens)[:, -1, :] / temperature
        logits[:, [PAD, UNK, BOS, SEP]] = float('-inf')
        cutoff = logits.topk(min(top_k, len(chars) + 1)).values[:, -1:]
        logits = logits.masked_fill(logits < cutoff, float('-inf'))
        next_token = torch.multinomial(logits.softmax(dim=-1), 1)
        index = next_token.item()
        if index == EOS:
            break
        answer.append(chars[index - 5])
        tokens = torch.cat([tokens, next_token], dim=1)
    return ''.join(answer).strip() or '(No response; try more training or another question.)'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(SEED)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if not args.generate_only:
        if not (DATA_PATH / "train.json").exists():
            raise FileNotFoundError(
                "Run python download_data/download_10_dailydialog.py"
            )
        splits = {name: json.loads((DATA_PATH / f"{name}.json").read_text(encoding="utf-8"))
                  for name in ("train", "validation", "test")}
        chars = sorted(set("".join(row["question"] + row["answer"]
                                     for row in splits["train"])))
        config = dict(
            vocab_size=len(chars) + 5,
            block_size=BLOCK_SIZE,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            dropout=DROPOUT,
        )
        train_loader, val_loader, test_loader = build_dataloaders(
            splits, chars, BLOCK_SIZE, args.batch_size
        )
        model = MiniGPT(**config).to(DEVICE)
        print(
            f"device={DEVICE} vocab={len(chars)} parameters={sum(p.numel() for p in model.parameters()):,}"
        )
        loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        best_val_loss = float("inf")
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, loss_fn, optimizer)
            val_loss = run_epoch(model, val_loader, loss_fn)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"model": model.state_dict(), "chars": chars, "config": config,
                     "format": "dailydialog-char-v1"},
                    CHECKPOINT,
                )
            print(
                f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_perplexity={math.exp(min(val_loss, 20)):.2f}"
            )
    if not CHECKPOINT.exists():
        raise FileNotFoundError("Train on DailyDialog first: python 10_mini_gpt_text_generation.py")
    saved = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    if saved.get("format") != "dailydialog-char-v1":
        raise ValueError("Incompatible checkpoint. Retrain on DailyDialog.")
    chars = saved["chars"]
    model = MiniGPT(**saved["config"]).to(DEVICE)
    model.load_state_dict(saved["model"])
    if not args.generate_only:
        test_loss = run_epoch(model, test_loader, loss_fn)
        print(
            f"test_loss={test_loss:.4f} test_perplexity={math.exp(min(test_loss, 20)):.2f}"
        )
    print("You:", args.prompt)
    print("Mini GPT:", generate(model, chars, args.prompt, args.max_new_tokens, args.temperature))
    if args.interactive:
        print(
            "Ask a short English question; /quit exits. Each question is independent."
        )
        while True:
            try:
                prompt = input("You: ")
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.strip() == "/quit":
                break
            try:
                print(
                    "Mini GPT:",
                    generate(
                        model, chars, prompt, args.max_new_tokens, args.temperature
                    ),
                )
            except ValueError as error:
                print(error)


if __name__ == "__main__":
    main()
