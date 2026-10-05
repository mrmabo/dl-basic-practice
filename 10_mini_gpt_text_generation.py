"""流程10：从零训练小型 GPT，学习英文文本的 next-token prediction。

做什么：用约 1.1 MB 的 Tiny Shakespeare 英文文本训练字符级语言模型。
怎么做：字符词表 -> 连续文本划分 train/val/test -> 右移一位的标签
-> token/position embedding -> 因果 self-attention + FFN + 残差
-> 每个位置的词表 logits -> CrossEntropyLoss -> 自回归采样。
目标：理解 GPT 如何生成文字，并能从空白文件重写完整流程。

这是 decoder-only Transformer；没有 encoder-decoder cross-attention。
字符就是本练习的 token，真实 LLM 通常使用子词 tokenizer。
因果遮罩禁止看到未来标签；CE 等价于最小化正确下一字符的负对数概率。
小数据从零训练只会模仿莎士比亚文本，不具备可靠的通用聊天/指令能力。
运行：先执行 download_data/download_10_tiny_shakespeare.py，再运行本文件。
--generate-only 加载已有权重；--interactive 进入提示词续写循环。
"""

import argparse
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "tiny_shakespeare" / "input.txt"
CHECKPOINT = ROOT / "checkpoints" / "10_mini_gpt.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BLOCK_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
DROPOUT = 0.1


class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size
        if len(tokens) <= block_size:
            raise ValueError("Each split needs more tokens than block_size.")

    def __len__(self):
        # Non-overlapping input blocks keep an epoch small; labels shift by one.
        return (len(self.tokens) - 1) // self.block_size

    def __getitem__(self, index):
        start = index * self.block_size
        x = self.tokens[start : start + self.block_size]
        y = self.tokens[start + 1 : start + self.block_size + 1]
        return x, y  # both [T], long; DataLoader produces [B, T]


def build_dataloaders(text, chars, block_size, batch_size):
    stoi = {char: index for index, char in enumerate(chars)}
    # Vocabulary is built from training text only. Unknown held-out characters
    # map to the training space character, rather than leaking vocabulary.
    tokens = torch.tensor([stoi.get(c, stoi[" "]) for c in text], dtype=torch.long)
    train_end, val_end = int(len(tokens) * 0.9), int(len(tokens) * 0.95)
    splits = [tokens[:train_end], tokens[train_end:val_end], tokens[val_end:]]
    return tuple(
        DataLoader(TextDataset(s, block_size), batch_size=batch_size, shuffle=i == 0)
        for i, s in enumerate(splits)
    )


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
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    return total_loss / total_tokens


@torch.no_grad()
def generate(model, chars, prompt, max_new_tokens=300, temperature=0.8, top_k=20):
    if temperature <= 0 or max_new_tokens < 0 or top_k < 1:
        raise ValueError("temperature/top_k must be positive; length nonnegative")
    model.eval()
    stoi = {c: i for i, c in enumerate(chars)}
    unknown = sorted(set(prompt) - set(chars))
    if unknown:
        raise ValueError(f"Prompt contains characters outside vocabulary: {unknown}")
    tokens = torch.tensor([[stoi[c] for c in (prompt or "\n")]], device=DEVICE)
    for _ in range(max_new_tokens):
        logits = model(tokens[:, -model.block_size :])[:, -1, :] / temperature
        cutoff = logits.topk(min(top_k, len(chars))).values[:, -1:]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
        next_token = torch.multinomial(logits.softmax(dim=-1), 1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return "".join(chars[i] for i in tokens[0].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(SEED)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if not args.generate_only:
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                "Run python download_data/download_10_tiny_shakespeare.py"
            )
        text = DATA_PATH.read_text(encoding="utf-8")
        chars = sorted(set(text[: int(len(text) * 0.9)]))
        config = dict(
            vocab_size=len(chars),
            block_size=BLOCK_SIZE,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            dropout=DROPOUT,
        )
        train_loader, val_loader, test_loader = build_dataloaders(
            text, chars, BLOCK_SIZE, args.batch_size
        )
        model = MiniGPT(**config).to(DEVICE)
        print(
            f"device={DEVICE} vocab={len(chars)} parameters={sum(p.numel() for p in model.parameters()):,}"
        )
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        best_val_loss = float("inf")
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, loss_fn, optimizer)
            val_loss = run_epoch(model, val_loader, loss_fn)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"model": model.state_dict(), "chars": chars, "config": config},
                    CHECKPOINT,
                )
            print(
                f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_perplexity={math.exp(min(val_loss, 20)):.2f}"
            )
    if not CHECKPOINT.exists():
        raise FileNotFoundError("Train the model before using --generate-only")
    saved = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    chars = saved["chars"]
    model = MiniGPT(**saved["config"]).to(DEVICE)
    model.load_state_dict(saved["model"])
    if not args.generate_only:
        test_loss = run_epoch(model, test_loader, loss_fn)
        print(
            f"test_loss={test_loss:.4f} test_perplexity={math.exp(min(test_loss, 20)):.2f}"
        )
    print(generate(model, chars, args.prompt, args.max_new_tokens, args.temperature))
    if args.interactive:
        print(
            "Enter an English prompt for continuation; /quit exits. No conversation memory."
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
