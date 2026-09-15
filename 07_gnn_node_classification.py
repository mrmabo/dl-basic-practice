"""完整流程 7：使用 Cora 公开数据训练两层 GCN 节点分类模型。"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

SEED = 42
EPOCHS = int(os.getenv("EPOCHS", "200"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "gnn_cora" / "cora"
CONTENT_PATH = DATA_DIR / "cora.content"
CITES_PATH = DATA_DIR / "cora.cites"
CHECKPOINT_PATH = ROOT / "best_gcn.pt"


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def stratified_masks(labels, train_per_class=20, val_per_class=30):
    rng = np.random.default_rng(SEED)
    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    val_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask = torch.zeros(len(labels), dtype=torch.bool)

    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        rng.shuffle(indices)
        train_mask[indices[:train_per_class]] = True
        val_mask[indices[train_per_class : train_per_class + val_per_class]] = True
        test_mask[indices[train_per_class + val_per_class :]] = True
    return train_mask, val_mask, test_mask


def load_cora():
    if not CONTENT_PATH.exists() or not CITES_PATH.exists():
        raise FileNotFoundError(
            f"没有找到 {CONTENT_PATH} 或 {CITES_PATH}\n"
            "请先运行：python download_data/download_07_gnn_cora.py"
        )

    node_ids, features, text_labels = [], [], []
    with CONTENT_PATH.open(encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            node_ids.append(parts[0])
            features.append([float(value) for value in parts[1:-1]])
            text_labels.append(parts[-1])

    node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    class_names = sorted(set(text_labels))
    class_to_index = {name: index for index, name in enumerate(class_names)}
    labels = np.array([class_to_index[name] for name in text_labels], dtype=np.int64)
    features = np.asarray(features, dtype=np.float32)

    # 对每个节点的词袋特征进行行归一化。
    row_sums = features.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    features = features / row_sums

    edges = []
    with CITES_PATH.open(encoding="utf-8") as file:
        for line in file:
            source_id, target_id = line.split()
            if source_id in node_to_index and target_id in node_to_index:
                source = node_to_index[source_id]
                target = node_to_index[target_id]
                edges.extend([(source, target), (target, source)])

    # 添加 self-loop，然后计算 D^(-1/2) A D^(-1/2)。
    num_nodes = len(node_ids)
    edges.extend((index, index) for index in range(num_nodes))
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    values = torch.ones(edge_index.size(1), dtype=torch.float32)
    adjacency = torch.sparse_coo_tensor(
        edge_index, values, size=(num_nodes, num_nodes)
    ).coalesce()
    row, column = adjacency.indices()
    degree = torch.zeros(num_nodes).scatter_add_(0, row, adjacency.values())
    normalized_values = (
        degree[row].pow(-0.5) * adjacency.values() * degree[column].pow(-0.5)
    )
    normalized_adjacency = torch.sparse_coo_tensor(
        adjacency.indices(), normalized_values, adjacency.size()
    ).coalesce()

    train_mask, val_mask, test_mask = stratified_masks(labels)
    return (
        torch.tensor(features, dtype=torch.float32),
        normalized_adjacency,
        torch.tensor(labels, dtype=torch.long),
        train_mask,
        val_mask,
        test_mask,
        class_names,
    )


class GraphConvolution(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.linear = nn.Linear(input_features, output_features, bias=False)

    def forward(self, features, adjacency):
        support = self.linear(features)
        return torch.sparse.mm(adjacency, support)


class GCN(nn.Module):
    def __init__(self, input_features, num_classes):
        super().__init__()
        self.gcn1 = GraphConvolution(input_features, 32)
        self.gcn2 = GraphConvolution(32, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, adjacency):
        hidden = torch.relu(self.gcn1(features, adjacency))
        hidden = self.dropout(hidden)
        return self.gcn2(hidden, adjacency)  # [N, 1433] -> [N, 7]


def evaluate(model, features, adjacency, labels, mask, loss_fn):
    model.eval()
    with torch.no_grad():
        logits = model(features, adjacency)
        loss = loss_fn(logits[mask], labels[mask])
        accuracy = (logits[mask].argmax(dim=1) == labels[mask]).float().mean()
    return loss.item(), accuracy.item()


def main():
    set_seed()
    (
        features,
        adjacency,
        labels,
        train_mask,
        val_mask,
        test_mask,
        class_names,
    ) = load_cora()
    features = features.to(DEVICE)
    adjacency = adjacency.to(DEVICE)
    labels = labels.to(DEVICE)
    train_mask = train_mask.to(DEVICE)
    val_mask = val_mask.to(DEVICE)
    test_mask = test_mask.to(DEVICE)

    model = GCN(features.size(1), len(class_names)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adjacency)
        loss = loss_fn(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        val_loss, val_acc = evaluate(
            model, features, adjacency, labels, val_mask, loss_fn
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        if epoch == 1 or epoch % 20 == 0:
            print(
                f"epoch={epoch:03d} train_loss={loss.item():.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    )
    test_loss, test_acc = evaluate(
        model, features, adjacency, labels, test_mask, loss_fn
    )
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}")

    model.eval()
    with torch.no_grad():
        predictions = model(features, adjacency).argmax(dim=1)
    node_indices = torch.where(test_mask)[0][:10]
    predicted_names = [class_names[index] for index in predictions[node_indices].cpu()]
    true_names = [class_names[index] for index in labels[node_indices].cpu()]
    print("node_indices:", node_indices.cpu().tolist())
    print("pred:", predicted_names)
    print("true:", true_names)


if __name__ == "__main__":
    main()
