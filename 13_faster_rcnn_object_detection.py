"""Flow 13: fine-tune Faster R-CNN for pedestrian object detection."""

import os
import random
from pathlib import Path
from typing import cast

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

SEED = 42
BATCH_SIZE = 2
EPOCHS = int(os.getenv("EPOCHS", "5"))
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "penn_fudan_ped"
CHECKPOINT = ROOT / "checkpoints" / "13_faster_rcnn.pt"
PREDICTION_IMAGE = ROOT / "checkpoints" / "13_detection_prediction.png"


class PennFudanDataset(Dataset):
    def __init__(self, augment=False):
        self.image_dir = DATA_DIR / "PennFudanPed" / "PNGImages"
        self.mask_dir = DATA_DIR / "PennFudanPed" / "PedMasks"
        if not self.image_dir.exists():
            raise FileNotFoundError(
                "Run: python download_data/download_13_penn_fudan.py"
            )
        self.images = sorted(self.image_dir.glob("*.png"))
        self.masks = sorted(self.mask_dir.glob("*.png"))
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = (
            pil_to_tensor(Image.open(self.images[index]).convert("RGB")).float() / 255
        )
        instance_mask = pil_to_tensor(Image.open(self.masks[index])).squeeze(0)
        object_ids = torch.unique(instance_mask)[1:]  # remove background id 0
        masks = instance_mask == object_ids[:, None, None]

        boxes = []
        for mask in masks:
            y, x = torch.where(mask)
            boxes.append([x.min(), y.min(), x.max(), y.max()])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        if self.augment and random.random() < 0.5:
            image = image.flip(-1)
            masks = masks.flip(-1)
            width = image.shape[-1]
            old_x1, old_x2 = boxes[:, 0].clone(), boxes[:, 2].clone()
            boxes[:, 0], boxes[:, 2] = width - old_x2, width - old_x1

        target = {
            "boxes": boxes,
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "masks": masks.to(torch.uint8),
            "image_id": torch.tensor(index),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_dataloaders():
    base_dataset = PennFudanDataset()
    indices = torch.randperm(
        len(base_dataset), generator=torch.Generator().manual_seed(SEED)
    )
    train_indices, val_indices, test_indices = (
        indices[:120].tolist(),
        indices[120:150].tolist(),
        indices[150:].tolist(),
    )
    train_set = Subset(PennFudanDataset(augment=True), train_indices)
    val_set = Subset(PennFudanDataset(), val_indices)
    test_set = Subset(PennFudanDataset(), test_indices)
    return (
        DataLoader(train_set, BATCH_SIZE, shuffle=True, collate_fn=collate_fn),
        DataLoader(val_set, 1, collate_fn=collate_fn),
        DataLoader(test_set, 1, collate_fn=collate_fn),
    )


def build_model():
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    current_predictor = cast(FastRCNNPredictor, model.roi_heads.box_predictor)
    input_features = current_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, num_classes=2)
    return model


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images = [image.to(DEVICE) for image in images]
        targets = [
            {key: value.to(DEVICE) for key, value in target.items()}
            for target in targets
        ]
        loss_dict = model(images, targets)
        loss = torch.stack(list(loss_dict.values())).sum()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    true_positive = false_positive = false_negative = 0
    with torch.no_grad():
        for images, targets in loader:
            predictions = model([image.to(DEVICE) for image in images])
            for prediction, target in zip(predictions, targets):
                keep = prediction["scores"].cpu() >= SCORE_THRESHOLD
                predicted_boxes = prediction["boxes"].cpu()[keep]
                true_boxes = target["boxes"]
                matched = set()
                for predicted_box in predicted_boxes:
                    if len(true_boxes) == 0:
                        false_positive += 1
                        continue
                    ious = box_iou(predicted_box.unsqueeze(0), true_boxes).squeeze(0)
                    best_index = int(ious.argmax())
                    if ious[best_index] >= IOU_THRESHOLD and best_index not in matched:
                        true_positive += 1
                        matched.add(best_index)
                    else:
                        false_positive += 1
                false_negative += len(true_boxes) - len(matched)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f1


def save_prediction(model, loader):
    images, _ = next(iter(loader))
    model.eval()
    with torch.no_grad():
        prediction = model([images[0].to(DEVICE)])[0]
    output = to_pil_image(images[0])
    draw = ImageDraw.Draw(output)
    for box, score in zip(prediction["boxes"].cpu(), prediction["scores"].cpu()):
        if score < SCORE_THRESHOLD:
            continue
        draw.rectangle(box.tolist(), outline="red", width=3)
        draw.text((box[0].item(), box[1].item()), f"{score:.2f}", fill="red")
    output.save(PREDICTION_IMAGE)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    CHECKPOINT.parent.mkdir(exist_ok=True)
    train_loader, val_loader, test_loader = build_dataloaders()
    model = build_model().to(DEVICE)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.005,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    best_f1 = -1.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        precision, recall, f1 = evaluate(model, val_loader)
        scheduler.step()
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), CHECKPOINT)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_precision={precision:.3f} val_recall={recall:.3f} val_f1={f1:.3f}"
        )

    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    )
    precision, recall, f1 = evaluate(model, test_loader)
    save_prediction(model, test_loader)
    print(f"test_precision={precision:.3f} test_recall={recall:.3f} test_f1={f1:.3f}")
    print(f"prediction image: {PREDICTION_IMAGE}")


if __name__ == "__main__":
    main()
