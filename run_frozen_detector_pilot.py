"""Smoke-test official frozen detectors on development-only pristine images.

This produces scores, not a benchmark result. The official repositories are
left unchanged. The adapter substitutes the available PyTorch device for the
repositories' hard-coded CUDA calls while retaining their validation transforms.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


HERE = Path(__file__).resolve().parent
CNN_REPO = HERE / "third_party" / "CNNDetection"
CLIP_REPO = HERE / "third_party" / "UniversalFakeDetect"
WORKING = HERE / "pilot_working_png"
OUTPUT = HERE / "data" / "pilot_pristine_scores.csv"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_detector(name, device):
    if name == "cnn":
        sys.path.insert(0, str(CNN_REPO))
        from networks.resnet import resnet50

        checkpoint = CNN_REPO / "weights" / "blur_jpg_prob0.5.pth"
        model = resnet50(num_classes=1)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        description = "RGB; no crop or resize; ToTensor; ImageNet normalization"
        checkpoints = {"cnn_head_and_backbone_sha256": sha256(checkpoint)}
    elif name == "clip":
        sys.path.insert(0, str(CLIP_REPO))
        from models.clip import clip
        from torch import nn

        backbone = HERE / "model_cache" / "ViT-L-14.pt"
        head = CLIP_REPO / "pretrained_weights" / "fc_weights.pth"
        # Reproduce CLIPModel with a local backbone path to keep the cache in
        # this workspace; the official class maps the name ViT-L/14 to 768.
        visual_model, _unused_loader_transform = clip.load(str(backbone), device="cpu")
        class FrozenCLIPHead(nn.Module):
            def __init__(self, visual):
                super().__init__()
                self.model = visual
                self.fc = nn.Linear(768, 1)

            def forward(self, x):
                return self.fc(self.model.encode_image(x))

        model = FrozenCLIPHead(visual_model)
        state = torch.load(head, map_location="cpu", weights_only=False)
        model.fc.load_state_dict(state, strict=True)
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        description = "RGB; CenterCrop(224); ToTensor; CLIP normalization; no resize"
        checkpoints = {"clip_backbone_sha256": sha256(backbone),
                       "clip_head_sha256": sha256(head)}
    else:
        raise ValueError(name)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, transform, description, checkpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["cnn", "clip"], required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process")

    image_paths = sorted(WORKING.glob("*.png"))[:args.limit]
    if not image_paths:
        raise RuntimeError("No development working PNGs found")
    model, transform, preprocessing, checkpoints = load_detector(args.detector, args.device)
    rows = []
    with torch.no_grad():
        for path in image_paths:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
                tensor = transform(image).unsqueeze(0).to(args.device)
            score = float(torch.sigmoid(model(tensor)).flatten()[0].item())
            if not 0 <= score <= 1:
                raise RuntimeError(f"Non-finite or out-of-range score: {path}")
            rows.append({"detector": args.detector,
                         "root_id": path.stem,
                         "sample_id": path.stem + "_pristine_working_png",
                         "split": "development_provisional",
                         "kind": "pristine_control",
                         "score": f"{score:.10f}",
                         "area_band": "", "location": ""})
            print(f"{args.detector} {path.stem} {score:.6f}", flush=True)

    existing = []
    if OUTPUT.exists():
        with OUTPUT.open(newline="") as stream:
            existing = list(csv.DictReader(stream))
    keep = [row for row in existing if row["detector"] != args.detector]
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(keep + rows)
    metadata = {"detector": args.detector,
                "device": args.device,
                "input": "pilot_working_png; development-only pristine derivatives",
                "preprocessing": preprocessing,
                "score": "sigmoid(logit); larger means more likely synthetic",
                "checkpoints": checkpoints,
                "count": len(rows),
                "purpose": "inference smoke test only; no performance estimate"}
    (HERE / "data" / f"pilot_{args.detector}_run.json").write_text(
        json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
