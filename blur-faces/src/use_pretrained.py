"""Download the pretrained ONNX face detector and blur faces in an image. Zero training.

The "use it" path: skip the dataset + training, just blur faces. Runs through
`onnxruntime` so you don't need torch or libreyolo installed.

Usage:
    python -m src.use_pretrained --image path/to/photo.jpg
    python -m src.use_pretrained --image in.jpg --out out.jpg --conf 0.30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from src.common import blur_boxes

HF_REPO = "LibreYOLO/face-rfdetr-nano"
HF_FILE = "face.onnx"
INPUT = 384
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_bgr):
    """BGR image -> (1, 3, 384, 384) float32 tensor, ImageNet-normalized."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT, INPUT), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[None, ...].astype(np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode(boxes, logits, src_w, src_h, conf):
    """RF-DETR outputs -> list of (x1, y1, x2, y2) in source pixel coords.

    boxes: (Q, 4) cxcywh normalized [0, 1]
    logits: (Q, C) raw class logits; apply sigmoid, take max class per query
    """
    scores = sigmoid(logits)
    best = scores.max(axis=-1)
    keep = best >= conf
    if not keep.any():
        return []
    b = boxes[keep]
    cx = b[:, 0] * src_w
    cy = b[:, 1] * src_h
    bw = b[:, 2] * src_w
    bh = b[:, 3] * src_h
    x1 = np.clip(cx - bw / 2, 0, src_w)
    y1 = np.clip(cy - bh / 2, 0, src_h)
    x2 = np.clip(cx + bw / 2, 0, src_w)
    y2 = np.clip(cy + bh / 2, 0, src_h)
    return list(zip(x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist()))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--conf", type=float, default=0.30)
    args = p.parse_args()

    if not args.image.exists():
        print(f"missing image: {args.image}", file=sys.stderr)
        return 1
    out_path = args.out or args.image.with_name(args.image.stem + ".blurred.jpg")

    print(f"downloading {HF_REPO}/{HF_FILE} (cached after first run)")
    onnx_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)

    print(f"loading ONNX from {onnx_path}")
    # CoreML chokes on some RF-DETR ops; stick to CUDA (if present) or CPU.
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [p for p in providers if p in ort.get_available_providers()]
    session = ort.InferenceSession(onnx_path, providers=providers or None)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    print(f"reading {args.image}")
    img = cv2.imread(str(args.image))
    if img is None:
        print(f"could not read image: {args.image}", file=sys.stderr)
        return 1
    h, w = img.shape[:2]

    tensor = preprocess(img)
    outs = session.run(output_names, {input_name: tensor})
    # Identify which output is boxes (last dim 4) vs logits.
    boxes_raw = next(o for o in outs if o.shape[-1] == 4)
    logits_raw = next(o for o in outs if o is not boxes_raw)
    boxes_out, logits_out = boxes_raw[0], logits_raw[0]

    boxes = decode(boxes_out, logits_out, w, h, conf=args.conf)
    print(f"{len(boxes)} face(s), blurring")
    blur_boxes(img, boxes)
    cv2.imwrite(str(out_path), img)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
