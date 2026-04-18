# Censor Faces. Three paths. 100% MIT.

[![Live demo](https://img.shields.io/badge/demo-live-brightgreen)](https://libreyolo.github.io/use-cases/censor-faces/demo/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LibreYOLO/use-cases/blob/main/censor-faces/notebooks/pipeline.ipynb)
[![PyPI](https://img.shields.io/pypi/v/libreyolo?label=libreyolo)](https://pypi.org/project/libreyolo/)
[![HF model](https://img.shields.io/badge/%F0%9F%A4%97-face--rfdetr--nano-yellow)](https://huggingface.co/LibreYOLO/face-rfdetr-nano)
[![License](https://img.shields.io/badge/license-MIT-green)](../LICENSE)

A face detector and a face blur, end to end. Use it in Python, use it in the browser, or train your own. All three run on the same RF-DETR Nano weights.

## Path 1: use it in the browser (zero install)

Open https://libreyolo.github.io/use-cases/censor-faces/demo/ in Chrome, allow the camera, and every face in frame is blurred in real time. The 108 MB ONNX is pulled from [LibreYOLO/face-rfdetr-nano](https://huggingface.co/LibreYOLO/face-rfdetr-nano) on first visit and cached by the browser.

The whole thing is one self-contained HTML file at [`demo/index.html`](./demo/index.html). Copy it anywhere, `open index.html`, same result.

## Path 2: use it in Python (60 seconds)

```bash
pip install onnxruntime opencv-python huggingface_hub numpy

python -m src.use_pretrained --image my_photo.jpg
# writes my_photo.censored.jpg
```

Same weights, downloaded from the same HF repo, run through `onnxruntime`. No torch, no libreyolo, no training stack.

## Path 3: build it (under an hour)

For when you want to learn the pipeline, fine-tune on your own data, or just not trust someone else's weights.

```bash
git clone https://github.com/LibreYOLO/use-cases
cd use-cases/censor-faces
pip install -r requirements.txt

python -m src.download_widerface         # ~360 MB, 1-2 min
python -m src.train --epochs 30          # ~30-40 min on MPS / CUDA
python -m src.eval                       # mAP on val
python -m src.censor --image my.jpg      # blur faces with your own weights
```

The defaults reproduce the pretrained release: 80-image WIDERFACE subset, 30 epochs, RF-DETR Nano. Bump `--train-images` and `--epochs` for a stronger model.

## Or run it in Colab

Click the Colab badge above. The notebook runs both paths end-to-end on a free T4 GPU.

## What's where

```
demo/
  index.html              path 1: in-browser face blur, self-contained
  favicon.svg
src/
  common.py               shared blur helper (no heavy deps)
  download_widerface.py   build only: dataset acquisition + YOLO conversion
  train.py                build only: RF-DETR Nano fine-tuning loop
  eval.py                 build only: mAP on val split
  censor.py               build only: load .pt weights, detect, blur
  use_pretrained.py       path 2: HF download + ONNX inference + blur
  webcam.py               build only: live webcam demo with a trained .pt
notebooks/
  pipeline.ipynb          Colab walkthrough of paths 2 and 3
```

Each script is independent. Re-run any one in isolation.

## Honest numbers

The released weights were trained on **80 images for 30 epochs** (the smoke-test config above). Validation AP@0.50 = 0.93 on a 20-image val split. Both numbers are noisy because the splits are tiny.

In practice that's enough to blur most clearly visible faces in well-lit photos. It's not enough for production use against tiny, occluded, or off-angle faces. For that, train on more data:

| `--train-images` | Realistic AP@0.50 | Time on MPS |
|---|---|---|
| 80 (released) | 0.85-0.95 (noisy) | 30-40 min |
| 500 | 0.85-0.90 (stable) | 2-3 hours |
| 2000 | 0.90+ | 8-12 hours |

## Export your own weights for the browser

Path 1 points at the pretrained HF model. If you trained your own via path 3 and want the browser to use your weights, export to ONNX:

```python
from libreyolo import LibreYOLO
LibreYOLO("weights/face.pt", nb_classes=1).export(format="onnx", opset=17)
# writes weights/rfdetr_n.onnx
```

Then edit `demo/index.html` and swap `MODEL_URL` for the path to your ONNX.

## FAQ

**Why RF-DETR Nano?** Transformer detector that learns very fast from few examples (the released weights got AP@0.50 = 0.93 from 150 gradient updates total). 108 MB ONNX, runs on CPU, GPU, or Apple Silicon.

**Can I use my own dataset?** Yes. Put it in Roboflow YOLO layout (`<dir>/train/{images,labels}/`, `<dir>/valid/{images,labels}/`, plus a `data.yaml` with `names`). Then `python -m src.train --data path/to/your/dataset`.

**Why is `rfdetr` pinned to 1.4.1?** libreyolo's wrapper imports `rfdetr.main` which was removed after that version. Pinning is the cleanest workaround until the wrapper updates.

**Does this respect privacy?** This blurs faces in your own data. It is not a face recognition or identification tool.

## License

MIT. Code, scripts, notebook. The released weights inherit the WIDERFACE dataset license for the data used to train them.
