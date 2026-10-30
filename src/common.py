"""MonReader reproducible common methods."""
from __future__ import annotations
import copy, hashlib, json, os, random, re, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score, average_precision_score

REGIMES = ('O', 'S', 'T', 'ST')
CLASS_TO_Y = {'notflip': 0, 'flip': 1}
FRAME_RE = re.compile(r'^(?P<video_id>\d+)_(?P<frame_number>\d+)$')

def setup(seed=42, smoke=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def write_json(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf8')

def get_settings(root):
    root = Path(root).resolve()
    raw = Path(os.getenv('MONREADER_DATA_ROOT', root/'data'/'raw')).expanduser().resolve()
    out = Path(os.getenv('MONREADER_OUTPUT_ROOT', root/'generated'/'outputs')).expanduser().resolve()
    return dict(root=str(root), raw=str(raw), out=str(out), seed=int(os.getenv('MONREADER_SEED', '42')),
                image_size=224, max_epochs=50, patience=4, qualifying_trials=30,
                bootstrap_resamples=5000, validation_batch_cap=32,
                max_batches=None, resnet_pretrained=True, decision_threshold=0.5,
                wandb_project=os.getenv('MONREADER_WANDB_PROJECT','monreader-scratch'),
                wandb_entity=os.getenv('MONREADER_WANDB_ENTITY') or None)

def load_settings(root):
    p=Path(root)/'generated'/'experiment_settings.json'
    if not p.exists():raise FileNotFoundError('Run Notebook 01 first: '+str(p))
    return json.loads(p.read_text())
