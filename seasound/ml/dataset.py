# -*- coding: utf-8 -*-
"""ml/dataset.py — manifest.csv 기반 세그먼트 Dataset."""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def spec_augment(x, n_freq=2, n_time=2, max_freq=16, max_time=24):
    """x: (1, F, T). 주파수/시간 대역을 0으로 가림 (정규화 후 0 = 평균값)."""
    _, F, T = x.shape
    for _ in range(n_freq):
        w = random.randint(0, max_freq)
        if 0 < w < F:
            f0 = random.randint(0, F - w)
            x[:, f0:f0 + w, :] = 0.0
    for _ in range(n_time):
        w = random.randint(0, max_time)
        if 0 < w < T:
            t0 = random.randint(0, T - w)
            x[:, :, t0:t0 + w] = 0.0
    return x


class SegmentDataset(Dataset):
    def __init__(self, manifest, feature_dir, class_to_idx, mean, std, augment=False):
        self.feature_dir = Path(feature_dir)
        self.files = manifest["feature_file"].tolist()
        self.seg_idx = manifest["seg_idx"].astype(int).tolist()
        self.labels = [class_to_idx[c] for c in manifest["label"]]
        self.rec_ids = manifest["recording_id"].tolist()
        self.mean, self.std, self.augment = float(mean), float(std), augment
        self._arrays = {}  # 워커마다 녹음 파일을 한 번만 열어 둠 (mmap)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        arr = self._arrays.get(f)
        if arr is None:
            arr = np.load(self.feature_dir / f, mmap_mode="r")
            self._arrays[f] = arr
        x = torch.from_numpy(np.array(arr[self.seg_idx[i]], dtype=np.float32))
        x = (x - self.mean) / self.std
        if self.augment:
            x = spec_augment(x)
        return x, self.labels[i], i
