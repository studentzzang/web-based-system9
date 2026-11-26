# -*- coding: utf-8 -*-
"""ml/infer.py — best.pt 로 WAV 한 개를 판정 (웹과 명령줄 공용)

명령줄 테스트 (seasound/ 폴더에서):
  python -m ml.infer --model runs/<실험ID>/best.pt --wav 경로/파일.wav

판정 방법: 학습 때와 같은 전처리(best.pt에 저장된 설정) -> 5초/2.5초 세그먼트
-> 세그먼트별 확률 -> 평균 (명세: 녹음 단위 판정 = 세그먼트 확률 평균)
"""
import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from data_io.preprocess import resample_audio
from dsp.mel import logmel, segment_starts
from ml.model import build_resnet18


class AudioClassifier:
    def __init__(self, model_path, device="auto"):
        self.model_path = Path(model_path)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.class_names = list(ckpt["class_names"])
        self.mean, self.std = float(ckpt["mean"]), float(ckpt["std"])
        pre = ckpt["preprocess"]
        self.target_sr = int(pre["target_sr"])
        self.seg_len_sec = float(pre["segment"]["length_sec"])
        self.seg_hop_sec = float(pre["segment"]["hop_sec"])
        self.mel_cfg = dict(pre["logmel"])
        # 구조만 만들고(pretrained=False, 인터넷 불필요) 학습된 가중치를 덮어씀
        self.model = build_resnet18(len(self.class_names), pretrained=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()
        self.info = {"model_path": self.model_path.as_posix(), "run_id": ckpt.get("run_id"),
                     "feature_id": ckpt.get("feature_id"), "epoch": ckpt.get("epoch"),
                     "seed": ckpt.get("seed"), "classes": self.class_names,
                     "n_fft": self.mel_cfg.get("n_fft"), "fmax": self.mel_cfg.get("fmax"),
                     "ref": self.mel_cfg.get("ref"), "device": self.device.type}

    def _features(self, audio, sr, min_duration_sec):
        if audio.ndim == 2:                       # 다채널 -> mono
            audio = audio.mean(axis=1)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        duration = len(audio) / sr
        if duration < min_duration_sec:
            raise ValueError(f"녹음이 너무 짧습니다 ({duration:.2f}초 < {min_duration_sec}초)")
        audio, sr = resample_audio(audio, int(sr), target_rate=self.target_sr)
        seg_len = round(self.seg_len_sec * sr)
        padded = False
        if len(audio) < seg_len:                  # 5초 미만: 뒤를 0으로 채움
            audio = np.pad(audio, (0, seg_len - len(audio)))
            padded = True
        starts = segment_starts(len(audio), sr, self.seg_len_sec, self.seg_hop_sec)
        feats = np.stack([logmel(audio[s:s + seg_len], sr, **self.mel_cfg) for s in starts])
        # 학습 때와 같게 float16 저장값을 거친 뒤 정규화
        feats = feats.astype(np.float16).astype(np.float32)
        feats = (feats - self.mean) / self.std
        return feats[:, None], starts, sr, duration, padded

    @torch.no_grad()
    def predict(self, audio, sr, min_duration_sec=1.0, batch_size=64):
        t0 = time.perf_counter()
        feats, starts, sr16, duration, padded = self._features(audio, sr, min_duration_sec)
        probs = []
        for i in range(0, len(feats), batch_size):
            x = torch.from_numpy(feats[i:i + batch_size]).to(self.device)
            probs.append(torch.softmax(self.model(x).float(), dim=1).cpu().numpy())
        seg_probs = np.concatenate(probs)
        rec_probs = seg_probs.mean(axis=0)
        k = int(rec_probs.argmax())
        return {
            "label": self.class_names[k],
            "confidence": round(float(rec_probs[k]), 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(self.class_names, rec_probs)},
            "n_segments": len(starts),
            "duration_sec": round(duration, 3),
            "padded": padded,
            "segments": [{"start_sec": round(s / sr16, 2),
                          "label": self.class_names[int(p.argmax())],
                          "probabilities": {c: round(float(v), 4) for c, v in zip(self.class_names, p)}}
                         for s, p in zip(starts, seg_probs)],
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def predict_bytes(self, data, min_duration_sec=1.0):
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return self.predict(audio, sr, min_duration_sec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    t0 = time.perf_counter()
    clf = AudioClassifier(args.model, args.device)
    t_load = time.perf_counter() - t0
    res = clf.predict_bytes(Path(args.wav).read_bytes())
    seg_count = res.pop("segments")
    print(json.dumps({"model": clf.info, **res}, indent=2, ensure_ascii=False))
    print(f"[time] 모델 로딩 {t_load:.2f}s, 판정 {res['elapsed_ms']}ms, 세그먼트 {len(seg_count)}개")


if __name__ == "__main__":
    main()