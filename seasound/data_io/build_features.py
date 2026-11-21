# -*- coding: utf-8 -*-
"""data_io/build_features.py — catalog.csv -> 5초 세그먼트 log-mel + manifest + 정규화 통계 + 전처리 로그

실행 (seasound/ 폴더에서):
  python -m data_io.build_features --config configs/preprocess.yaml

출력: data/features/{feature_id}/
  recordings/{recording_id}.npy   녹음별 log-mel (n_seg, 1, n_mels, frames)
  manifest.csv                    세그먼트 목록 (recording_id, seg_idx, start_sec, split, label, feature_file)
  norm_stats.json                 train split만으로 계산한 mean/std
  feature_info.json               설정, catalog/manifest/특징 해시, 요약 (학습 코드가 읽음)
  preprocess.jsonl                전처리 로그 (녹음별 한 줄)
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_io.loader import load_wav
from data_io.preprocess import resample_audio
from dsp.mel import logmel, segment_starts
from logging_utils.logger import JsonlLogger, env_info, sha256_file, sha256_json


def safe_name(recording_id):
    return recording_id.replace("\\", "/").replace("/", "__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/preprocess.yaml"))
    ap.add_argument("--output-root", type=Path, default=None,
                    help="재현성 확인용으로 다른 폴더에 한 번 더 만들 때 사용")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N개 녹음만 (빠른 테스트용)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    catalog_path = Path(cfg["catalog"])
    catalog = pd.read_csv(catalog_path, dtype=str, keep_default_na=False)
    for col in ("recording_id", "path", "unified_label", "split"):
        if col not in catalog.columns:
            raise ValueError(f"catalog.csv에 '{col}' 열이 없습니다.")
    if args.limit:
        catalog = catalog.head(args.limit)

    catalog_sha = sha256_file(catalog_path)
    feature_id = sha256_json({"config": cfg, "catalog_sha256": catalog_sha,
                              "limit": args.limit})[:12]
    out_dir = (args.output_root or Path(cfg["output_root"])) / feature_id
    rec_dir = out_dir / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preprocess.jsonl").unlink(missing_ok=True)  # 같은 feature_id 재실행 시 로그 새로 시작
    logger = JsonlLogger(out_dir / "preprocess.jsonl")
    logger.log("start", feature_id=feature_id, config=cfg, catalog=catalog_path,
               catalog_sha256=catalog_sha, n_recordings=len(catalog), env=env_info())
    print(f"[info] feature_id={feature_id}  out={out_dir}")

    wav_root = Path(cfg.get("wav_root", "data/vtuad_wav"))
    if not (wav_root / "vessel").is_dir() and not (wav_root / "background").is_dir():
        raise SystemExit(f"[fail] WAV 폴더를 찾을 수 없습니다: {wav_root.resolve()}\n"
                         f"       그 안에 vessel/ 와 background/ 폴더가 있어야 합니다. "
                         f"configs/preprocess.yaml 의 wav_root 를 확인하세요.")
    seg_cfg, mel_cfg = cfg["segment"], cfg["logmel"]
    target_sr = int(cfg["target_sr"])
    seg_len = round(seg_cfg["length_sec"] * target_sr)
    store_dtype = np.dtype(cfg.get("store_dtype", "float16"))

    rows, n_ok, n_skip, n_err = [], 0, 0, 0
    t_all = time.perf_counter()
    for i, r in enumerate(catalog.itertuples(index=False), 1):
        t0 = time.perf_counter()
        base = {"recording_id": r.recording_id, "file": r.path,
                "label": r.unified_label, "split": r.split}
        try:
            audio, orig_sr = load_wav(r.path, vessel_root=wav_root / "vessel",
                                      background_root=wav_root / "background")
            orig_duration = len(audio) / orig_sr
            audio, sr = resample_audio(audio, orig_sr, target_rate=target_sr)
            starts = segment_starts(len(audio), sr, seg_cfg["length_sec"], seg_cfg["hop_sec"])

            info = {**base, "orig_sr": orig_sr, "target_sr": sr,
                    "wav_duration_sec": round(orig_duration, 3)}
            cat_dur = getattr(r, "duration_sec", "")
            if cat_dur not in ("", None):
                info["catalog_duration_sec"] = float(cat_dur)
                info["duration_diff_sec"] = round(orig_duration - float(cat_dur), 3)

            if not starts:
                n_skip += 1
                logger.log("recording", **info, status="skipped", reason="shorter_than_segment",
                           n_segments=0, elapsed_sec=round(time.perf_counter() - t0, 3))
                continue

            feats = np.stack([logmel(audio[s:s + seg_len], sr, **mel_cfg) for s in starts])
            feats = feats[:, None].astype(store_dtype)  # (n_seg, 1, n_mels, frames)
            feat_file = f"recordings/{safe_name(r.recording_id)}.npy"
            np.save(out_dir / feat_file, feats)

            rows.extend({"recording_id": r.recording_id, "seg_idx": k,
                         "start_sec": round(s / sr, 3), "split": r.split,
                         "label": r.unified_label, "feature_file": feat_file}
                        for k, s in enumerate(starts))
            n_ok += 1
            logger.log("recording", **info, status="ok", n_segments=len(starts),
                       feature_shape=list(feats.shape), feature_file=feat_file,
                       elapsed_sec=round(time.perf_counter() - t0, 3))
        except Exception as e:
            n_err += 1
            logger.log("recording", **base, status="error", error=repr(e),
                       elapsed_sec=round(time.perf_counter() - t0, 3))
        if i % 20 == 0 or i == len(catalog):
            print(f"[{i}/{len(catalog)}] ok={n_ok} skipped={n_skip} error={n_err}", flush=True)

    if not rows:
        # 처리된 녹음이 하나도 없으면 원인을 보여주고 중단
        errors = [json.loads(line) for line in
                  (out_dir / "preprocess.jsonl").read_text(encoding="utf-8").splitlines()]
        errors = [e for e in errors if e.get("status") in ("error", "skipped")]
        print(f"\n[fail] 처리된 녹음이 0개입니다. (error={n_err}, skipped={n_skip})")
        for e in errors[:3]:
            print(f"  - {e.get('file')}: {e.get('error') or e.get('reason')}")
        print(f"  전체 로그: {out_dir / 'preprocess.jsonl'}")
        raise SystemExit(1)

    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")

    # 정규화 통계: train split만, 저장된 값(float16) 그대로 사용 -> 학습 입력과 정확히 일치
    s = s2 = 0.0
    n = 0
    train_files = sorted(manifest.loc[manifest["split"] == "train", "feature_file"].unique())
    for f in train_files:
        a = np.load(out_dir / f).astype(np.float64)
        s += float(a.sum())
        s2 += float((a * a).sum())
        n += a.size
    if n == 0:
        raise RuntimeError("train split 세그먼트가 없습니다. catalog의 split 열을 확인하세요.")
    mean = s / n
    std = float(np.sqrt(max(s2 / n - mean * mean, 1e-12)))
    norm = {"mean": mean, "std": std, "n_values": n, "computed_on": "train",
            "n_train_recordings": len(train_files)}
    (out_dir / "norm_stats.json").write_text(json.dumps(norm, indent=2), encoding="utf-8")

    # 특징 전체 해시 (재현성 확인용: 두 번 만들어서 이 값이 같으면 결과 동일)
    h = hashlib.sha256()
    for f in sorted(manifest["feature_file"].unique()):
        h.update(f.encode())
        h.update(sha256_file(out_dir / f).encode())
    features_sha = h.hexdigest()

    summary = (manifest.groupby(["label", "split"])
               .agg(recordings=("recording_id", "nunique"), segments=("seg_idx", "size"))
               .reset_index())
    feature_info = {
        "feature_id": feature_id, "config": cfg,
        "catalog": catalog_path.as_posix(), "catalog_sha256": catalog_sha,
        "manifest_sha256": sha256_file(manifest_path), "features_sha256": features_sha,
        "norm_stats": norm, "n_ok": n_ok, "n_skipped": n_skip, "n_error": n_err,
        "n_segments": len(manifest), "summary": summary.to_dict(orient="records"),
    }
    (out_dir / "feature_info.json").write_text(
        json.dumps(feature_info, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.log("done", **{k: v for k, v in feature_info.items() if k not in ("config", "summary")},
               elapsed_sec=round(time.perf_counter() - t_all, 1))

    print(summary.pivot_table(index="label", columns="split", values="segments",
                              fill_value=0).to_string())
    print(f"[norm] mean={mean:.4f} std={std:.4f}")
    print(f"[sha256] manifest={feature_info['manifest_sha256'][:16]}  features={features_sha[:16]}")
    print(f"[done] ok={n_ok} skipped={n_skip} error={n_err}  -> {out_dir}")
    if n_err:
        print("[warn] 에러 난 녹음은 preprocess.jsonl 에서 status=error 로 확인하세요.")


if __name__ == "__main__":
    main()
