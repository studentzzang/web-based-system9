# -*- coding: utf-8 -*-
"""ml/train.py — 설정 파일 기반 분류기 학습 + 녹음 단위 평가 + 학습 로그

실행 (seasound/ 폴더에서):
  python -m ml.train --config configs/train.yaml
  python -m ml.train --config configs/train.yaml --seed 43     # 시드만 바꿔 재실행

출력: runs/{실험ID}/
  config.yaml        실제 사용한 설정 사본 (시드 반영)
  train.jsonl        학습 로그 (start / epoch / test / done)
  history.csv        에폭별 지표 표
  best.pt            best 모델 (+ 클래스, 정규화 통계, 전처리 설정 -> Classifier가 그대로 사용)
  test_metrics.json  세그먼트·녹음 단위 지표, 혼동행렬
  test_predictions_recording.csv  녹음별 정답/예측/클래스별 확률 (원인 분석용)
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # GPU 결정적 연산용 (torch import 전)

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from logging_utils.logger import JsonlLogger, env_info, make_run_id
from ml.dataset import SegmentDataset
from ml.metrics import recording_level, report
from ml.model import build_resnet18


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        # 일부 연산(adaptive avg pool backward 등)은 GPU에서 완전 결정적 구현이 없어 경고만 냄
        torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def predict(model, loader, device, criterion, amp):
    model.eval()
    probs, labels, idxs, loss_sum, n = [], [], [], 0.0, 0
    for x, y, i in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
            loss_sum += criterion(logits.float(), y).item() * x.size(0)
        n += x.size(0)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        labels.append(y.cpu().numpy())
        idxs.append(i.numpy())
    return (np.concatenate(probs), np.concatenate(labels),
            np.concatenate(idxs), loss_sum / max(n, 1))


def evaluate(model, loader, dataset, device, criterion, amp, class_names):
    probs, labels, idxs, loss = predict(model, loader, device, criterion, amp)
    seg = report(labels, probs.argmax(1), class_names)
    rec_ids = np.asarray(dataset.rec_ids)[idxs]
    uniq, rec_probs, rec_labels = recording_level(probs, rec_ids, labels)
    rec = report(rec_labels, rec_probs.argmax(1), class_names)
    return {"loss": loss, "segment": seg, "recording": rec,
            "rec_ids": uniq, "rec_probs": rec_probs, "rec_labels": rec_labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    ap.add_argument("--seed", type=int, default=None, help="설정 파일의 seed 덮어쓰기")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.tag:
        cfg["tag"] = args.tag
    seed, tc = int(cfg["seed"]), cfg["train"]
    seed_everything(seed, tc.get("deterministic", True))

    # ---- 데이터 ----
    feature_dir = Path(cfg["data"]["feature_dir"])
    finfo = json.loads((feature_dir / "feature_info.json").read_text(encoding="utf-8"))
    norm = finfo["norm_stats"]
    class_names = list(cfg["data"]["classes"])
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    manifest = pd.read_csv(feature_dir / "manifest.csv", dtype={"recording_id": str})
    excluded = sorted(set(manifest["label"]) - set(class_names))
    manifest = manifest[manifest["label"].isin(class_names)].reset_index(drop=True)
    parts = {s: manifest[manifest["split"] == s].reset_index(drop=True)
             for s in ("train", "val", "test")}

    counts = {s: {"recordings": df.groupby("label")["recording_id"].nunique().to_dict(),
                  "segments": df["label"].value_counts().to_dict()} for s, df in parts.items()}

    # ---- 실험 폴더와 로그 ----
    run_id = make_run_id(cfg["tag"], seed)
    run_dir = Path(cfg.get("output_root", "runs")) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    logger = JsonlLogger(run_dir / "train.jsonl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(tc.get("amp", True)) and device.type == "cuda"
    logger.log("start", run_id=run_id, seed=seed, config=cfg, device=device.type,
               feature_id=finfo["feature_id"], catalog_sha256=finfo["catalog_sha256"],
               split_manifest_sha256=finfo["manifest_sha256"],
               features_sha256=finfo["features_sha256"], norm_stats=norm,
               classes=class_names, excluded_labels=excluded, counts=counts, env=env_info())
    print(f"[info] run_id={run_id} device={device} classes={class_names}")
    print(f"[info] excluded labels: {excluded}")
    for s, df in parts.items():
        print(f"[info] {s}: {df['recording_id'].nunique()} recordings, {len(df)} segments")

    datasets = {s: SegmentDataset(df, feature_dir, class_to_idx, norm["mean"], norm["std"],
                                  augment=(s == "train" and tc.get("spec_augment", True)))
                for s, df in parts.items()}
    nw = int(tc.get("num_workers", 0))
    g = torch.Generator()
    g.manual_seed(seed)
    common = dict(num_workers=nw, pin_memory=device.type == "cuda",
                  worker_init_fn=worker_init_fn, persistent_workers=nw > 0)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=tc["batch_size"], shuffle=True,
                            drop_last=True, generator=g, **common),
        "val": DataLoader(datasets["val"], batch_size=tc["batch_size"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], batch_size=tc["batch_size"], shuffle=False, **common),
    }

    # ---- 모델 ----
    model = build_resnet18(len(class_names), cfg["model"].get("pretrained", True)).to(device)
    power = float(tc.get("class_weight_power", 0.0))
    seg_counts = np.bincount(datasets["train"].labels, minlength=len(class_names)).astype(np.float64)
    if power > 0:
        w = np.where(seg_counts > 0, (1.0 / np.maximum(seg_counts, 1.0)) ** power, 0.0)
        w = w * (len(w) / max(w.sum(), 1e-12))
        weight = torch.tensor(w, dtype=torch.float32, device=device)
    else:
        w, weight = np.ones(len(class_names)), None
    logger.log("class_weights", train_segment_counts=seg_counts.astype(int).tolist(),
               weights=[round(float(v), 4) for v in w])
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # ---- 학습 ----
    select = tc.get("select_metric", "val_rec_macro_f1")
    best, best_epoch, bad = -1.0, -1, 0
    history = []
    t_start = time.time()
    for epoch in range(1, tc["epochs"] + 1):
        t0 = time.time()
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for x, y, _ in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
                loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += x.size(0)
        scheduler.step()

        ev = evaluate(model, loaders["val"], datasets["val"], device, criterion, amp, class_names)
        row = {"epoch": epoch, "train_loss": loss_sum / max(seen, 1),
               "train_acc": correct / max(seen, 1), "val_loss": ev["loss"],
               "val_seg_acc": ev["segment"]["accuracy"], "val_seg_macro_f1": ev["segment"]["macro_f1"],
               "val_rec_acc": ev["recording"]["accuracy"], "val_rec_macro_f1": ev["recording"]["macro_f1"],
               "lr": scheduler.get_last_lr()[0], "epoch_sec": round(time.time() - t0, 1)}
        history.append(row)
        improved = row[select] > best
        logger.log("epoch", **row, improved=improved,
                   val_rec_per_class_f1={c["class"]: round(c["f1"], 4)
                                         for c in ev["recording"]["per_class"]})
        print(f"[epoch {epoch:03d}] loss={row['train_loss']:.4f}/{row['val_loss']:.4f} "
              f"seg_f1={row['val_seg_macro_f1']:.4f} rec_f1={row['val_rec_macro_f1']:.4f} "
              f"{'*' if improved else ''} ({row['epoch_sec']}s)", flush=True)

        if improved:
            best, best_epoch, bad = row[select], epoch, 0
            torch.save({"model_state": model.state_dict(), "class_names": class_names,
                        "mean": norm["mean"], "std": norm["std"],
                        "preprocess": finfo["config"], "feature_id": finfo["feature_id"],
                        "epoch": epoch, select: best, "run_id": run_id, "seed": seed},
                       run_dir / "best.pt")
        else:
            bad += 1
            if bad >= tc["patience"]:
                logger.log("early_stop", best_epoch=best_epoch, best=best, patience=tc["patience"])
                print(f"[early-stop] best epoch {best_epoch} ({select}={best:.4f})")
                break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, lineterminator="\n")

    # ---- 테스트 (best 모델) ----
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    ev = evaluate(model, loaders["test"], datasets["test"], device, criterion, amp, class_names)
    test = {"best_epoch": best_epoch, "loss": ev["loss"],
            "segment": ev["segment"], "recording": ev["recording"]}
    (run_dir / "test_metrics.json").write_text(json.dumps(test, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    with open(run_dir / "test_predictions_recording.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["recording_id", "true", "pred"] + [f"p_{c}" for c in class_names])
        for rid, p, t in zip(ev["rec_ids"], ev["rec_probs"], ev["rec_labels"]):
            wr.writerow([rid, class_names[t], class_names[int(p.argmax())]]
                        + [f"{v:.4f}" for v in p])

    r, s = ev["recording"], ev["segment"]
    logger.log("test", best_epoch=best_epoch, rec_macro_f1=r["macro_f1"], rec_acc=r["accuracy"],
               seg_macro_f1=s["macro_f1"], seg_acc=s["accuracy"],
               rec_classes_missing=r["classes_missing"],
               rec_per_class={c["class"]: {k: round(c[k], 4) for k in ("precision", "recall", "f1")}
                              | {"support": c["support"]} for c in r["per_class"]})
    logger.log("done", run_id=run_id, elapsed_sec=round(time.time() - t_start, 1),
               target_macro_f1=0.60, target_met=r["macro_f1"] >= 0.60)

    print(f"[test] recording: macro_f1={r['macro_f1']:.4f} acc={r['accuracy']:.4f} "
          f"| segment: macro_f1={s['macro_f1']:.4f} acc={s['accuracy']:.4f}")
    for c in r["per_class"]:
        print(f"  {c['class']:12s} P={c['precision']:.3f} R={c['recall']:.3f} "
              f"F1={c['f1']:.3f} n={c['support']}")
    if r["classes_missing"]:
        print(f"[warn] test에 없는 클래스: {r['classes_missing']} (macro F1 계산에서 제외)")
    print(f"[done] {run_dir}")


if __name__ == "__main__":
    main()
