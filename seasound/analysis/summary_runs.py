# -*- coding: utf-8 -*-
"""analysis/summary_runs.py — runs/ 아래 모든 학습 결과의 성능만 터미널에 표로 출력

실행 (seasound/ 폴더에서):
  python -m analysis.summary_runs
  python -m analysis.summary_runs --classes 4     # 4클래스 실험만
  python -m analysis.summary_runs --csv            # runs/summary.csv 로도 저장
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_run(d):
    m = json.loads((d / "test_metrics.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8")) if (d / "config.yaml").exists() else {}
    hist = pd.read_csv(d / "history.csv") if (d / "history.csv").exists() else None
    rec, seg = m["recording"], m["segment"]
    # 전처리 설정: feature_info.json 이 있으면 읽고, 없으면 (다른 컴퓨터) 빈칸
    pre = {}
    fdir = cfg.get("data", {}).get("feature_dir")
    if fdir and (ROOT / fdir / "feature_info.json").exists():
        pre = json.loads((ROOT / fdir / "feature_info.json").read_text(encoding="utf-8"))["config"]["logmel"]
    t = cfg.get("train", {})
    row = {
        "run": d.name,
        "cls": len(cfg.get("data", {}).get("classes", [])) or len(rec["per_class"]),
        "n_fft": pre.get("n_fft", "-"),
        "ref": pre.get("ref", "-"),
        "lr": t.get("lr", "-"),
        "best_ep": m.get("best_epoch"),
        "val_F1": round(hist["val_rec_macro_f1"].max(), 3) if hist is not None else None,
        "test_F1": round(rec["macro_f1"], 3),
        "test_acc": round(rec["accuracy"], 3),
        "seg_F1": round(seg["macro_f1"], 3),
        "pass_0.60": "O" if rec["macro_f1"] >= 0.60 else "X",
    }
    per_class = {c["class"]: round(c["f1"], 3) for c in rec["per_class"]}
    return row, per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, default=None, help="이 클래스 수의 실험만")
    ap.add_argument("--csv", action="store_true", help="runs/summary.csv 로 저장")
    args = ap.parse_args()

    runs = sorted(d for d in (ROOT / "runs").glob("*") if (d / "test_metrics.json").exists())
    if not runs:
        raise SystemExit("runs/ 에 test_metrics.json 이 있는 실험이 없습니다.")
    rows, pcs = [], []
    for d in runs:
        try:
            r, pc = load_run(d)
            rows.append(r)
            pcs.append({"run": d.name, **pc})
        except Exception as e:
            print(f"[skip] {d.name}: {e}")
    df = pd.DataFrame(rows)
    pc = pd.DataFrame(pcs).set_index("run")
    if args.classes:
        df = df[df["cls"] == args.classes]
        pc = pc.loc[df["run"]]

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("\n== 실험별 성능 (test = 녹음 단위, 명세 기준 0.60)")
    print(df.to_string(index=False))
    print("\n== 클래스별 F1 (test, 녹음 단위)")
    print(pc.dropna(axis=1, how="all").fillna("-").to_string())

    for n in sorted(df["cls"].unique()):
        sub = df[df["cls"] == n]
        best_val = sub.loc[sub["val_F1"].idxmax()] if sub["val_F1"].notna().any() else None
        best_test = sub.loc[sub["test_F1"].idxmax()]
        print(f"\n[{n}클래스] test 최고: {best_test['run']} ({best_test['test_F1']})", end="")
        if best_val is not None:
            print(f"  |  val 최고(설정 선택 기준): {best_val['run']} ({best_val['val_F1']})", end="")
    print("\n\n※ 클래스 수가 다른 실험끼리는 점수를 직접 비교할 수 없음. 설정 선택은 val_F1, 보고는 test_F1.")

    if args.csv:
        out = ROOT / "runs" / "summary.csv"
        df.merge(pc.reset_index(), on="run").to_csv(out, index=False, encoding="utf-8-sig")
        print(f"저장: {out}")


if __name__ == "__main__":
    main()
