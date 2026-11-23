# -*- coding: utf-8 -*-
"""analysis/check_features.py — 전처리 결과 점검

실행 (seasound/ 폴더에서):
  python -m analysis.check_features data/features/3acce8e86614
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    if len(sys.argv) < 2:
        raise SystemExit("사용법: python -m analysis.check_features data/features/<feature_id>")
    d = Path(sys.argv[1])
    info = json.loads((d / "feature_info.json").read_text(encoding="utf-8"))
    logs = [json.loads(l) for l in (d / "preprocess.jsonl").read_text(encoding="utf-8").splitlines()]
    recs = pd.DataFrame([l for l in logs if l.get("event") == "recording"])
    man = pd.read_csv(d / "manifest.csv", dtype={"recording_id": str})
    ok = True

    print(f"== 1. 처리 결과  ({d})")
    print(f"   ok={info['n_ok']}  skipped={info['n_skipped']}  error={info['n_error']}  "
          f"세그먼트={info['n_segments']}")
    bad = recs[recs["status"] != "ok"]
    if len(bad):
        ok = False
        print("   [!] 실패/건너뜀 녹음:")
        for _, r in bad.head(10).iterrows():
            print(f"       {r['file']}: {r.get('error') if isinstance(r.get('error'), str) else r.get('reason')}")
        if len(bad) > 10:
            print(f"       ... 외 {len(bad) - 10}개")

    print("\n== 2. 클래스 × split  (녹음 수 / 세그먼트 수)")
    rec_t = man.groupby(["label", "split"])["recording_id"].nunique().unstack(fill_value=0)
    seg_t = man.groupby(["label", "split"]).size().unstack(fill_value=0)
    cols = [c for c in ("train", "val", "test") if c in rec_t.columns]
    table = pd.DataFrame({c: rec_t[c].astype(str) + " / " + seg_t[c].astype(str) for c in cols})
    print(table.to_string())
    empty = [(lab, c) for lab in rec_t.index for c in cols if rec_t.loc[lab, c] == 0]
    if empty:
        print(f"   [참고] 녹음이 0개인 칸: {empty}")

    print("\n== 3. 누수 검사 (한 녹음이 여러 split에 있는지)")
    multi = man.groupby("recording_id")["split"].nunique()
    leak = multi[multi > 1]
    print(f"   {'[!] 누수 ' + str(len(leak)) + '개' if len(leak) else 'OK: 없음'}")
    ok &= len(leak) == 0

    print("\n== 4. 세그먼트 수 공식 확인  floor((길이-5)/2.5)+1")
    seg_cfg = info["config"]["segment"]
    good = recs[recs["status"] == "ok"].copy()
    good["n_segments"] = good["n_segments"].astype(int)
    L, H = seg_cfg["length_sec"], seg_cfg["hop_sec"]
    expect = np.floor((good["wav_duration_sec"] - L) / H + 1e-9) + 1
    mism = good[expect.astype(int) != good["n_segments"]]
    print(f"   {'OK' if mism.empty else '[!] 불일치 ' + str(len(mism)) + '개'}"
          f"  (녹음당 세그먼트: 최소 {good['n_segments'].min()}, 중앙 {int(good['n_segments'].median())}, "
          f"최대 {good['n_segments'].max()})")

    print("\n== 5. catalog 길이 vs 실제 WAV 길이 (duration_diff_sec)")
    if "duration_diff_sec" in good:
        diff = good["duration_diff_sec"].abs()
        print(f"   |차이| 최대 {diff.max():.2f}s, 1초 넘는 녹음 {int((diff > 1).sum())}개, "
              f"300초 근처 {int(((diff - 300).abs() < 5).sum())}개")
    total_h = good["wav_duration_sec"].sum() / 3600
    print(f"   전체 길이 {total_h:.1f}시간, 처리 시간 합계 {good['elapsed_sec'].sum() / 60:.1f}분")

    print("\n== 6. 특징 파일 샘플 검사")
    f0 = man["feature_file"].iloc[0]
    a = np.load(d / f0, mmap_mode="r")
    sample = np.asarray(a[0], dtype=np.float32)
    finite = np.isfinite(sample).all()
    print(f"   {f0}: shape={a.shape} dtype={a.dtype}  값 범위 {sample.min():.1f} ~ {sample.max():.1f} dB"
          f"  {'OK' if finite else '[!] NaN/Inf 있음'}")
    ok &= bool(finite)

    n = info["norm_stats"]
    print("\n== 7. 정규화 통계 (train만)")
    print(f"   mean={n['mean']:.4f}  std={n['std']:.4f}  (train 녹음 {n['n_train_recordings']}개)")
    print(f"\n== 해시  manifest={info['manifest_sha256'][:16]}  features={info['features_sha256'][:16]}")
    print(f"\n결론: {'학습 진행 가능' if ok else '[!] 위 표시된 항목 확인 필요'}")


if __name__ == "__main__":
    main()
