# runs/ — 학습 실험 결과 보는법

`python -m ml.train --config configs/train.yaml` 을 실행할 때마다 이 폴더 아래에 실험 폴더가 하나씩 생긴다.

```
runs/
└─ 20261123-000204_resnet18_unified_s42/     # {날짜-시각}_{tag}_s{seed}
   ├─ config.yaml                    # 이번 실행에 실제로 쓴 설정 (시드 포함)
   ├─ train.jsonl                    # 학습 로그 (한 줄 = 한 이벤트)
   ├─ history.csv                    # 에폭별 지표 표
   ├─ test_metrics.json              # 최종 테스트 지표 + 혼동행렬
   ├─ test_predictions_recording.csv # 테스트 녹음별 정답·예측·확률
   └─ best.pt                        # best 모델 가중치 (Classifier 모듈이 사용)
```

실험 ID(폴더 이름)만 알면 그 실험의 설정, 데이터, 결과를 모두 다시 찾을 수 있다.

---

## 1. 무엇부터 보나

| 순서 | 볼 것 | 확인할 내용 |
|---|---|---|
| 1 | 터미널 마지막 출력 또는 `test_metrics.json` | **녹음 단위 macro F1 ≥ 0.60** (프로젝트 기준 목표치) |
| 2 | `test_metrics.json` 의 `per_class` | 어떤 클래스가 평균을 깎는지 |
| 3 | `test_metrics.json` 의 `confusion_matrix` | 틀린 녹음이 어느 클래스로 갔는지 |
| 4 | `history.csv` | 과적합 여부, best 에폭 위치 |
| 5 | `test_predictions_recording.csv` | 틀린 녹음 목록 → 원인 분석 대상 선정 |

---

## 2. 파일별 설명

### config.yaml
학습에 쓴 설정 사본. `--seed` 등 명령줄로 바꾼 값도 반영되어 있다.
같은 결과를 다시 내려면 이 파일로 다시 실행하면 된다.

```powershell
python -m ml.train --config runs/<실험ID>/config.yaml
```

### train.jsonl
학습 로그. `event` 값으로 종류를 구분한다.

| event | 내용 |
|---|---|
| `start` | 실험 ID, 시드, 설정 전체, 사용한 전처리(`feature_id`), **분할 파일 해시**(`split_manifest_sha256`), 정규화 통계, 클래스별 녹음·세그먼트 수, 실행 환경(Python·OS·GPU·라이브러리 버전) |
| `class_weights` | 클래스별 train 세그먼트 수와 손실 가중치 |
| `epoch` | 에폭별 손실·정확도·macro F1, 클래스별 val F1, 개선 여부 |
| `early_stop` | 조기 종료 시점과 best 에폭 |
| `test` | best 모델의 테스트 결과 요약 |
| `done` | 총 소요 시간, 목표치(0.60) 달성 여부 |

두 실험이 **같은 데이터로 학습했는지**는 `start` 의 `feature_id`, `split_manifest_sha256` 이 같은지로 확인한다.

### history.csv
에폭별 지표 표. 엑셀로 열면 된다.

| 열 | 의미 |
|---|---|
| `train_loss` / `val_loss` | 학습/검증 손실 |
| `train_acc` | 학습 세그먼트 정확도 |
| `val_seg_acc`, `val_seg_macro_f1` | 검증 세그먼트 단위 지표 |
| `val_rec_acc`, `val_rec_macro_f1` | 검증 **녹음 단위** 지표 (best 모델 선택 기준) |
| `lr` | 해당 에폭 학습률 (cosine으로 감소) |
| `epoch_sec` | 에폭 소요 시간 |

### test_metrics.json
best 모델로 테스트한 결과. `segment` 와 `recording` 두 단위로 따로 들어 있다.

```
best_epoch, loss
segment:   { accuracy, macro_f1, macro_f1_all_classes, classes_missing, per_class[], confusion_matrix }
recording: { 위와 같음 }   ← 보고서 기준 지표
```

### test_predictions_recording.csv
테스트 녹음 하나당 한 줄: `recording_id, true, pred, p_<클래스>...`
틀린 녹음만 보기:

```powershell
Import-Csv runs\<실험ID>\test_predictions_recording.csv | Where-Object { $_.true -ne $_.pred } | Format-Table recording_id, true, pred
```

확률(`p_*`)이 두 클래스 사이에서 비슷하면 애매한 녹음, 한쪽으로 확실하게 틀렸으면 라벨 오류나 데이터 특성을 의심한다.

### best.pt
`select_metric`(기본 `val_rec_macro_f1`)이 가장 높았던 에폭의 모델.
가중치와 함께 클래스 목록, 정규화 mean/std, 전처리 설정이 저장되어 있어 추론(Classifier) 때 그대로 쓴다.

---

## 3. 평가지표 설명

### 세그먼트 단위 vs 녹음 단위
- **세그먼트 단위**: 5초 조각 하나하나를 따로 채점한다.
- **녹음 단위**: 한 녹음의 모든 세그먼트 확률을 **평균**해서 녹음 하나당 한 번 판정한다 (명세: "녹음 단위 판정: 세그먼트 확률 평균").
- 프로젝트 목표(Macro F1 0.60)는 **녹음 단위**로 본다. 세그먼트 단위는 참고용.
- 긴 녹음은 세그먼트가 많아서 세그먼트 단위 지표를 지배한다. 녹음 단위는 녹음마다 한 표라 이 편향이 없다.

### 클래스별 지표 (`per_class`)
어떤 클래스 C 에 대해:

| 지표 | 정의 | 뜻 |
|---|---|---|
| **Precision** (정밀도) | C 로 예측한 것 중 실제 C 비율 | 낮으면: 다른 클래스를 C 로 잘못 끌어옴 |
| **Recall** (재현율) | 실제 C 중 C 로 맞힌 비율 | 낮으면: C 를 다른 클래스로 놓침 |
| **F1** | Precision과 Recall의 조화평균 = 2PR/(P+R) | 둘 다 높아야 높음 |
| **support** | 테스트에 있는 실제 C 의 개수 | 작으면 F1이 크게 출렁임 (1개 차이로 ±0.2 가능) |

### 전체 지표
| 지표 | 정의 | 주의 |
|---|---|---|
| **accuracy** | 전체 중 맞힌 비율 | 많은 클래스(Tug/Work)를 잘 맞히면 높게 나옴. 불균형 데이터에선 과대평가됨 |
| **macro_f1** | 테스트에 **존재하는** 클래스들의 F1 단순 평균 | **주 지표.** 클래스 크기와 무관하게 모든 클래스를 같은 비중으로 봄 |
| **macro_f1_all_classes** | 설정의 모든 클래스 F1 평균 (테스트에 없는 클래스는 0) | 참고용. 없는 클래스 때문에 깎인 값 |
| **classes_missing** | 테스트에 하나도 없는 클래스 | 있으면 보고서에 명시 |

### 혼동행렬 (`confusion_matrix`)
- **행 = 정답, 열 = 예측.** 클래스 순서는 `per_class` 순서와 같다.
- 대각선 = 맞힌 개수, 대각선 밖 = 틀린 개수.
- 한 행에서 대각선 밖 큰 값 → 그 클래스가 **어디로 새는지** (Recall 문제).
- 한 열에서 대각선 밖 큰 값 → 그 클래스가 **무엇을 끌어오는지** (Precision 문제).

### 손실 (loss)
- 교차 엔트로피 손실. 낮을수록 정답 클래스에 높은 확률을 줌.
- 클래스 가중치(`class_weight_power`)가 적용된 값이라 실험 간 직접 비교 시 같은 가중치 설정끼리만 비교.

---

## 4. history.csv 로 학습 상태 판단

| 패턴 | 의미 | 대응 (configs/train.yaml) |
|---|---|---|
| train_loss ↓, val_loss ↑ (초반부터) | **과적합** — 녹음을 외우는 중 | `lr` 낮추기, `weight_decay` 올리기, 증강 강화 |
| 둘 다 높게 정체 | 학습 부족 | `lr`, `epochs` 늘리기 |
| best 에폭이 1~3 | 검증셋이 작아 선택이 불안정하거나 과적합 | 여러 시드로 반복해 평균 보기 |
| val_rec_macro_f1 이 에폭마다 크게 출렁임 | 검증 녹음 수가 적음 | 지표 해석 시 오차 감안 |

---

## 5. 재현성 확인

명세: 같은 설정으로 3회 재실행 시 결과 일치 여부 (AI 지표 ±0.02 이내).

```powershell
.\scripts\run_train_repro.ps1      # 같은 설정·시드로 3회 학습 → runs/ 에 repro1~3 생성
```

세 폴더의 `test_metrics.json` → `recording.macro_f1` 을 비교한다.
GPU 일부 연산은 완전히 결정적이지 않아 소수점 아래 차이가 날 수 있다.

---

## 6. 실험 비교할 때

- 바꾼 설정 하나당 실험 하나. `tag` 를 바꿔서 실행하면 폴더 이름으로 구분된다.
  ```powershell
  python -m ml.train --config configs/train.yaml --tag lr1e-4
  ```
- 비교 전 `train.jsonl` 의 `start` 에서 `feature_id`, `split_manifest_sha256` 이 같은지 확인 (같은 데이터인지).
- 보고서 표 예시:

| 실험 ID | 바꾼 것 | 클래스 수 | rec macro F1 | seg macro F1 | best epoch |
|---|---|---|---|---|---|
| 20261123-000204_resnet18_unified_s42 | 기본 설정 | 6 | 0.480 | 0.446 | 4 |