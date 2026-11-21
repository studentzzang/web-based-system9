# 가상 수중청음기 계산과 한계

현재 AIS는 NOAA Marine Cadastre의 2026-06-30 샌프란시스코만 기록이며, 샘플 음원은 VTUAD WAV다. 두 자료는 같은 선박을 같은 시각에 녹음한 것이 아니다. 샘플의 선종만 AIS 선종과 맞춘다. 데이터·선종별 샘플 경로·소스 레벨 가정은 `configs/simulator.yaml`에서 바꿀 수 있다.

## 수신 레벨

- 선박과 청음기 사이의 거리는 위경도에서 구면 거리로 계산한다.
- `TL = 20 log10(R_m) + alpha(f) * R_km` (dB). `alpha`는 Thorp 근사식의 dB/km 값이다. 거리가 100 m보다 가까우면 계산에는 100 m를 사용한다.
- `SL = 선종별 기준값 + clamp(3 log2(max(SOG, 1) / 10), -12, 12)` (dB). 기준값과 속도 계수는 측정값이 아닌 설정상의 가정이다.
- 각 선박의 `RL = SL - TL`이며, 여러 선박의 수신 레벨은 `10 log10(sum(10^(RL_i/10)))`으로 합친다.

손 계산 예: 주파수 1 kHz, 거리 1 km이면 Thorp 흡수계수는 약 0.069 dB/km다. 따라서 `TL = 20 log10(1000) + 0.069 * 1 = 60.069 dB`다. Cargo 기준 소스 레벨 160 dB, 속도 10 kn이면 `RL = 99.931 dB`다. `tests/test_simulator.py`에서 이 예제를 검증한다.

## 음향 합성 및 재현

반경 안에서 최근 15분 이내에 AIS 위치가 있는 **모든 선박**의 선종별 샘플을 사용한다. YAML의 시드·MMSI·샘플 경로로 각 녹음의 5초 구간을 결정하고, 16 kHz 단일 채널로 변환한다. 각 구간의 평균과 RMS를 맞춘 뒤 계산된 RL 차이에 따라 이득을 적용해 합친다. 최종 신호는 피크만 부드럽게 제한한다. 같은 AIS·WAV·YAML·입력 좌표와 시각이면 같은 WAV 바이트가 생성된다.

브라우저에서 청음 위치·반경·시간을 바꾸면 합성 음향을 새로 만들어 교체한다. 청음 요청마다 `runs/sim-.../` 아래에 설정 사본, `simulation.jsonl`(선택 시각과 시간별 선박·거리·SL·TL·RL·샘플 경로), `preview.wav`를 저장한다.

이 모델은 구면 확산과 단일 주파수 흡수만 사용한다. 해저 지형, 수온 구조, 선박별 실제 방사 소음, 수중에서의 방향성과 반사는 반영하지 않는다. 합성 WAV의 재생 볼륨은 측정된 실제 음압으로 보정되지 않았다.

## 입력 파일과 API

- `data/ais_clean.csv`: MMSI, UTC 시각, 위경도, SOG, COG, 선종 코드와 통합 선종 라벨이 필요하다. 기본 경로는 `SEASOUND_AIS_CSV`로 바꿀 수 있다.
- `data/catalog.csv`와 WAV 폴더: 샘플은 선종 라벨에 맞춰 YAML에서 상대 WAV 경로로 지정한다. 기본 WAV 폴더는 `data/vtuad_wav`이며 `SEASOUND_AUDIO_ROOT`로 바꿀 수 있다.
- `configs/simulator.yaml`: 주파수, 시드, 거리 하한, 선종별 소스 레벨, 속도 계수, 샘플 경로, 음향 길이·샘플링레이트·재생 이득을 지정한다. `SEASOUND_SIMULATOR_CONFIG`로 다른 설정 파일을 사용할 수 있다.
- `GET /api/simulator/meta`: AIS 날짜·범위, 기본 청음 위치와 모델 정보를 반환한다.
- `GET /api/simulator/scene?at=...&lat=...&lon=...&radius_km=...&window_minutes=...`: 지도 위치, 반경 내 선박별 SL·TL·RL 및 시간별 합산 레벨을 JSON으로 반환한다. 반경은 0.5~20 km, 그래프 구간은 15~120분이다.
- `GET /api/simulator/audio`는 같은 입력으로 5초 WAV를 반환하고, 실험 ID를 `X-Simulation-Run-Id` 헤더에 담는다. 로그 출력 위치는 `SEASOUND_SIMULATOR_RUNS_ROOT`로 바꿀 수 있다.
