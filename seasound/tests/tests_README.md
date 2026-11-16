# SeaSound - 실행 및 테스트 안내

## 1. 테스트 실행 환경

모든 명령어는 seasound 폴더에서 실행한다.
Python 가상환경(.venv)을 활성화한 상태여야 한다.

## 2. 개별 테스트 명령어

### M2 - WAV 로더 테스트
python -m unittest discover -s tests -p "test_loader.py" -v

검증 내용:
- WAV 파일 로딩
- Mono 및 float32 배열 확인
- 부분 구간 로딩과 잘못된 시간 범위 검사

결과: 3개 테스트 통과


### M3 - 리샘플링 테스트
python -m unittest discover -s tests -p "test_preprocess.py" -v

검증 내용:
- 32kHz -> 16kHz 변환
- 동일 샘플링레이트 입력 처리
- 잘못된 샘플링레이트 검사

결과: 3개 테스트 통과


### M4 - STFT 계산 테스트
python -m unittest discover -s tests -p "test_stft.py" -v

검증 내용:
- 실제 WAV의 STFT 배열 크기 확인
- 125Hz 합성 신호의 주파수 검출
- 잘못된 샘플링레이트 검사

결과: 3개 테스트 통과


### M5 - STFT 캐시 테스트
python -m unittest discover -s tests -p "test_stft_cache.py" -v

검증 내용:
- STFT 결과의 NPZ 저장
- 두 번째 호출 시 기존 캐시 재사용
- 계산 전후 배열 일치 확인

결과: 1개 테스트 통과


## 3. 모든 테스트 한 번에 실행

python -m unittest discover -s tests -p "test_*.py" -v

현재 M2~M5 개별 테스트는 총 10개 통과했다.
통합 실행 결과는 별도로 확인해야 한다.


## 4. 주요 설정값

샘플링레이트: 16000Hz
FFT 크기: 2048
Hop Length: 512
Window: Hann
Center: False

설정 파일: configs/analysis.yaml


## 5. STFT 캐시 저장 위치

analysis_cache/[설정 해시]/vessel/0.npz

저장 데이터:
- frequencies: 주파수 배열
- times: 시간 배열
- power: STFT 파워 배열
- config_hash: 설정 해시
- source_size: WAV 크기
- source_mtime_ns: WAV 수정 시각
- source_sha256: WAV 내용 해시

캐시 파일은 필요할 때 다시 생성할 수 있으므로
analysis_cache/ 폴더는 GitHub에서 제외한다.


## 6. 라이브러리 설치

python -m pip install -r requirements.txt

현재 환경의 설치 라이브러리 목록 생성:
python -m pip freeze > requirements.txt
