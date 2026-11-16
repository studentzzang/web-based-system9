
# SeaSound - 폴더 및 파일 설명

## data_io/ 설명

### 하는 일
1. 데이터셋의 WAV 파일을 읽고 처리
2. 학습용 데이터 목록과 분할 관리
3. 오디오 전처리를 위한 공통 함수 제공

### 사용하는 곳
1. Explorer: WAV 로딩 및 전처리
2. AI Classifier: 학습 데이터 로딩 및 전처리


## loader.py 설명

### 하는 일
1. WAV 읽기
WAV에 저장된 오디오 샘플을 float32 배열로 읽음

2. 샘플링레이트 반환
원본 WAV의 32,000Hz 같은 값을 함께 반환

3. 부가 기능
필요한 시간 구간만 읽고, 경로·Mono·시간 범위를 검사

### 사용하는 곳
1. Explorer: WAV를 불러와 STFT로 주파수 분석할 때
2. AI Classifier: WAV를 불러와 5초 세그먼트로 분할하기 전


## preprocess.py 설명

### 하는 일
1. 리샘플링
원본 WAV의 샘플링레이트를 16kHz로 변환

2. 데이터 형식 유지
결과를 float32 배열로 반환

3. 부가 기능
잘못된 샘플링레이트와 배열 형식 등을 검사

### 사용하는 곳
1. Explorer: STFT 분석 전 오디오 샘플링레이트 통일
2. AI Classifier: AI 입력 데이터 전처리


## split.py 설명

### 하는 일
1. 데이터 분할
VTUAD 녹음을 Train / Val / Test로 분할

2. 시드 고정
동일한 조건으로 실행하면 같은 분할 결과 생성

3. CSV 관리
분할 정보를 catalog.csv에 기록

### 사용하는 곳
1. AI Classifier: 학습·검증·평가 데이터 구분
2. 데이터 관리: 분할 결과 재현 및 검증



## configs/ 설명

### 하는 일
1. STFT 등의 분석 설정 저장
2. YAML을 통해 FFT 크기, Hop Length 등을 관리
3. 분석 조건을 코드와 분리하여 재현성 확보

### 사용하는 곳
1. stft_cache.py: STFT 계산 설정 불러오기
2. 향후 분석 조건별 비교 실험


## data/ 설명

### 하는 일
1. VTUAD WAV 원본 데이터 보관
2. catalog.csv로 녹음 목록과 Train/Val/Test 관리
3. metadata.csv로 원본 메타데이터 관리

### 사용하는 곳
1. Explorer: 녹음 검색 및 선택
2. AI Classifier: 학습·검증·평가 데이터 조회


## analysis_cache/ 설명

### 하는 일
1. STFT 계산 결과를 NPZ 파일로 저장
2. YAML 설정별로 캐시 분리
3. 동일한 분석 결과의 중복 계산 방지

### 사용하는 곳
1. stft_cache.py: 캐시 저장 및 불러오기
2. Explorer: 향후 스펙트로그램 조회


## tests/ 설명

### 하는 일
1. Python 기능이 정상 작동하는지 자동 검증
2. WAV 로딩, 리샘플링, STFT, 캐시 기능 테스트
3. 코드 변경 후 기존 기능의 정상 작동 여부 확인

### 주요 파일
- test_loader.py: WAV 로더 검증
- test_preprocess.py: 리샘플링 검증
- test_stft.py: STFT 계산 검증
- test_stft_cache.py: NPZ 캐시 검증

### 사용하는 곳
1. 개발 중 기능 단위 테스트
2. GitHub에 코드 업로드하기 전 동작 점검
