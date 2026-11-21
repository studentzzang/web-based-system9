# 재현성 확인: 같은 설정·같은 시드로 3회 학습 (명세: 3회 재실행 결과 일치 여부)
for ($i = 1; $i -le 3; $i++) {
    python -m ml.train --config configs/train.yaml --tag "repro$i"
}
