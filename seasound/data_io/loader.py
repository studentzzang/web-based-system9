import math
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf


# SeaSound 프로젝트 루트 경로
root = Path(__file__).resolve().parent.parent

# 실제 WAV 파일이 저장된 폴더   
vessel_root = root / "data/vtuad_wav/vessel"
background_root = root / "data/vtuad_wav/background"


def load_wav(
    path: str,
    vessel_root: str | Path = vessel_root,
    background_root: str | Path = background_root,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
) -> tuple[np.ndarray, int]:

    # 1. catalog.csv의 상대 경로 확인
    relative = PurePosixPath(path.replace("\\", "/"))

    if len(relative.parts) != 2 or relative.suffix.lower() != ".wav":
        raise ValueError(f"잘못된 WAV 경로: {path}")

    folder, filename = relative.parts

    # 2. 실제 WAV 폴더 선택
    if folder == "vessel":
        selected_root = Path(vessel_root).resolve(strict=True)

    elif folder == "background":
        selected_root = Path(background_root).resolve(strict=True)

    else:
        raise ValueError(f"알 수 없는 폴더: {folder}")

    file_path = (selected_root / filename).resolve(strict=True)

    if not file_path.is_relative_to(selected_root):
        raise ValueError("데이터 폴더 밖의 파일은 읽을 수 없습니다.")

    # 3. 요청 시간 검증
    if not math.isfinite(start_sec) or start_sec < 0:
        raise ValueError("시작 시간은 0 이상이어야 합니다.")

    if duration_sec is not None:
        if not math.isfinite(duration_sec) or duration_sec <= 0:
            raise ValueError("구간 길이는 0보다 커야 합니다.")

    # 4. WAV 파일 읽기
    with sf.SoundFile(file_path) as wav:

        if wav.channels != 1:
            raise ValueError("Mono WAV만 지원합니다.")

        sample_rate = wav.samplerate

        start_frame = round(start_sec * sample_rate)

        if duration_sec is None:
            end_frame = wav.frames
        else:
            end_frame = round(
                (start_sec + duration_sec) * sample_rate
            )

        if not (0 <= start_frame < end_frame <= wav.frames):
            raise ValueError("요청한 구간이 WAV 범위를 벗어났습니다.")

        wav.seek(start_frame)

        audio = wav.read(
            frames=end_frame - start_frame,
            dtype="float32",
        )

        if len(audio) != end_frame - start_frame:
            raise ValueError("WAV 데이터를 끝까지 읽지 못했습니다.")

    # 5. 오디오 배열과 원본 샘플링레이트 반환
    return audio, sample_rate
