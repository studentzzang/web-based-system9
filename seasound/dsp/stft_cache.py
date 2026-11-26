
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf
import yaml

from data_io.preprocess import resample_audio
from dsp.stft import compute_stft


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/analysis.yaml"
WAV_ROOT = ROOT / "data/vtuad_wav"
CACHE_ROOT = ROOT / "analysis_cache"


def load_config():
    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    stft_config = config["stft"]

    if config["sample_rate"] != 16000:
        raise ValueError("현재 STFT는 16kHz만 지원합니다.")

    if (stft_config["window"] != "hann"
            or stft_config["center"] is not False):
        raise ValueError("현재 Hann / center=False만 지원합니다.")

    if not (0 < stft_config["hop_length"] <= stft_config["n_fft"]):
        raise ValueError("잘못된 STFT 설정입니다.")

    return config


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_or_compute_stft(recording_path, *, audio_root=None):
    # 1. YAML 설정 읽기
    config = load_config()
    options = config["stft"]

    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:12]

    # 2. 실제 WAV 경로 찾기
    relative = PurePosixPath(
        recording_path.replace("\\", "/")
    )

    root = Path(audio_root or WAV_ROOT).expanduser().resolve()
    if (relative.is_absolute() or ".." in relative.parts or ":" in recording_path
            or relative.suffix.lower() != ".wav"):
        raise ValueError("잘못된 녹음 경로입니다.")
    if audio_root is None and (len(relative.parts) != 2
                               or relative.parts[0] not in ("vessel", "background")):
        raise ValueError("잘못된 녹음 경로입니다.")
    wav_path = (root / relative).resolve()
    if not wav_path.is_relative_to(root):
        raise ValueError("허용되지 않은 경로입니다.")
    if not wav_path.is_file():
        raise FileNotFoundError("WAV 파일을 찾지 못했습니다.")
    stat = wav_path.stat()

    # 기본 VTUAD 캐시를 재사용하고 외부 데이터셋은 루트별로 구분합니다.
    cache_dir = CACHE_ROOT / config_hash
    if root != WAV_ROOT.resolve():
        root_id = hashlib.sha256(os.fsencode(root)).hexdigest()
        cache_dir = cache_dir / "roots" / root_id
    cache_path = cache_dir / relative.with_suffix(".npz")

    # 4. 기존 캐시 확인
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                valid = (
                    data["config_hash"].item() == config_hash
                )

                if valid:
                    same_file_info = (
                        data["source_size"].item() == stat.st_size
                        and data["source_mtime_ns"].item()
                        == stat.st_mtime_ns
                    )

                    # 파일 정보가 바뀌면 실제 내용 해시 검사
                    if not same_file_info:
                        valid = (
                            data["source_sha256"].item()
                            == sha256_file(wav_path)
                        )

                if valid:
                    return (
                        data["frequencies"],
                        data["times"],
                        data["power"],
                        True,
                    )
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            pass  # 캐시가 손상되었다면 새로 계산

    # 5. 캐시가 없거나 유효하지 않으면 계산
    with sf.SoundFile(wav_path) as source:
        if audio_root is None and source.channels != 1:
            raise ValueError("Mono WAV만 지원합니다.")
        audio = source.read(dtype="float32", always_2d=True).mean(axis=1)
        sr = source.samplerate
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("분석 가능한 WAV 샘플이 필요합니다.")

    audio, sr = resample_audio(
        audio, sr, target_rate=config["sample_rate"]
    )

    frequencies, times, power = compute_stft(
        audio,
        sr,
        n_fft=options["n_fft"],
        hop_length=options["hop_length"],
    )

    # 6. NPZ 저장
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # 완성되지 않은 파일이 정상 캐시가 되지 않도록 임시 저장
    with tempfile.NamedTemporaryFile(
        dir=cache_path.parent,
        suffix=".npz",
        delete=False,
    ) as temp:
        temp_path = Path(temp.name)

    try:
        np.savez_compressed(
            temp_path,
            frequencies=frequencies,
            times=times,
            power=power,
            config_hash=config_hash,
            source_size=stat.st_size,
            source_mtime_ns=stat.st_mtime_ns,
            source_sha256=sha256_file(wav_path),
        )
        os.replace(temp_path, cache_path)
    finally:
        temp_path.unlink(missing_ok=True)

    return frequencies, times, power, False
