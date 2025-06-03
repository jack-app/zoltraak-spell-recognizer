import re
from typing import Tuple

import librosa
import numpy as np


def parse_filename(filename: str) -> Tuple[int, str, int]:
    """
    ファイル名から話者ID、内容、テイク番号を抽出します。

    Args:
        filename (str): ファイル名（例: "1_zolt_1.wav"）

    Returns:
        Tuple[int, str, int]: (話者ID, 内容, テイク番号)

    Raises:
        ValueError: ファイル名の形式が不正な場合
    """
    base = filename.split("/")[-1].replace(".wav", "")
    match = re.match(r"(\d+)_([a-zA-Z0-9]+)_(\d+)", base)
    if match:
        speaker, content, take = match.groups()
        return int(speaker), content, int(take)
    else:
        raise ValueError(f"Invalid filename: {filename}")


def remove_silence(
    waveform: np.ndarray, sr: int = 16000, top_db: int = 30
) -> np.ndarray:
    """
    無音区間を除去します。

    Args:
        waveform (np.ndarray): 入力音声データ
        sr (int): サンプルレート
        top_db (int): 無音判定の閾値（dB）

    Returns:
        np.ndarray: 無音区間を除去した音声データ
    """
    intervals = librosa.effects.split(waveform, top_db=top_db)
    return np.concatenate([waveform[start:end] for start, end in intervals])
