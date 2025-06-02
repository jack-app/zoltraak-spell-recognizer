import sounddevice as sd
import numpy as np
import mmap
import os
from typing import List, Tuple, Optional

RECOGNIZE_INTERVAL_SECONDS = 0.5
RECOGNIZE_WINDOW_SECONDS = 1.0

def list_audio_devices() -> List[Tuple[int, str]]:
    """
    利用可能なオーディオ入力デバイスを一覧表示します。

    Returns:
        List[Tuple[int, str]]: デバイスインデックスと名前を含むタプルのリスト
    """
    devices = sd.query_devices()
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((i, device['name']))
    return input_devices

def select_audio_device() -> int:
    """
    利用可能なオーディオデバイスを表示し、ユーザーに選択させます。

    Returns:
        int: 選択されたデバイスのインデックス
    """
    devices = list_audio_devices()
    print("\n利用可能なオーディオ入力デバイス:")
    for idx, name in devices:
        print(f"{idx}: {name}")

    while True:
        try:
            selection = int(input("\nデバイスインデックスを選択してください: "))
            if any(idx == selection for idx, _ in devices):
                return selection
            print("無効なデバイスインデックスです。もう一度お試しください。")
        except ValueError:
            print("有効な数字を入力してください。")

def initialize_memory_mapped_files() -> None:
    """
    呪文検出用のメモリマップトファイルを初期化します。
    存在しない場合は /tmp/is_spell_detected と /tmp/spell_id を作成します。
    """
    # is_spell_detectedの初期化（1ビット）
    if not os.path.exists('/tmp/is_spell_detected'):
        with open('/tmp/is_spell_detected', 'wb') as f:
            f.write(b'\x00')

    # spell_idの初期化（int16）
    if not os.path.exists('/tmp/spell_id'):
        with open('/tmp/spell_id', 'wb') as f:
            f.write(np.int16(0).tobytes())

def update_memory_mapped_files(spell_id: int) -> None:
    """
    メモリマップトファイルを呪文検出結果で更新します。

    Args:
        spell_id (int): 検出された呪文ID（呪文が検出されない場合は0）
    """
    # is_spell_detectedの更新
    with open('/tmp/is_spell_detected', 'r+b') as f:
        mm = mmap.mmap(f.fileno(), 0)
        mm[0] = 1 if spell_id > 0 else 0
        mm.close()

    # spell_idの更新
    with open('/tmp/spell_id', 'r+b') as f:
        mm = mmap.mmap(f.fileno(), 0)
        mm.write(np.int16(spell_id).tobytes())
        mm.close()

def recognize_speech(audio_data: np.ndarray) -> int:
    """
    音声データから音声を認識します。
    現在はプレースホルダーとして0を返します。

    Args:
        audio_data (np.ndarray): 処理する音声データ

    Returns:
        int: 呪文ID（呪文が検出されない場合は0、検出された場合は1以上）
    """
    # TODO: 実際の音声認識を実装する
    return 0

def main() -> None:
    """
    呪文認識システムを実行するメイン関数。
    """
    print("呪文認識システムを初期化中...")

    # メモリマップトファイルの初期化
    initialize_memory_mapped_files()

    # オーディオデバイスの選択
    device_idx = select_audio_device()

    # デバイス情報の取得
    device_info = sd.query_devices(device_idx)
    sample_rate = int(device_info['default_samplerate'])

    # バッファサイズの計算
    window_samples = int(RECOGNIZE_WINDOW_SECONDS * sample_rate)
    interval_samples = int(RECOGNIZE_INTERVAL_SECONDS * sample_rate)

    print(f"\n呪文認識を開始します。デバイス: {device_info['name']}")
    print(f"サンプルレート: {sample_rate} Hz")
    print(f"ウィンドウサイズ: {RECOGNIZE_WINDOW_SECONDS} 秒")
    print(f"インターバル: {RECOGNIZE_INTERVAL_SECONDS} 秒")

    try:
        # 最初のウィンドウ分の音声を録音
        audio_buffer = sd.rec(window_samples, samplerate=sample_rate, channels=1, dtype=np.float32)
        sd.wait()

        while True:
            # インターバル分の音声を録音
            new_audio = sd.rec(interval_samples, samplerate=sample_rate, channels=1, dtype=np.float32)
            sd.wait()

            # バッファを更新（古いデータを削除し、新しいデータを追加）
            audio_buffer = np.roll(audio_buffer, -interval_samples, axis=0)
            audio_buffer[-interval_samples:] = new_audio

            # 音声認識
            spell_id = recognize_speech(audio_buffer)

            print(f"呪文ID: {spell_id}")

            # 呪文が検出された場合、メモリマップトファイルを更新
            if spell_id > 0:
                update_memory_mapped_files(spell_id)

    except KeyboardInterrupt:
        print("\n呪文認識を停止中...")
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    main()
