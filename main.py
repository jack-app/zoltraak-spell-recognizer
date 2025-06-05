import mmap
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import torch
from matplotlib.animation import FuncAnimation

# 日本語表示のための設定
pass  # NOQA
import japanize_matplotlib  # NOQA

from audio_encoder.hubert import HuBERTEncoder
from source_separation.sepformer import SepformerSeparator
from spell_classifier import SpellClassifier


class SpellRecognizer:
    """
    呪文認識システムのメインクラス。
    音声入力から呪文を認識し、メモリマップトファイルを通じて結果を共有します。
    """

    # 認識のインターバル（秒）
    RECOGNIZE_INTERVAL_SECONDS = 0.5
    # 認識に使用するウィンドウサイズ（秒）
    RECOGNIZE_WINDOW_SECONDS = 0.75

    def __init__(self):
        """
        呪文認識システムを初期化します。
        """
        self.device_idx = None
        self.sample_rate = None
        self.window_samples = None
        self.interval_samples = None
        self.audio_buffer = None

        # プロット関連の変数
        self.fig = None
        self.ax = None
        self.line = None
        self.animation = None

        # 分類器の初期化
        self.source_separator = SepformerSeparator()
        self.audio_encoder = HuBERTEncoder(max_length=4608)
        self.classifier = SpellClassifier(
            source_separator=self.source_separator,
            audio_encoder=self.audio_encoder,
            distance_threshold=0.5,
            distance_type="euclidean",
        )

    def setup_plot(self):
        """
        波形表示用のプロットを初期化します。
        """
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, self.RECOGNIZE_WINDOW_SECONDS)
        self.ax.set_xlabel("時間 (秒)")
        self.ax.set_ylabel("振幅")
        self.ax.set_title("リアルタイム音声波形")
        self.ax.grid(True)

        # 時間軸の生成
        self.time = np.linspace(0, self.RECOGNIZE_WINDOW_SECONDS, self.window_samples)
        (self.line,) = self.ax.plot(self.time, np.zeros(self.window_samples))

    def update_plot(self, frame):
        """
        プロットを更新します。

        Args:
            frame: アニメーションフレーム（未使用）

        Returns:
            tuple: 更新されたプロット要素
        """
        if self.audio_buffer is not None:
            self.line.set_ydata(self.audio_buffer.flatten())
        return (self.line,)

    @staticmethod
    def list_audio_devices() -> List[Tuple[int, str]]:
        """
        利用可能なオーディオ入力デバイスを一覧表示します。

        Returns:
            List[Tuple[int, str]]: デバイスインデックスと名前を含むタプルのリスト
        """
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                input_devices.append((i, device["name"]))
        return input_devices

    @staticmethod
    def select_audio_device() -> int:
        """
        利用可能なオーディオデバイスを表示し、ユーザーに選択させます。

        Returns:
            int: 選択されたデバイスのインデックス
        """
        devices = SpellRecognizer.list_audio_devices()
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

    @staticmethod
    def initialize_memory_mapped_files() -> None:
        """
        呪文検出用のメモリマップトファイルを初期化します。
        存在しない場合は /tmp/is_spell_detected と /tmp/spell_id を作成します。
        """
        # is_spell_detectedの初期化（1ビット）
        if not os.path.exists("/tmp/is_spell_detected"):
            with open("/tmp/is_spell_detected", "wb") as f:
                f.write(b"\x00")

        # spell_idの初期化（int16）
        if not os.path.exists("/tmp/spell_id"):
            with open("/tmp/spell_id", "wb") as f:
                f.write(np.int16(0).tobytes())

    @staticmethod
    def update_memory_mapped_files(spell_id: int) -> None:
        """
        メモリマップトファイルを呪文検出結果で更新します。

        Args:
            spell_id (int): 検出された呪文ID（呪文が検出されない場合は0）
        """
        # is_spell_detectedの更新
        with open("/tmp/is_spell_detected", "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            mm[0] = 1 if spell_id > 0 else 0
            mm.close()

        # spell_idの更新
        with open("/tmp/spell_id", "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            mm.write(np.int16(spell_id).tobytes())
            mm.close()

    def recognize_speech(self, audio_data: np.ndarray, sr: int) -> int:
        """
        音声データから呪文を認識します。

        Args:
            audio_data (np.ndarray): 処理する音声データ
            sr (int): サンプリングレート

        Returns:
            int: 呪文ID（呪文が検出されない場合は0、検出された場合は1以上）
        """
        # 音声データをtorch.Tensorに変換
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)

        spell_id = self.classifier.classify(audio_tensor, sr)
        if spell_id is not None:
            return spell_id

        return 0

    def initialize(self) -> None:
        """
        認識システムを初期化します。
        デバイスの選択、サンプルレートの設定、バッファサイズの計算を行います。
        """
        # メモリマップトファイルの初期化
        self.initialize_memory_mapped_files()

        # オーディオデバイスの選択
        self.device_idx = self.select_audio_device()

        # デバイス情報の取得
        device_info = sd.query_devices(self.device_idx)
        self.sample_rate = int(device_info["default_samplerate"])

        # バッファサイズの計算
        self.window_samples = int(self.RECOGNIZE_WINDOW_SECONDS * self.sample_rate)
        self.interval_samples = int(self.RECOGNIZE_INTERVAL_SECONDS * self.sample_rate)

        print(f"\n呪文認識を開始します。デバイス: {device_info['name']}")
        print(f"サンプルレート: {self.sample_rate} Hz")
        print(f"ウィンドウサイズ: {self.RECOGNIZE_WINDOW_SECONDS} 秒")
        print(f"インターバル: {self.RECOGNIZE_INTERVAL_SECONDS} 秒")

        # プロットの初期化
        self.setup_plot()

    def run(self) -> None:
        """
        呪文認識システムを実行します。
        """
        try:
            # 最初のウィンドウ分の音声を録音
            self.audio_buffer = sd.rec(
                self.window_samples,
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
            )
            sd.wait()

            # アニメーションの開始
            self.animation = FuncAnimation(
                self.fig,
                self.update_plot,
                interval=int(self.RECOGNIZE_INTERVAL_SECONDS * 1000),  # ミリ秒単位
                blit=True,
            )
            plt.show(block=False)

            while True:
                # インターバル分の音声を録音
                new_audio = sd.rec(
                    self.interval_samples,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype=np.float32,
                )
                sd.wait()

                # バッファを更新（古いデータを削除し、新しいデータを追加）
                self.audio_buffer = np.roll(
                    self.audio_buffer, -self.interval_samples, axis=0
                )
                self.audio_buffer[-self.interval_samples :] = new_audio

                # 音声認識
                spell_id = self.recognize_speech(
                    self.audio_buffer.flatten(), self.sample_rate
                )

                if spell_id > 0:
                    print(f"呪文を検出: ID {spell_id}")
                    # 呪文が検出された場合、メモリマップトファイルを更新
                    self.update_memory_mapped_files(spell_id)

                # プロットの更新
                plt.pause(0.001)  # プロットの更新を確実に行う

        except KeyboardInterrupt:
            print("\n呪文認識を停止中...")
            if self.animation is not None:
                self.animation.event_source.stop()
            plt.close()
        # except Exception as e:
        #     print(f"エラー: {e}")
        #     if self.animation is not None:
        #         self.animation.event_source.stop()
        #     plt.close()


def main() -> None:
    """
    メイン関数。呪文認識システムを初期化して実行します。
    """
    recognizer = SpellRecognizer()
    recognizer.initialize()
    recognizer.run()


if __name__ == "__main__":
    main()
