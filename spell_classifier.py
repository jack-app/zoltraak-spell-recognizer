import os
from typing import Dict, List, Literal, Optional

import numpy as np
import soundfile as sf
from tqdm import tqdm

from audio_encoder.base import BaseAudioEncoder
from source_separation.base import BaseSourceSeparator


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    ユークリッド距離を計算します。

    Args:
        x (np.ndarray): ベクトル1
        y (np.ndarray): ベクトル2

    Returns:
        float: ユークリッド距離
    """
    return np.linalg.norm(x - y)


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    コサイン距離を計算します。

    Args:
        x (np.ndarray): ベクトル1
        y (np.ndarray): ベクトル2

    Returns:
        float: コサイン距離 (1 - コサイン類似度)
    """
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    マンハッタン距離を計算します。

    Args:
        x (np.ndarray): ベクトル1
        y (np.ndarray): ベクトル2

    Returns:
        float: マンハッタン距離
    """
    return np.sum(np.abs(x - y))


class SampleManager:
    """
    呪文のサンプルデータを管理するクラス。
    各呪文のサンプル特徴量を保持し、距離計算に使用します。
    """

    # 利用可能な距離関数
    DISTANCE_FUNCTIONS = {
        "euclidean": euclidean_distance,
        "cosine": cosine_distance,
        "manhattan": manhattan_distance,
    }

    def __init__(
        self,
        distance_type: Literal["euclidean", "cosine", "manhattan"] = "euclidean",
        distance_threshold: float = 0.5,
    ):
        """
        サンプルマネージャーを初期化します。

        Args:
            distance_type (Literal["euclidean", "cosine", "manhattan"]): 使用する距離関数
            distance_threshold (float): 距離の閾値
        """
        self.samples: Dict[int, List[np.ndarray]] = {}
        self.distance_threshold = distance_threshold
        self.distance_func = self.DISTANCE_FUNCTIONS[distance_type]

    def add_sample(self, spell_id: int, features: np.ndarray) -> None:
        """
        新しいサンプルを追加します。

        Args:
            spell_id (int): 呪文ID
            features (np.ndarray): 特徴量
        """
        if spell_id not in self.samples:
            self.samples[spell_id] = []
        self.samples[spell_id].append(features)

    def get_nearest_spell(
        self, features: np.ndarray, min_samples: int = 3
    ) -> Optional[int]:
        """
        与えられた特徴量に最も近い呪文を判定します。

        Args:
            features (np.ndarray): 判定対象の特徴量
            min_samples (int): 判定に必要な最小サンプル数

        Returns:
            Optional[int]: 最も近い呪文のID。判定できない場合はNone
        """
        if not self.samples:
            return None

        min_distance = float("inf")
        nearest_spell_id = None

        for spell_id, samples in self.samples.items():
            if len(samples) < min_samples:
                continue

            # 各サンプルとの距離を計算
            distances = [self.distance_func(features, sample) for sample in samples]
            avg_distance = np.mean(distances)

            if avg_distance < min_distance:
                min_distance = avg_distance
                nearest_spell_id = spell_id

        # 最小距離が閾値未満の場合のみ呪文を判定
        if min_distance < self.distance_threshold:
            return nearest_spell_id
        return None

    def set_distance_threshold(self, threshold: float) -> None:
        """
        距離の閾値を設定します。

        Args:
            threshold (float): 新しい閾値
        """
        self.distance_threshold = threshold


class SpellClassifier:
    """
    呪文認識のための分類器クラス。
    音源分離と音声エンコーディングを統合し、呪文の認識を行います。
    """

    def __init__(
        self,
        source_separator: BaseSourceSeparator,
        audio_encoder: BaseAudioEncoder,
        distance_type: Literal["euclidean", "cosine", "manhattan"] = "euclidean",
        distance_threshold: float = 0.5,
    ):
        """
        分類器を初期化します。

        Args:
            source_separator (BaseSourceSeparator): 音源分離器のインスタンス
            audio_encoder (BaseAudioEncoder): 音声エンコーダーのインスタンス
            distance_type (Literal["euclidean", "cosine", "manhattan"]): 使用する距離関数
            distance_threshold (float): 距離の閾値
        """
        self.source_separator = source_separator
        self.audio_encoder = audio_encoder
        self.initialized = False
        self.sample_manager = SampleManager(
            distance_type=distance_type,
            distance_threshold=distance_threshold,
        )

    def initialize(self) -> None:
        """
        分類器を初期化します。
        音源分離器と音声エンコーダーを初期化し、サンプルデータを読み込みます。
        """
        if not self.initialized:
            self.source_separator.initialize()
            self.audio_encoder.initialize()
            self._load_anchor_samples()
            self.initialized = True

    def _load_anchor_samples(self) -> None:
        """
        anchorsディレクトリからサンプルデータを読み込みます。
        各クラスIDのディレクトリ内のWAVファイルを読み込み、特徴量に変換します。
        """
        anchors_dir = "anchors"
        if not os.path.exists(anchors_dir):
            print(f"Warning: {anchors_dir} directory not found")
            return

        # 各クラスIDのディレクトリを処理
        class_ids = [
            d
            for d in os.listdir(anchors_dir)
            if os.path.isdir(os.path.join(anchors_dir, d))
        ]

        for class_id_str in tqdm(class_ids, desc="Loading class IDs"):
            try:
                class_id = int(class_id_str)
                class_dir = os.path.join(anchors_dir, class_id_str)

                # ディレクトリ内のWAVファイルを処理
                wav_files = [f for f in os.listdir(class_dir) if f.endswith(".wav")]

                for wav_file in tqdm(
                    wav_files,
                    desc=f"Loading WAV files for class {class_id}",
                    leave=False,
                ):
                    wav_path = os.path.join(class_dir, wav_file)
                    try:
                        # WAVファイルを読み込み
                        audio_data, sample_rate = sf.read(wav_path)

                        # 特徴量に変換
                        features = self.audio_encoder.encode(audio_data, sample_rate)

                        # サンプルとして追加
                        self.sample_manager.add_sample(class_id, features)
                    except Exception as e:
                        print(f"Error loading {wav_path}: {e}")

            except ValueError:
                print(f"Warning: Invalid class ID directory: {class_id_str}")

    def process_audio(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> List[np.ndarray]:
        """
        音声データを処理します。
        音源分離を行い、分離された音声をエンコードします。

        Args:
            audio_data (np.ndarray): 入力音声データ
            sample_rate (int): サンプルレート

        Returns:
            List[np.ndarray]: 各音源の特徴量のリスト。
                最初の要素が目的の音源の特徴量、残りの要素がその他の音源の特徴量となります。
        """
        if not self.initialized:
            raise RuntimeError(
                "Classifier is not initialized. Call initialize() first."
            )

        # 音源分離
        separated_audios = self.source_separator.separate(audio_data, sample_rate)

        # 各音源をエンコード
        encoded_features = [
            self.audio_encoder.encode(audio, sample_rate) for audio in separated_audios
        ]

        return encoded_features

    def classify(self, features: np.ndarray, min_samples: int = 3) -> Optional[int]:
        """
        特徴量から呪文を分類します。

        Args:
            features (np.ndarray): 分類対象の特徴量
            min_samples (int): 分類に必要な最小サンプル数

        Returns:
            Optional[int]: 分類された呪文ID。分類できない場合はNone
        """
        return self.sample_manager.get_nearest_spell(features, min_samples)

    def add_sample(self, spell_id: int, features: np.ndarray) -> None:
        """
        新しいサンプルを追加します。

        Args:
            spell_id (int): 呪文ID
            features (np.ndarray): 特徴量
        """
        self.sample_manager.add_sample(spell_id, features)

    def set_distance_threshold(self, threshold: float) -> None:
        """
        距離の閾値を設定します。

        Args:
            threshold (float): 新しい閾値
        """
        self.sample_manager.set_distance_threshold(threshold)

    def get_feature_dimension(self) -> int:
        """
        特徴量の次元数を取得します。

        Returns:
            int: 特徴量の次元数
        """
        if not self.initialized:
            raise RuntimeError(
                "Classifier is not initialized. Call initialize() first."
            )
        return self.audio_encoder.get_feature_dimension()
