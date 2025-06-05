import os
import warnings
from typing import Dict, List, Literal, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from audio_encoder.base import BaseAudioEncoder
from source_separation.base import BaseSourceSeparator
from utils.audio import remove_silence


def get_device() -> torch.device:
    """
    利用可能な最適なデバイスを取得します。
    CUDA > Metal > CPU の優先順位で選択します。

    Returns:
        torch.device: 使用するデバイス
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    ユークリッド距離を計算します。

    Args:
        x (np.ndarray): ベクトル1, shape=(sample_size_x, feature_dim)
        y (np.ndarray): ベクトル2, shape=(sample_size_y, feature_dim)

    Returns:
        float: ユークリッド距離の行列. shape=(sample_size_x, sample_size_y)
    """
    x = torch.from_numpy(x).to(get_device())
    y = torch.from_numpy(y).to(get_device())
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    return torch.linalg.norm(x - y, dim=-1).cpu().numpy()


def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    コサイン距離を計算します。

    Args:
        x (np.ndarray): ベクトル1, shape=(sample_size_x, feature_dim)
        y (np.ndarray): ベクトル2, shape=(sample_size_y, feature_dim)

    Returns:
        np.ndarray: コサイン距離の行列. shape=(sample_size_x, sample_size_y)
    """
    x = torch.from_numpy(x).to(get_device())
    y = torch.from_numpy(y).to(get_device())
    x_norm = torch.linalg.norm(x, dim=1, keepdim=True)
    y_norm = torch.linalg.norm(y, dim=1, keepdim=True)
    return (1 - torch.mm(x, y.t()) / (x_norm * y_norm.t())).cpu().numpy()


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    マンハッタン距離を計算します。

    Args:
        x (np.ndarray): ベクトル1, shape=(sample_size_x, feature_dim)
        y (np.ndarray): ベクトル2, shape=(sample_size_y, feature_dim)

    Returns:
        np.ndarray: マンハッタン距離の行列. shape=(sample_size_x, sample_size_y)
    """
    x = torch.from_numpy(x).to(get_device())
    y = torch.from_numpy(y).to(get_device())
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    return torch.sum(torch.abs(x - y), dim=-1).cpu().numpy()


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
    呪文分類器クラス。
    音源分離と特徴量抽出を行い、アンカーサンプルとの距離に基づいて呪文を分類します。
    """

    DISTANCE_FUNCTIONS = {
        "euclidean": euclidean_distance,
        "cosine": cosine_distance,
        "manhattan": manhattan_distance,
    }

    def __init__(
        self,
        source_separator: BaseSourceSeparator,
        audio_encoder: BaseAudioEncoder,
        distance_type: Literal["euclidean", "cosine", "manhattan"] = "euclidean",
        distance_threshold: float = 0.5,
    ):
        """
        呪文分類器を初期化します。

        Args:
            source_separator (BaseSourceSeparator): 音源分離器
            audio_encoder (BaseAudioEncoder): 音声エンコーダー
            distance_type (Literal["euclidean", "cosine", "manhattan"]): 使用する距離関数
            distance_threshold (float): 距離の閾値
        """
        self.samples: Dict[int, List[np.ndarray]] = {}
        self.distance_threshold = distance_threshold
        self.distance_func = self.DISTANCE_FUNCTIONS[distance_type]
        self.source_separator = source_separator
        self.audio_encoder = audio_encoder
        self.distance_threshold = distance_threshold
        self.samples: Dict[int, List[np.ndarray]] = {}
        self.device = get_device()
        print(f"Using device: {self.device}")

        self.initialize()

        self.load_anchor_samples("anchors")

    def initialize(self) -> None:
        """
        分類器を初期化します。
        音源分離器と音声エンコーダーを初期化します。
        """
        self.source_separator.initialize()
        self.audio_encoder.initialize()
        # デバイスを設定
        self.source_separator.set_device(self.device)
        self.audio_encoder.set_device(self.device)

    def add_sample(self, class_id: int, features: np.ndarray) -> None:
        """
        サンプルを追加します。

        Args:
            class_id (int): クラスID
            features (np.ndarray): 特徴量
        """
        if class_id not in self.samples:
            self.samples[class_id] = []
        self.samples[class_id].append(features)

    def get_feature_dimension(self) -> int:
        """
        特徴量の次元数を取得します。

        Returns:
            int: 特徴量の次元数
        """
        return self.audio_encoder.get_feature_dimension()

    def load_anchor_samples(self, anchors_dir: str) -> None:
        """
        アンカーサンプルを読み込みます。

        Args:
            anchors_dir (str): アンカーサンプルのディレクトリパス
        """
        if not os.path.exists(anchors_dir):
            raise FileNotFoundError(
                f"アンカーディレクトリが見つかりません: {anchors_dir}"
            )

        # クラスIDごとのディレクトリを処理
        for class_id in tqdm(os.listdir(anchors_dir), desc="Loading anchor classes"):
            class_dir = os.path.join(anchors_dir, class_id)
            if not os.path.isdir(class_dir):
                continue

            try:
                class_id_int = int(class_id)
            except ValueError:
                warnings.warn(f"無効なクラスIDディレクトリ: {class_id}")
                continue

            # WAVファイルを処理
            wav_files = [f for f in os.listdir(class_dir) if f.endswith(".wav")]
            for wav_file in tqdm(wav_files, desc=f"Loading class {class_id}"):
                try:
                    file_path = os.path.join(class_dir, wav_file)
                    y, sr = librosa.load(file_path)
                    y = remove_silence(y, sr=sr)

                    # 音声データをtorch.Tensorに変換
                    audio_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

                    # 音源分離と特徴量抽出
                    print("Processing audio...")
                    features_list = self.process_audio(audio_tensor, sr)

                    # 各音源の特徴量を保存
                    for i, features in enumerate(features_list):
                        self.add_sample(class_id_int, features)

                except Exception as e:
                    raise e
                    continue

    def process_audio(self, audio: torch.Tensor, sr: int) -> List[np.ndarray]:
        """
        音声データを処理します。
        音源分離と特徴量抽出を行います。

        Args:
            audio (torch.Tensor): 入力音声データ
            sr (int): サンプリングレート

        Returns:
            List[np.ndarray]: 各音源の特徴量のリスト
        """
        # 入力音声を適切なデバイスに移動
        audio = audio.to(self.device)

        # 音源分離
        separated_sources = self.source_separator.separate(audio, sr)

        # 各音源の特徴量を抽出
        features_list = []
        for source in separated_sources:
            features = self.audio_encoder.encode(source, sr)
            features_list.append(features.cpu().numpy())

        return features_list

    def classify(self, audio_data: torch.Tensor, sr: int) -> Optional[int]:
        """
        特徴量を分類します。

        Args:
            audio_data (torch.Tensor): 分類する音声, shape=(length,)
            sr (int): サンプリングレート

        Returns:
            Optional[int]: 分類結果のクラスID。分類できない場合はNone。
        """
        if not self.samples:
            return None

        # 入力音声を適切なデバイスに移動
        audio_data = audio_data.to(self.device)

        features = self.process_audio(audio_data, sr)
        features = np.array(features)  # shape=(num_sources, feature_dim)

        min_distance = float("inf")
        best_class_id = None

        for class_id, class_samples in self.samples.items():
            # クラス内の全サンプルとの距離を一度に計算
            class_samples_array = np.array(
                class_samples
            )  # shape=(num_samples, feature_dim)
            distances = self.distance_func(
                features, class_samples_array
            )  # shape=(num_sources, num_samples)

            # 各音源について最小距離を取得
            min_distances = np.min(distances, axis=1)  # shape=(num_sources,)
            class_min_distance = np.min(min_distances)  # 全音源の中での最小距離

            if class_min_distance < min_distance:
                min_distance = class_min_distance
                best_class_id = class_id

        # 閾値以下の距離の場合のみ分類結果を返す
        if min_distance <= self.distance_threshold:
            return best_class_id

        return None
