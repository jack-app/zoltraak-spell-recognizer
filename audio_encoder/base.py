from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseAudioEncoder(ABC):
    """
    音声エンコーダーのベースクラス。
    すべての音声エンコーダーはこのクラスを継承する必要があります。
    """

    @abstractmethod
    def encode(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        音声データをエンコードします。

        Args:
            audio_data (np.ndarray): 入力音声データ
            sample_rate (int): サンプルレート

        Returns:
            np.ndarray: エンコードされた特徴量
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """
        エンコーダーを初期化します。
        モデルのロードやパラメータの設定などを行います。
        """
        pass

    @abstractmethod
    def get_feature_dimension(self) -> int:
        """
        エンコードされた特徴量の次元数を取得します。

        Returns:
            int: 特徴量の次元数
        """
        pass
