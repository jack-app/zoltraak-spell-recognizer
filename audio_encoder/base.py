from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch


class BaseAudioEncoder(ABC):
    """
    音声エンコーダーのベースクラス。
    すべての音声エンコーダーはこのクラスを継承する必要があります。
    """

    @abstractmethod
    def encode(self, audio_data: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        音声データをエンコードします。

        Args:
            audio_data (torch.Tensor): 入力音声データ
            sample_rate (int): サンプルレート

        Returns:
            torch.Tensor: エンコードされた特徴量
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
