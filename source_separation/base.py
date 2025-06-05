from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch


class BaseSourceSeparator(ABC):
    """
    音源分離のベースクラス。
    すべての音源分離アルゴリズムはこのクラスを継承する必要があります。
    """

    @abstractmethod
    def separate(self, audio_data: np.ndarray, sample_rate: int) -> List[torch.Tensor]:
        """
        音声データから音源を分離します。

        Args:
            audio_data (np.ndarray): 入力音声データ
            sample_rate (int): サンプルレート

        Returns:
            List[torch.Tensor]: 分離された音声データのリスト。
                最初の要素が目的の音源、残りの要素がその他の音源となります。
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """
        音源分離器を初期化します。
        モデルのロードやパラメータの設定などを行います。
        """
        pass

    @abstractmethod
    def set_device(self, device: torch.device) -> None:
        """
        エンコーダーを特定のデバイスに設定します。
        """
        pass
