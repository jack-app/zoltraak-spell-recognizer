from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseSourceSeparator(ABC):
    """
    音源分離のベースクラス。
    すべての音源分離アルゴリズムはこのクラスを継承する必要があります。
    """

    @abstractmethod
    def separate(self, audio_data: np.ndarray, sample_rate: int) -> List[np.ndarray]:
        """
        音声データから音源を分離します。

        Args:
            audio_data (np.ndarray): 入力音声データ
            sample_rate (int): サンプルレート

        Returns:
            List[np.ndarray]: 分離された音声データのリスト。
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
