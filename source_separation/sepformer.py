import torch
import torchaudio.functional as F
from speechbrain.inference.separation import SepformerSeparation as separator

from source_separation.base import BaseSourceSeparator


class SepformerSeparator(BaseSourceSeparator):
    """
    Sepformerモデルを使用した音源分離器。
    入力音声を複数の音源に分離します。
    """

    def __init__(self, model_source: str = "speechbrain/sepformer-libri3mix"):
        """
        分離器を初期化します。

        Args:
            model_source (str): 使用するSepformerモデルのソース
        """
        super().__init__()
        self.model_source = model_source
        self.separator = None
        self.initialized = False
        self.initialize()
        self.output_tensor_device = torch.device("cpu")

    def initialize(self) -> None:
        """
        分離器を初期化します。
        """
        self.separator = separator.from_hparams(source=self.model_source)
        self.initialized = True

    def set_device(self, device: torch.device) -> None:
        """
        分離器を特定のデバイスに設定します。
        """
        self.output_tensor_device = device
        self.separator = separator.from_hparams(
            source=self.model_source, run_opts={"device": str(device)}
        )

    def separate(
        self, audio_data: torch.Tensor, sample_rate: int
    ) -> list[torch.Tensor]:
        """
        音声データを複数の音源に分離します。

        Args:
            audio_data (torch.Tensor): 入力音声データ
            sample_rate (int): サンプルレート

        Returns:
            list[torch.Tensor]: 分離された音源のリスト
        """
        if not self.initialized:
            raise RuntimeError("Separator is not initialized. Call initialize() first.")

        # 音声データの形状を調整
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)  # (1, length)

        # 音源分離
        audio_data = audio_data.to("cpu")
        with torch.no_grad():
            separated = self.separator.separate_batch(audio_data)[0].T  # (3, length)

        return separated.to(self.output_tensor_device)
