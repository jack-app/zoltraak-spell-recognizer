import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModel

from audio_encoder.base import BaseAudioEncoder


class HuBERTEncoder(BaseAudioEncoder):
    """
    HuBERTモデルを使用した音声エンコーダー。
    音声データを固定長の特徴量ベクトルに変換します。
    """

    def __init__(
        self,
        model_name: str = "rinna/japanese-hubert-base",
        max_length: int = 6000,  # 0.75秒
    ):
        """
        エンコーダーを初期化します。

        Args:
            model_name (str): 使用するHuBERTモデルの名前
            max_length (int): 入力音声の最大長（サンプル数）
        """
        self.model_name = model_name
        self.max_length = max_length
        self.feature_extractor = None
        self.model = None
        self.initialized = False
        self.initialize()

    def initialize(self) -> None:
        """
        モデルと特徴量抽出器を初期化します。
        """
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.initialized = True

    def encode(self, audio_data: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        音声データを特徴量ベクトルに変換します。

        Args:
            audio_data (torch.Tensor): 入力音声データ
            sample_rate (int): サンプルレート

        Returns:
            torch.Tensor: 特徴量ベクトル
        """
        if not self.initialized:
            raise RuntimeError("Encoder is not initialized. Call initialize() first.")

        # 音声データの長さを調整
        if len(audio_data) > self.max_length:
            audio_data = audio_data[: self.max_length]
        elif len(audio_data) < self.max_length:
            audio_data = torch.nn.functional.pad(
                audio_data,
                (0, self.max_length - len(audio_data)),
                mode="constant",
                value=0,
            )

        # サンプリングレートを16000Hzに変更
        # if sample_rate != 16000:
        #     audio_data = librosa.resample(
        #         audio_data.numpy(),
        #         orig_sr=sample_rate,
        #         target_sr=16000,
        #     )
        #     sample_rate = 16000
        #     inputs = torch.tensor(audio_data)

        # 特徴量抽出
        inputs = self.feature_extractor(
            audio_data.numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        # モデルでエンコード
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.squeeze(0)  # (1, seq_len, hidden_dim)
            embedding = embedding.reshape(-1)  # (seq_len * hidden_dim,)

        return embedding

    def get_feature_dimension(self) -> int:
        """
        特徴量の次元数を取得します。

        Returns:
            int: 特徴量の次元数
        """
        if not self.initialized:
            raise RuntimeError("Encoder is not initialized. Call initialize() first.")
        # 160はHuBERTのstride
        return self.model.config.hidden_size * self.max_length // 160
