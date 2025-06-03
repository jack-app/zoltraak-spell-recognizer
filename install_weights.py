from speechbrain.inference.separation import SepformerSeparation as separator
from transformers import AutoFeatureExtractor, AutoModel

if __name__ == "__main__":
    print("loading model weights...")
    model_name: str = "rinna/japanese-hubert-base"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    separator = separator.from_hparams(
        source="speechbrain/sepformer-libri3mix",
    )

    print("succesfully loaded model weights")
