import dataclasses

from official.modeling.hyperparams.base_config import Config
from typing import Optional, List


@dataclasses.dataclass
class DataConfig(Config):
    x_column: str = 'text'
    y_column: str = 'category'
    train_batch_size: int = 8
    valid_batch_size: int = 8
    input_path: str = 'data/sample.csv'
    seed: int = 42
    split_ratio: float = 0.2


@dataclasses.dataclass
class ExperimentConfig(Config):
    preprocessor_model: str = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    pretrained_model: str = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
    output_dir: str = 'model'
    labels: Optional[List[str]] = None
    max_len: int = 512
    learning_rate: float = 2e-5
    epochs: int = 1
    optimizer: str = 'adam'
    dropout: float = 0.2
    dataset: DataConfig = DataConfig()
