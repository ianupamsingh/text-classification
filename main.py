import os
from typing import Optional

from config import ExperimentConfig, DataConfig
from model import BERTClassifier


def get_experiment_config(configuration: Optional[str] = None) -> ExperimentConfig:
    configuration_ = None
    if type(configuration) == str:
        params_path = os.path.join(configuration, 'params.yaml')
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"{params_path} does not exists")
        try:
            configuration_ = ExperimentConfig.from_yaml(params_path)
        except AttributeError:
            raise ValueError(f"{params_path} is not valid")
    else:
        # return default
        configuration_ = ExperimentConfig(
            preprocessor_model='https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            pretrained_model='https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2',
            output_dir='model',
            labels=None,
            max_len=512,
            learning_rate=2e-5,
            epochs=1,
            optimizer='adam',
            dropout=0.2,
            dataset=DataConfig(
                x_column='text',
                y_column='category',
                train_batch_size=8,
                valid_batch_size=8,
                input_path='data/sample.csv',
                seed=42,
                split_ratio=0.2
            ))
    return configuration_


if __name__ == '__main__':
    config = get_experiment_config()
    # train
    classifier = BERTClassifier(config)
    classifier.train()

