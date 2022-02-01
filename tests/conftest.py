import pytest
import os
import shutil
import pandas as pd

from config import DataConfig, ExperimentConfig


@pytest.fixture(scope='class')
def setup_model():
    if os.path.exists('test_model'):
        shutil.rmtree('test_model')
    os.mkdir('test_model')
    yield
    shutil.rmtree('test_model')


@pytest.fixture(scope='class')
def config_model():
    config = ExperimentConfig(dataset=DataConfig(input_path='test_data/sample.csv'))
    return config


@pytest.fixture(scope='class')
def setup_dataset():
    if os.path.exists('test_data'):
        shutil.rmtree('test_data')
    os.mkdir('test_data')
    temp = [['sample text', 'class1'], ['sample text', 'class2']] * 100
    data = pd.DataFrame(temp, columns=['text', 'category'])
    data.to_csv('test_data/sample.csv')
    yield
    shutil.rmtree('test_data')


@pytest.fixture(scope='class')
def config_dataset():
    config = DataConfig(input_path='test_data/sample.csv')
    return config
