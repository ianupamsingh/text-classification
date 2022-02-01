import pytest
from dataset import Dataset


@pytest.mark.usefixtures('setup_dataset', 'config_dataset')
class TestDataset:
    def test_initialise(self, config_dataset):
        _ = Dataset(config_dataset)

    def test_labels(self, config_dataset):
        dataset = Dataset(config_dataset)
        assert len(dataset.label_encoder.classes_) == 2, "Labels encoded do not match"

    def test_split_data(self, config_dataset):
        dataset = Dataset(config_dataset)
        train_df, valid_df = dataset.split_data()
        assert len(train_df) == 160, "Train split does not match"
        assert len(valid_df) == 40, "Valid split does not match"

    def test_generate(self, config_dataset):
        dataset = Dataset(config_dataset)
        train_data, valid_data = dataset.generate()
        for data in train_data.take(1):
            assert len(data[0]['text'].numpy()) == config_dataset.train_batch_size, "Train batch size don't match"
            assert len(data[1].numpy()) == config_dataset.train_batch_size, "Label Train batch size don't match"
        for data in valid_data.take(1):
            assert len(data[0]['text'].numpy()) == config_dataset.valid_batch_size, "Valid batch size don't match"
            assert len(data[1].numpy()) == config_dataset.valid_batch_size, "Label Valid batch size don't match"
