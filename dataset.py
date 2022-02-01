from typing import Optional

from config import DataConfig
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf


class Dataset:
    """Reads pd.Dataframe to create tf.data.Dataset"""
    def __init__(self, config: DataConfig, max_len: Optional[int] = 512):
        self.config = config
        self.max_len = max_len
        self.df = pd.read_csv(self.config.input_path)
        if self.config.x_column not in self.df.keys() or self.config.y_column not in self.df.keys():
            raise ValueError(f'Make sure columns `{self.config.x_column}` and `{self.config.y_column}` are present in df')
        self.label_encoder = None
        self.encode_labels()

    def generate(self):
        """Generate `train` and `valid` datasets from given `pd.Dataframe`"""
        train_df, valid_df = self.split_data()

        train_df.rename(columns={self.config.x_column: 'text'}, inplace=True)
        valid_df.rename(columns={self.config.x_column: 'text'}, inplace=True)
        train_df['dummy'] = ''
        valid_df['dummy'] = ''
        train_data = tf.data.Dataset.from_tensor_slices((dict(train_df[['text', 'dummy']]), train_df['label_id'].values))
        valid_data = tf.data.Dataset.from_tensor_slices((dict(valid_df[['text', 'dummy']]), valid_df['label_id'].values))

        train_data = train_data.shuffle(10000).batch(self.config.train_batch_size, drop_remainder=True)\
            .prefetch(tf.data.experimental.AUTOTUNE)

        valid_data = valid_data.batch(self.config.valid_batch_size, drop_remainder=True)\
            .prefetch(tf.data.experimental.AUTOTUNE)

        return train_data, valid_data

    def encode_labels(self):
        """Creates label encoding from label classes present in `y_column`"""
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.df[self.config.y_column])
        self.df['label_id'] = self.label_encoder.transform(self.df[self.config.y_column])

    def split_data(self):
        """Stratified splitting to create `train` and `valid` Dataframes"""
        split = StratifiedShuffleSplit(n_splits=1, test_size=self.config.split_ratio, random_state=self.config.seed)
        train_df, valid_df = None, None
        for train_index, test_index in split.split(self.df, self.df[self.config.y_column]):
            train_df = self.df.loc[train_index]
            valid_df = self.df.loc[test_index]
        return train_df, valid_df
