import os
import yaml
from typing import Union, Optional, List

import numpy as np
import tensorflow as tf
import tensorflow_text
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow_hub as hub

from config import ExperimentConfig
from dataset import Dataset


class BERTClassifier:
    """BERT based Text classification model"""
    def __init__(self, config: Union[ExperimentConfig, str], training: Optional[bool] = True):
        if type(config) == str:
            self.config = ExperimentConfig.from_yaml(os.path.join(config, 'params.yaml'))
        else:
            self.config = config

        if training:
            self.model = None
            self.history = None
            if os.path.exists(self.config.output_dir) and os.listdir(self.config.output_dir):
                raise FileExistsError(f"Output directory '{self.config.output_dir}' already exists and is not empty.")
            if not os.path.exists(self.config.output_dir):
                os.makedirs(self.config.output_dir)
        else:
            if not os.path.exists(self.config.output_dir):
                raise FileNotFoundError(f"Model directory '{self.config.output_dir}' does not exists.")
            if not os.path.exists(os.path.join(self.config.output_dir, 'saved_model')):
                raise FileNotFoundError(f"SavedModel not found in directory '{self.config.output_dir}'.")
            self.model = tf.keras.models.load_model(os.path.join(self.config.output_dir, 'saved_model'))

    def _create_model(self) -> tf.keras.Model:
        """Creates and compiles `tf.keras.Model` that takes `text` input and outputs `logits`"""
        # Step 1: tokenize batches of text inputs.
        self.preprocessor = hub.load(self.config.preprocessor_model)
        text_input = [tf.keras.layers.Input(shape=(), dtype=tf.string, name='text'),
                      tf.keras.layers.Input(shape=(), dtype=tf.string, name='dummy')]
        tokenize = hub.KerasLayer(self.preprocessor.tokenize, name='tokenizer')
        tokenized_inputs = [tokenize(t) for t in text_input]

        # Step 2: pack input sequences for the Transformer encoder.
        bert_pack_inputs = hub.KerasLayer(self.preprocessor.bert_pack_inputs,
                                          arguments=dict(seq_length=self.config.max_len), name='packer')
        encoder_inputs = bert_pack_inputs(tokenized_inputs)

        input_word_ids = encoder_inputs['input_word_ids']
        input_mask = encoder_inputs['input_mask']
        input_type_ids = encoder_inputs['input_type_ids']

        # Step 3: pass `encoder_inputs` to `bert_layer`
        self.bert_layer = hub.KerasLayer(self.config.pretrained_model, trainable=True, name='bert')
        pooled_output, sequence_output = self.bert_layer([input_word_ids, input_mask, input_type_ids])

        # Step 4: Add dropout
        dropout = tf.keras.layers.Dropout(self.config.dropout, name='dropout')(pooled_output)

        # Step 5: Add classifier layer
        output = tf.keras.layers.Dense(len(self.config.labels), activation='softmax', name='output')(dropout)

        model = tf.keras.Model(inputs=text_input, outputs=output)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        return model

    def _get_data(self):
        dataset = Dataset(self.config.dataset, self.config.max_len)
        train_data, valid_data = dataset.generate()
        self.config.labels = list(dataset.label_encoder.classes_)
        return train_data, valid_data

    def train(self):
        """Start training model using given `ExperimentConfig`"""
        train_data, valid_data = self._get_data()

        # create model
        self.model = self._create_model()

        # save config
        yaml.dump(self.config.as_dict(), open(os.path.join(self.config.output_dir, 'params.yaml'), 'w'),
                  default_flow_style=False)

        # define callbacks
        checkpoint = ModelCheckpoint(os.path.join(self.config.output_dir, 'saved_model'),
                                     monitor='val_accuracy', save_best_only=True, verbose=1)
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

        # train
        self.history = self.model.fit(train_data, validation_data=valid_data, epochs=self.config.epochs,
                                      callbacks=[checkpoint, early_stopping], verbose=1)

    def predict(self, texts: Union[str, List[str]]) -> List[str]:
        """Predicts class of given text(s)
            Args:
                texts: str or list[str], text to predict classes for
            Ret:
                labels: list[str], predicted class texts belong to
        """
        if type(texts) == str:
            texts = [texts]
        dummy_text = [''] * len(texts)

        predictions = self.model.predict({'text': tf.constant(texts), 'dummy': tf.constant(dummy_text)})
        label_ids = np.argmax(predictions, axis=1)

        if not self.config.labels:
            raise ValueError(f'Labels not defined. Make sure `params.yaml` is present in `{self.config.output_dir}`'
                             f' and contains `labels` as not `None`')
        labels = [self.config.labels[label_id] for label_id in label_ids]
        return labels

