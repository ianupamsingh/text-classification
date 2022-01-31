import os
from typing import Union, List
from fastapi import FastAPI
import pydantic

from config import ExperimentConfig
from model import BERTClassifier

app = FastAPI()
classifier = None


class ClassifyRequest(pydantic.BaseModel):
    text: Union[str, List[str]]


class InitRequest(pydantic.BaseModel):
    config: Union[str, ExperimentConfig]


@app.post('/init_classifier')
def init_classifier(request: InitRequest):
    if type(request.config) == str:
        params_path = os.path.join(request.config, 'params.yaml')
        if not os.path.exists(params_path):
            return {"error": f"{params_path} does not exists"}
        try:
            config = ExperimentConfig.from_yaml(params_path)
        except AttributeError:
            return {"error": f"{params_path} is not a valid ExperimentConfig"}
    else:
        try:
            config = ExperimentConfig.from_json(request.config)
        except TypeError:
            return {"error": "Make sure `config` is either `str` or a valid `ExperimentConfig`"}

    global classifier
    classifier = BERTClassifier(config, training=False)

    return {'success': 'Initialized model'}


@app.post('/classify')
def classify(request: ClassifyRequest):
    if type(request.text) != str and type(request.text) != list:
        return {"error": "`text` should be string of list[string]"}

    global classifier
    if not classifier:
        return {"error": "Model is not initialized, please use /init_classifier API to initialize model"}

    predictions = classifier.predict(request.text)

    return {'class': predictions}

