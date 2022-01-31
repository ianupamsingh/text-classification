# Text Classiication using BERT

## Installation
Install required packages:
>`pip install -r requirement.txt`

Activate environement
>`source path/to/pyenv/bin/activate`

## Training
Training the model requires you to define `config.ExpermientConfig`, 
which can be done using the config.py or instantiating `config.ExpermientConfig` `from_json` or `from_yaml`

The default config is already configured if you want to test run.

After defining config:
> `python main.py`

## Serve as an API
The model can be served as an API to make inferences after training.
- Run server
  - `python serve.py`
  - This will expose following API endpoints on `http://localhost:8000`
    - POST `/init_classifier`
      ```
      Request body: 
      {
          "config": "string path of trained model or ExperimentConfig"
      }
      ```
    - POST `/classifiy`
      ```
      Request body:
      {
            "text": "string or list(string)" 
      }
      ```
