# Nuclear-public-opinion-hierarchical-classification
This is the implementation of the paper "Position Analysis of Internet Public Opinions - 
A Case Study of Nuclear Energy Policy in Taiwan".


## Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weiji14/deepbedmap/]


## Environment Setup
```
virtualenv -p python3 venv
source ./venv/bin/activate
pip install -r requirements.txt
```


## Usage

Train model
```
python run_bert_base.py
```
Evaluate model
```
python evaluate.py
```
