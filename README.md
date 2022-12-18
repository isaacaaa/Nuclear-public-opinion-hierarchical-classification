# Nuclear-public-opinion-hierarchical-classification
This is the implementation of the paper [A Teacher-Student Approach to Cross-Domain Transfer Learning with Multi-level Attention](https://dl.acm.org/doi/abs/10.1145/3486622.3494009) .


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
