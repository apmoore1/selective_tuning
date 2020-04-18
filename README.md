# Selective Tuning

## Requirements

1. Python > 3.6.1
2. `pip install -r requirements.txt`

## Data

In the [data](./data) directory is the following datasets:
1. [sst](./data/sst) - which is the Stanford Sentiment Treebank. Which has be downloaded from [this link.](https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip). Number of train, dev, and test instances; 8544, 1101, and 2210 respectively.

## Tasks

1. Fine grained sentence level sentiment analysis using the Stanford Sentiment Treebank with 5 sentiment labels.

## Models
In the [model_configs directory](./model_configs) there are two models:
1. [Standard word embedding model](./model_configs/word_embedding.jsonnet)
2. [BERT model](./model_configs/bert.jsonnet)

These model files use a batch size of 1 and to ensure that the optimiser updates every **N** we set the **num_gradient_accumulation_steps** to **N**.

## Running these models with VSCode in debug mode

If you want to run these models in VSCode in debug model use the [model_test.py](./model_test.py) file and put your break point on any line in the [selective_tuning/allen/training/modified_trainer.py](./selective_tuning/allen/training/modified_trainer.py) file.

The [modified_trainer.py](./selective_tuning/allen/training/modified_trainer.py) is the file that contains the code that stores the gradients for each sample within a batch on lines 367 to 413.