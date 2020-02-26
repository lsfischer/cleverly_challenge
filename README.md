# Cleverly challenge

This is my sumission to Cleverly's coding challenge. This repo contains all the files used and created during the development of this project.

## Repo structure

```
cleverly_challenge/
    ├── code
        ├── cleverly_challenge.ipynb ; main walkthrough notebook
        └── get_bert_embeddings.py ; Script used to get BERT embeddings
    ├── data
        ├── Clothing_Shoes_and_Jewelry_5.csv ; main dataset file
        ├── bert_encoded_summary.pkl ; Pickled dataframe containing BERT embeddings
        └── neural_net_model.h5 ; Trained neural network for bert embeddings
    └── requirements.txt ; library requirements to install
```

## Getting started

To get this notebook up and running first install all required libraries

```
pip install -r requirements.txt
```

Next make sure NLTK's stopwords and punkt files are downloaded (this is also done in the first cell of `cleverly_challenge.ipynb`)

```python
import nltk
nltk.download("stopwords")
nltk.download("punkt")
```

You now have all dependencies for this project and are able to run `cleverly_challenge.ipynb`

## DistilBERT Embeddings 

DistilBERT embeddings were also created for this project. They were created using [HuggingFace's DistilBERT from their transformers package](https://github.com/huggingface/transformers) and [Jay Allamar's](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/) was followed to create the embeddings. 

Due to RAM constraint issues given the size of the dataset and the memory needed by this model, the script used to create these embeddings was run on a Google Cloud VM with 406 Gb of RAM. Despite the large ammount of RAM in this machine it was still not possible to encode the entire dataset and so only a sample of 50% of the data was used. It is to note that only the `summary` feature was encoded given that `reviewText` was still to large to encode even just 10% of the dataset. A solution to this RAM constraint would be a point of improvement in the feature where different NLP models would be tested.

## Challenge questions

The answers to the questions asked for this challenge are all present in `cleverly_challenge.ipynb`.
During the development of this challenge the main question of interest was "Is looking at the summary of a review enough to find it helpful or do we have to read the entire review?" with the answer to this question also studied in `cleverly_challenge.ipynb`.