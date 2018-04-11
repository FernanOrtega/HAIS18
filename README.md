# HAIS18
Python project with the proposal's implementation from the paper: 

> **A Hybrid Approach to Mining Conditions** - HAIS, 2018 - *Fernando O. Gallego and Rafael Corchuelo*

## Repository contents
- datasets/ *Dataset's folder*
  * dataset-en
  * dataset-en-lite
  * dataset-es
  * dataset-es-lite
- models/ *Word2vec models' folder*
  * w2v-modelv2-en
  * w2v-modelv2-en
- LICENCE
- candidates_creator.py
- main.py
- model_factory.py
- README
- validation.py
- word_preprocessing.py
- word_vectorizer.py


## Requirements
- Python 3.5.4 or above
- Theano 0.9.0
- Keras 2.0.8
- NLTK 3.2.4 with _punkt_ and _SnowballData_ models installed.
- Numpy 1.13.1
- Scikit-learn 0.19.0
- Gensim 2.3.0


## Usage
main.py is the entry point of our experiments. It contains the following script parameters:
1. relative path of the dataset's file
2. language selected
3. relative path of the word2vec model's file
4. number of folds to perform k-fold cross validation
5. deep learning model to use in the experiment (name of the class inside model_factory.py)
6. relative path of the output csv file with the performance results
7. score threshold to consider whether a candidate is a condition or not

### Example of use:
```
python main.py dataset/dataset-en en models/w2v-modelv2-en 4 ModelA results/results-ModelA-en.csv 0.75
```
