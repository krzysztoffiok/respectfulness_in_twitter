import os
import argparse
import sys
import flair
import torch

"""
Example use:
# remember to start by setting a proper --dependent_variable.

# to train FastText model with 2-layer biLSTM
python3 model_train.py --dependent_variable=respect --folds=5 --test_run=fasttext

# to train RoBERTa large scalar mix of 4 last layers model with 2-layer biLSTM
python3 model_train.py --dependent_variable=respect --folds=5 --test_run=roberta_lstm

# to fine-tune a transformer model e.g. RoBERTa large model and use CLS token output
python3 model_train.py --dependent_variable=respect --folds=5 --test_run=roberta-large --fine_tune

"""

parser = argparse.ArgumentParser(description='Classify data')

parser.add_argument('--folds', required=False, type=int, default=1)
parser.add_argument('--epochs', required=False, type=int, default=1000)
parser.add_argument('--block_print', required=False, default=False,
                    action='store_true', help='Block printable output')
parser.add_argument('--fine_tune', required=False, default=False,
                    action='store_true', help='Fine tune pre-trained LM')
parser.add_argument('--test_run', required=True, type=str, default='fasttext')
parser.add_argument('--dependent_variable', required=False, type=str, default='respect')
parser.add_argument('--mbs', required=False, type=int, default=8)
parser.add_argument("--filepath", required=False, default='input_file.xlsx', type=str)

args = parser.parse_args()
filepath = os.path.join('./data/source_data/', args.filepath)
data_file_name = args.filepath.split('.')[0]
folds = args.folds
epochs = args.epochs
test_run = args.test_run
path = list()
dependent_variable = args.dependent_variable
fine_tune = args.fine_tune
mbs = args.mbs

# disable printing out
block_print = args.block_print

if block_print:
    sys.stdout = open(os.devnull, 'w')

# prepare paths for loading data for k fold training
for i in range(folds):
    path.append(f"./data/models/{dependent_variable}_{data_file_name}_{str(i)}/")
    try:
        os.mkdir(f'./data/models/{dependent_variable}_{data_file_name}_{str(i)}')
    except FileExistsError:
        continue

# k folds training
for i in range(folds):
    # imports should be done repeatedly because of not investigated Flair issue
    import flair.datasets
    from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings, TransformerDocumentEmbeddings,\
        TransformerWordEmbeddings
    from flair.models import TextClassifier
    from flair.trainers import ModelTrainer
    from pathlib import Path
    import torch
    from flair.data import FlairDataset

    # load training data
    corpus: flair.data.Corpus = flair.datasets.ClassificationCorpus(Path(os.path.join(path[i])),
                                                                    test_file='test_.tsv',
                                                                    dev_file='dev.tsv',
                                                                    train_file='train.tsv')

    if test_run == "fasttext":
        word_embeddings = [WordEmbeddings('en-twitter')]
    elif test_run == "roberta_lstm":
        # from unknown reasons on google colaboratory scalar mix option causes issues with torch.device selection
        # if you run locally, please add: ,pooling_operation="first", use_scalar_mix=True)
        word_embeddings = [TransformerWordEmbeddings(model="roberta-large", layers="21,22,23,24", batch_size=6)]

    # Case 1: Use selected word_embeddings model fed to an RNN
    if not fine_tune:
        # define the RNN model
        document_embeddings = DocumentRNNEmbeddings(word_embeddings,
                                                    hidden_size=512,
                                                    reproject_words=True,
                                                    reproject_words_dimension=256,
                                                    rnn_type="LSTM",
                                                    bidirectional=True,
                                                    rnn_layers=2,
                                                    )

    # Case 2: fine-tune transformer model and use CLS output
    else:
        transformer_model = test_run
        if "/" in test_run:
            test_run = test_run.split("/")[1]
        document_embeddings = TransformerDocumentEmbeddings(model=transformer_model, fine_tune=True)

    # define the neural classifier
    classifier = TextClassifier(document_embeddings,
                                label_dictionary=corpus.make_label_dictionary(),
                                multi_label=False)

    # define the training regime for model+RNN
    if not fine_tune:
        # train model
        trainer = ModelTrainer(classifier, corpus)

        trainer.train(base_path="{}".format(path[i]),
                      max_epochs=epochs,
                      learning_rate=0.1,
                      mini_batch_size=8,
                      anneal_factor=0.5,
                      patience=20,
                      # if enough memory is available this can be uncommented
                      # embeddings_storage_mode='gpu',
                      shuffle=True,
                      min_learning_rate=0.002,
                      )

    # define the training regime for Fine-Tuning
    else:
        # train model
        trainer = ModelTrainer(classifier, corpus, optimizer=torch.optim.Adam)

        trainer.train(
            "{}".format(path[i]),
            learning_rate=3e-6,  # low learning rate as per BERT paper
            mini_batch_size=int(mbs),  # set this high if yo have lots of data, otherwise low
            max_epochs=4,  # very few epochs of fine-tuning
            min_learning_rate=3e-6,  # lower the min learning rate
            # if enough memory is available this can be uncommented
            # embeddings_storage_mode='gpu',
            shuffle=True,
        )

    # rename the model files to fit test_run and fold
    os.rename(src=f"{path[i]}best-model.pt", dst=f"{path[i]}{test_run}_best-model.pt")
    os.remove(f"{path[i]}final-model.pt")
    os.rename(src=f"{path[i]}training.log", dst=f"{path[i]}{test_run}_training_{i}.log")
    os.remove(f"{path[i]}weights.txt")
    os.rename(src=f"{path[i]}loss.tsv", dst=f"{path[i]}{test_run}_{i}_loss.tsv")
