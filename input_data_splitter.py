import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os
import argparse

"""
Splits input data into n folds for Flair model training and later ML.
Example use:

python3 input_data_splitter.py --folds=5 --dependent_variable=respect
"""

parser = argparse.ArgumentParser(description='split_input_data')
parser.add_argument("--dependent_variable", required=False, default="respect", type=str)
parser.add_argument("--folds", required=False, default=5, type=int)
args = parser.parse_args()

dependent_variable = args.dependent_variable
folds = args.folds

# read original data file
df = pd.read_excel("./data/source_data/input_file.xlsx", converters={'dummy_id': str})

# use only selected columns
df = df[[dependent_variable, "text"]]
# change format of 'sentiment' label for further training in Flair framework
df[dependent_variable] = '__label__' + df[dependent_variable].astype(str)

# Cross validation
# setup random state and folds
np.random.seed(13)
kf = KFold(n_splits=folds, random_state=13, shuffle=True)

# create data splits for Deep Learning Language Models trained with Flair framework
train_indexes = dict()
val_indexes = dict()
test_indexes = dict()

# train sets for Machine Learning
train_ml = dict()
i = 0

# this split (with folds=5) results in: 20% test, 10% val, 70% train for Flair framework
# and the same 20% test and 80 % train for Machine Learning
indexes = list(range(0, len(df)))
for train_index, test_index in kf.split(indexes):
    test_indexes[i] = test_index
    train_ml[i] = train_index
    train_index, val_index = train_test_split(train_index, test_size=0.125, random_state=13, shuffle=True)
    train_indexes[i] = train_index
    val_indexes[i] = val_index
    i += 1
    
# test sets for Machine Learning are equal to those for Flair framework
test_ml = test_indexes

# create folders for FLAIR data splits and .tsv files for training
folds_path = list()
for fold in range(folds):
    folds_path.append(f'./data/model_{dependent_variable}_{str(fold)}/')
    try:
        os.mkdir(f'./data/model_{dependent_variable}_{str(fold)}')
    except FileExistsError:
        None  # continue
    df.iloc[test_indexes[fold]].to_csv(os.path.join(folds_path[fold], "test_.tsv"),
                                       index=False, header=False, encoding='utf-8', sep='\t')
    df.iloc[train_indexes[fold]].to_csv(os.path.join(folds_path[fold], "train.tsv"),
                                        index=False, header=False, encoding='utf-8', sep='\t')
    df.iloc[val_indexes[fold]].to_csv(os.path.join(folds_path[fold], "dev.tsv"),
                                      index=False, header=False, encoding='utf-8', sep='\t')
