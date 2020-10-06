import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
import sklearn
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
import random
import datatable as dt
import argparse
import _utils
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='Classify data')
parser.add_argument('--fold', required=False, type=int, default=0)
parser.add_argument('--folds', required=False, type=int, default=5)

parser.add_argument('--estimators', required=False, type=int, default=250,
                    help="number of estimators in machine learning classification models")
parser.add_argument('--test_run', required=True, type=str, default="", help="model name")
parser.add_argument('--dependent_variable', required=True, type=str, default="respect",
                    help="dependent_variable e.g. respect")
parser.add_argument('--pooled', required=False, default=False, action='store_true')
parser.add_argument("--filepath", required=False, default='input_file.xlsx', type=str)

args = parser.parse_args()
filepath = os.path.join('./data/source_data/', args.filepath)
data_file_name = args.filepath.split('.')[0]
folds = args.folds
_fold = args.fold

_estimators = args.estimators
test_run = args.test_run
dependent_variable = args.dependent_variable
pooled = args.pooled


# read original data file
df = pd.read_excel(filepath, converters={'dummy_id': str})

# use only selected columns
df = df[[dependent_variable, "text"]]

# Cross validation
# setup random state
np.random.seed(13)

# create data splits for Deep Learning Language Models trained with Flair framework
train_indexes = dict()
test_indexes = dict()

# train sets for Machine Learning
train_ml = dict()

# define number of folds
kf = _utils.splitter(folds=folds, df=df)

i = 0

# this split (with folds=5) results in: 20% test, 10% val, 70% train for Flair framework
# and the same 20% test and 80 % train for Machine Learning
indexes = list(range(0, len(df)))
for train_index, test_index in kf:
    test_indexes[i] = test_index
    train_ml[i] = train_index
    i += 1

# test sets for Machine Learning are equal to those for Flair framework
test_ml = test_indexes


# <h3> Vector representations (embeddings) created by selected Deep Learning Language Models trained previously
# on here addressed task
# define which embedding files to read

model_list = [test_run]
embeddings = [(x, x) for x in model_list]

# instantiate list of data frames with features and a list of feature names for each df
dfemblist = list()

# Initialize a dictionary with all features used later on in Machine Learning
allFeatures = dict()

df_ml = pd.DataFrame()

# read embedding files and define corresponding feature names (lists of names)
if "TF" not in model_list:
    for emname, embedding in embeddings:
        embfeaturedict = dict()
        for fold in range(folds):
            # embeddings from Flair
            if model_list[0] != "LIWC" and model_list[0] != "SEANCE":
                # read encoded sentences by the selected language model
                if pooled:
                    dfemb = dt.fread(
                        f"./data/embeddings/{dependent_variable}"
                        f"_{data_file_name}_{embedding}_encoded_sentences_pooled.csv").to_pandas()
                else:
                    dfemb = dt.fread(
                        f"./data/embeddings/{dependent_variable}"
                        f"_{data_file_name}_{embedding}_encoded_sentences_{fold}.csv").to_pandas()

                # define number of feature columns (columns - 3)
                number_of_feature_columns = len(dfemb.columns) - 3

                embfeatures = [f"{emname}{fold}row"]

                # create unique feature (column) names
                embfeatures.extend([f"{emname}{fold}{x}" for x in range(number_of_feature_columns)])
                embfeatures.extend([f"{emname}{fold}_category_", f"{emname}{fold}_dummy_id_"])
                dfemb.columns = embfeatures

                # append features from each language model in tuple ((model_name,fold), [features])
                embfeaturedict[fold] = [f"{emname}{fold}{x}" for x in range(number_of_feature_columns)]
            # embeddings from LIWC or SEANCE
            elif "LIWC" in model_list or "SEANCE" in model_list:
                if "LIWC" in model_list:
                    dfemb, embfeatures = _utils.read_liwc()
                elif "SEANCE" in model_list:
                    dfemb, embfeatures = _utils.read_seance()

                embfeatures = [f"{x}{fold}" for x in embfeatures]
                dfemb.columns = embfeatures
                embfeaturedict[fold] = embfeatures

            # append encoded sentences by the selected language model to a list of data frames
            dfemblist.append(dfemb)
        # create entry in dictionary with all features for each trained language model
        allFeatures[emname] = embfeaturedict

    # concat all Data Frames: liwc, TF, DL embedding into one df_ml that will be used in Machine Learning
    for dfemb in dfemblist:
        df_ml = pd.concat([df_ml, dfemb], axis=1)

# Term Frequency model
elif "TF" in model_list:
    dftf, allFeatures = _utils.term_frequency(train_ml, df, allFeatures, dependent_variable)
    df_ml = dftf

# define the target variable in the final df_ml data frame
df_ml["target_ml"] = df[dependent_variable]

# <h1> Machine Learning part
# Define list of model names
all_language_models = [x for x in model_list]

# instantiate dictionary for data frames with results
allPreds = {}
allTrues = {}
allTexts = {}

# define which classification models to use
models = [xgb.XGBClassifier(objective='multi:softprob', n_jobs=24, learning_rate=0.03,
                            max_depth=10, subsample=0.7, colsample_bytree=0.6,
                            random_state=2020, n_estimators=_estimators)]  # optionally add: tree_method='gpu_hist'

if pooled:
    pooled_text = 'pooled'
else:
    pooled_text = ''

# use features from selected language models
for language_model in all_language_models:
    # for training of selected classification models
    for classification_model in models:
        preds, trues = _utils.ML_classification(allFeatures=allFeatures, train_ml=train_ml, test_ml=test_ml,
                                                df_ml=df_ml, classification_model=classification_model,
                                                language_model=language_model, folds=folds,
                                                pooled_text=pooled_text)

        # save model predictions
        allPreds[f"{language_model}{pooled_text}_{type(classification_model).__name__}"] = preds.copy()
        allTrues[f"{language_model}{pooled_text}_{type(classification_model).__name__}"] = trues.copy()

# save model predictions together with true sentiment labels
pd.DataFrame(allPreds).to_csv(f"./data/partial_results/{dependent_variable}_{data_file_name}_{test_run}"
                              f"{pooled_text}_predictions.csv")
pd.DataFrame(allTrues).to_csv(f"./data/partial_results/{dependent_variable}_{data_file_name}_{test_run}"
                              f"{pooled_text}_trues.csv")

_utils.compute_metrics(dependent_variable=dependent_variable,
                       test_run=test_run,
                       data_file_name=data_file_name,
                       pooled_text=pooled_text)

