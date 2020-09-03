import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
import sklearn
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
import random
import shap
import datatable as dt
import argparse
import _utils
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='Classify data')
parser.add_argument('--fold', required=False, type=int, default=0)
parser.add_argument('--folds', required=False, type=int, default=5)
parser.add_argument('--shap', required=False, default=None, help='If argument is passed, SHAP explanations for'
                                                                 ' a selected LM will be computed. Possible values'
                                                                 'include e.g. SEANCE, LIWC, Term Frequency')
parser.add_argument('--samples', required=False, type=int, default=200,
                    help="number of samples of data to explain by SHAP")
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
_shap = args.shap
_estimators = args.estimators
_shap_samples = args.samples
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

# read embedding files and define corresponding feature names (lists of names)
for emname, embedding in embeddings:
    embfeaturedict = dict()
    for fold in range(folds):
        # read encoded sentences by the selected language model
        if pooled:
            dfemb = dt.fread(
                f"./data/embeddings/{dependent_variable}"
                f"_{data_file_name}_{embedding}_encoded_sentences_pooled.csv").to_pandas()
        else:
            dfemb = dt.fread(
                f"./data/embeddings/{dependent_variable}"
                f"_{data_file_name}_{embedding}_encoded_sentences_{fold}.csv").to_pandas()

        embfeatures = [f"{emname}{fold}row"]

        # define number of feature columns (columns - 3)
        number_of_feature_columns = len(dfemb.columns) - 3

        # create unique feature (column) names
        embfeatures.extend([f"{emname}{fold}{x}" for x in range(number_of_feature_columns)])
        embfeatures.extend([f"{emname}{fold}_sentiment_", f"{emname}{fold}_dummy_id_"])
        dfemb.columns = embfeatures

        # append features from each language model in tuple ((model_name,fold), [features])
        embfeaturedict[fold] = [f"{emname}{fold}{x}" for x in range(number_of_feature_columns)]

        # append encoded sentences by the selected language model to a list of data frames
        dfemblist.append(dfemb)

    # create entry in dictionary with all features for each trained language model
    allFeatures[emname] = embfeaturedict

# concat all Data Frames: liwc, TF, DL embedding into one df_ml that will be used in Machine Learning
df_ml = pd.DataFrame()
for dfemb in dfemblist:
    df_ml = pd.concat([df_ml, dfemb], axis=1)

# define the target variable in the final df_ml data frame
df_ml["target_ml"] = df[dependent_variable]

# <h1> Machine Learning part
# Define list of model names
all_language_models = [x for x in model_list]

if _shap is None:

    # instantiate dictionary for data frames with results
    allPreds = {}
    allTrues = {}

    # define which classification models to use
    models = [xgb.XGBClassifier(objective='multi:softprob', n_jobs=24, learning_rate=0.03,
                                max_depth=10, subsample=0.7, colsample_bytree=0.6,
                                random_state=2020, n_estimators=_estimators)]  # optionally add: tree_method='gpu_hist'

    # use features from selected language models
    for language_model in all_language_models:
        # for training of selected classification models
        for classification_model in models:
            preds, trues = _utils.ML_classification(allFeatures=allFeatures, train_ml=train_ml, test_ml=test_ml,
                                                    df_ml=df_ml, classification_model=classification_model,
                                                    language_model=language_model, folds=folds)

            # save model predictions
            allPreds[f"{language_model}_{type(classification_model).__name__}"] = preds.copy()
            allTrues[f"{language_model}_{type(classification_model).__name__}"] = trues.copy()

    # save model predictions together with true sentiment labels
    pd.DataFrame(allPreds).to_csv(f"./data/partial_results/{dependent_variable}_{data_file_name}_{test_run}"
                                  f"_predictions.csv")
    pd.DataFrame(allTrues).to_csv(f"./data/partial_results/{dependent_variable}_{data_file_name}_{test_run}"
                                  f"_trues.csv")

    _utils.compute_metrics(dependent_variable=dependent_variable, test_run=test_run, data_file_name=data_file_name)
