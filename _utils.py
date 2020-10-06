from sklearn import metrics
import shap
import random
from sklearn import metrics
from scipy.stats import wasserstein_distance
import datatable as dt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold
import datatable as dt
import matplotlib.pyplot as plt


def ML_classification(allFeatures, train_ml, test_ml, df_ml, classification_model, language_model, folds, pooled_text):
    """
    Function to train classification models on features provided by language models
    Example use: classification_model=RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_split=2,
                             min_samples_leaf=1, max_features='auto', n_jobs=-1, random_state=2020)
                language_model=
    possible options for language model list are: "Term Frequency", "LIWC", "Pooled FastText", "Pooled RoBERTa"
     or "Universal Sentence Encoder"
    """
    # list of analyzed language models
    model = classification_model
    preds = list()
    trues = list()

    print("Now training: ", language_model, pooled_text, " ", type(model).__name__)
    # for each fold
    for fold in range(folds):
        # chose appropriate features and data
        features = set(allFeatures[language_model][fold])
        train_index = train_ml[fold]
        test_index = test_ml[fold]

        train_data = df_ml[features].iloc[train_index]
        target_train_data = df_ml["target_ml"].iloc[train_index]
        test_data = df_ml[features].iloc[test_index]
        target_test_data = df_ml.iloc[test_index]["target_ml"]
        model.fit(train_data, target_train_data)

        preds.append(model.predict(test_data).tolist())
        trues.append(target_test_data.tolist())
    return sum(preds, []), sum(trues, [])


def compute_metrics(dependent_variable, test_run, data_file_name, pooled_text):

    # code borrowed from https://gist.github.com/nickynicolson/202fe765c99af49acb20ea9f77b6255e
    def cm2df(cm, labels):
        df = pd.DataFrame()
        # rows
        for i, row_label in enumerate(labels):
            rowdata = {}
            # columns
            for j, col_label in enumerate(labels):
                rowdata[col_label] = cm[i, j]
            df = df.append(pd.DataFrame.from_dict({row_label: rowdata}, orient='index'))
        return df[labels]

    # read result files to compute metrics including SemEval2017-specific
    preds = dt.fread(f"./data/partial_results/{dependent_variable}_{data_file_name}_{test_run}{pooled_text}"
                     f"_predictions.csv").to_pandas()
    trues = dt.fread(f"./data/partial_results/{dependent_variable}_{data_file_name}_{test_run}{pooled_text}"
                     f"_trues.csv").to_pandas()
    modelColNames = preds.columns.to_list()
    modelColNames.remove("C0")

    # define classes and indexes of true values for each class. For each model the true index values are the
    # same since the test set was the same.
    classes = set(trues[f"{modelColNames[0]}"])
    cls_index = dict()
    for cls in classes:
        cls_index[cls] = trues[trues[f"{modelColNames[0]}"] == cls].index.to_list()

    # for each model compute the metrics
    allmetrics = dict()
    for model in modelColNames:
        model_metrics = dict()
        mae = metrics.mean_absolute_error(y_true=trues[f"{model}"], y_pred=preds[f"{model}"])
        emd = wasserstein_distance(trues[f"{model}"], preds[f"{model}"])
        mcc = metrics.matthews_corrcoef(y_true=trues[f"{model}"], y_pred=preds[f"{model}"])
        f1 = metrics.f1_score(y_true=trues[f"{model}"], y_pred=preds[f"{model}"], average="macro")

        cm = metrics.confusion_matrix(y_true=trues[f"{model}"], y_pred=preds[f"{model}"])
        cm_as_df = cm2df(cm, [0, 1, 2])
        cm_as_df.to_excel(f'./results/confusion_matrix_{dependent_variable}_{data_file_name}_{test_run}{pooled_text}.xlsx')

        # class wise computation of mean absolute error and later averaging over classes to implement
        # MAEM from SemEval2017
        mae_dict = {}

        for cls in classes:
            local_trues = trues[f"{model}"].iloc[cls_index[cls]]
            local_preds = preds[f"{model}"].iloc[cls_index[cls]]
            mae_dict[cls] = metrics.mean_absolute_error(y_true=local_trues, y_pred=local_preds)

        mmae = np.array(list(mae_dict.values())).mean()
        _metrics = {"MMAE": mmae, "MAE": mae, "EMD": emd, "MCC": mcc, "F1": f1}
        for metric in _metrics.keys():
            model_metrics[metric] = _metrics[metric]

        allmetrics[model] = model_metrics

    dfmetrics = pd.DataFrame.from_dict(allmetrics)
    dfmetrics.to_csv(f"./results/{dependent_variable}_{data_file_name}_{test_run}{pooled_text}_metric_results.csv")
    print(dfmetrics)


def term_frequency(train_ml, df, allFeatures, dependent_variable):
    foldTFfeatures = {}
    allWords = []
    for fold, rows in train_ml.items():
        vectorizer = CountVectorizer(min_df=4, binary=True)
        tf = vectorizer.fit_transform(df.iloc[rows]["text"])
        dftf = pd.DataFrame(tf.A, columns=vectorizer.get_feature_names())
        mi_imps = list(zip(mutual_info_classif(dftf, df.iloc[rows][dependent_variable], discrete_features=True),
                           dftf.columns))
        mi_imps = sorted(mi_imps, reverse=True)
        topFeaturesN = 300
        foldTFfeatures[fold] = [f"TF_{y}" for x, y in mi_imps[0:topFeaturesN]].copy()
        # save all words found by TF models as important features
        allWords.extend([y for x, y in mi_imps[0:topFeaturesN]].copy())

    # add the Term Frequency language model key to dictionary with allFeatures from various language models
    allFeatures["TF"] = foldTFfeatures

    # Create TF features for all the text instances and create a corresponding data frame
    allWords = list(set(allWords))
    vectorizer = CountVectorizer(min_df=4, binary=True, vocabulary=allWords)
    tf = vectorizer.fit_transform(df["text"])
    dftf = pd.DataFrame(tf.A, columns=vectorizer.get_feature_names())
    dftf.columns = [f"TF_{x}" for x in dftf.columns]

    return dftf, allFeatures


def splitter(folds, df):

    # this split (with folds=5) results in: 20% test, 10% val, 70% train for Flair framework
    # and the same 20% test and 80 % train for Machine Learning
    indexes = list(range(0, len(df)))

    # setup random state and folds
    np.random.seed(13)
    if folds >= 2:
        kf = KFold(n_splits=folds, random_state=13, shuffle=True)
        kf = kf.split(indexes)
    elif folds == 1:
        kf = KFold(n_splits=2, random_state=13, shuffle=True)
        kf = kf.split(indexes)
    elif folds == 0:
        print("Please provide a number greater than 0.")
        quit()
    return kf


def read_liwc():
    dfliwc = pd.read_excel(f"./data/source_data/LIWCrespect.xlsx", converters={'dummy_id': str})

    # rename columns to get unique names
    dfliwc.rename(columns={'text': 'text_liwc', "sentiment": 'liwc_sent'}, inplace=True)

    # define LIWC features names
    liwcfeatures = ['WC', 'Analytic', 'Clout', 'Authentic',
           'Tone', 'WPS', 'Sixltr', 'Dic', 'function', 'pronoun', 'ppron', 'i',
           'we', 'you', 'shehe', 'they', 'ipron', 'article', 'prep', 'auxverb',
           'adverb', 'conj', 'negate', 'verb', 'adj', 'compare', 'interrog',
           'number', 'quant', 'affect', 'posemo', 'negemo', 'anx', 'anger', 'sad',
           'social', 'family', 'friend', 'female', 'male', 'cogproc', 'insight',
           'cause', 'discrep', 'tentat', 'certain', 'differ', 'percept', 'see',
           'hear', 'feel', 'bio', 'body', 'health', 'sexual', 'ingest', 'drives',
           'affiliation', 'achieve', 'power', 'reward', 'risk', 'focuspast',
           'focuspresent', 'focusfuture', 'relativ', 'motion', 'space', 'time',
           'work', 'leisure', 'home', 'money', 'relig', 'death', 'informal',
           'swear', 'netspeak', 'assent', 'nonflu', 'filler', 'AllPunc', 'Period',
           'Comma', 'Colon', 'SemiC', 'QMark', 'Exclam', 'Dash', 'Quote',
           'Apostro', 'Parenth', 'OtherP']
    return dfliwc, liwcfeatures


def read_seance():
    dfseance = dt.fread(f"./data/source_data/respect_seance.csv").to_pandas()
    dfseance = dfseance.sort_values(["filename"])
    dfseance.drop(["filename"], axis=1, inplace=True)
    # create a list of seance features
    seancefeatures = dfseance.columns.to_list()
    # make seance feature names unique
    seancefeatures = [f"S_{x}" for x in seancefeatures]
    dfseance.columns = seancefeatures
    dfseance.index = [x for x in range(len(dfseance))]
    return dfseance, seancefeatures
