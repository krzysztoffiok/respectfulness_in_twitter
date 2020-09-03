import pandas as pd
import argparse
import numpy as np
import flair
import torch
import datatable
import time
import math
import os

""" example use to create tweet-level embeddings
# remember to start by setting a proper --dependent_variable

python3 embed_data.py --dependent_variable=respect --fold=3 --test_run=fasttext

Important: needs to be run separately for each fold e.g. with bash command:
for i in {0..4}; do python3 embed_data.py --dependent_variable=respect --fold=$i --test_run=fasttext  ; done

# with LSTM training over the pre-trained model
python3 embed_data.py --dependent_variable=respect --fold=3 --test_run=fasttext
python3 embed_data.py --fold=5 --test_run=roberta_lstm

# no LSTM, just mean of pre-trained token embeddings without fine-tuning
python3 embed_data.py --pool=True  --test_run=fasttext
python3 embed_data.py --pool=True  --test_run=roberta

# CLS token output of fine-tuned transformer model
python3 embed_data.py --fold=0 --test_run=roberta_ft

# universal sentence encoder
python3 embed_data.py --use=True

"""
flair.device = torch.device('cuda')

parser = argparse.ArgumentParser(description='Embed data')
parser.add_argument('--test_run', required=False, default='',
                    type=str, help='name of model')
parser.add_argument("--nrows", required=False, default=40000, type=int)
parser.add_argument("--fold", required=False, default=5, type=int,help="precise fold number, not total number of folds")
parser.add_argument("--pool", required=False, default=False, type=bool)
parser.add_argument("--use", required=False, default=False, type=bool)
parser.add_argument('--dependent_variable', required=False, type=str, default='respect')
parser.add_argument("--bs", required=False, default=32, type=int)
parser.add_argument("--filepath", required=False, default='input_file.xlsx', type=str)
args = parser.parse_args()

filepath = os.path.join('./data/source_data/', args.filepath)
data_file_name = args.filepath.split('.')[0]
data_file_extension = args.filepath.split('.')[1]
fold = args.fold
test_run = args.test_run
if "/" in test_run:
    test_run = test_run.split("/")[1]
nrows = args.nrows
pool = args.pool
_use = args.use
dependent_variable = args.dependent_variable

bs = args.bs

# read data
if data_file_extension == "xlsx":
    df = pd.read_excel(filepath, converters={'dummy_id': str})
elif data_file_extension == "csv":
    df = pd.read_csv(filepath, converters={'dummy_id': str})

print(len(df))
df = df.head(nrows)
data = df.copy()

data = data[['text', dependent_variable, "dummy_id"]]

# if not universal sentence encoder
if not _use:
    # load Flair
    import torch
    import flair
    from flair.models import TextClassifier

    # load various embeddings
    from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, RoBERTaEmbeddings,\
        SentenceTransformerDocumentEmbeddings, TransformerWordEmbeddings
    from flair.data import Sentence

    # if trained embeddings
    if not pool:
        # embeddings trained to the "downstream task"
        model = TextClassifier.load(f'./data/models/{dependent_variable}'
                                    f'_{data_file_name}_{fold}/{test_run}_best-model.pt')
        document_embeddings = model.document_embeddings
        print("model loaded")

    # if simple pooled embeddings or pretrained embeddings
    else:
        if test_run == "fasttext":
            document_embeddings = DocumentPoolEmbeddings([WordEmbeddings('en-twitter')])
        elif test_run == "albert-base-v2":
            document_embeddings = DocumentPoolEmbeddings([TransformerWordEmbeddings(model=test_run)])
        elif test_run == "roberta":
            document_embeddings = DocumentPoolEmbeddings(
                [RoBERTaEmbeddings(pretrained_model_name_or_path="roberta-large", layers="21,22,23,24",
                                   pooling_operation="first", use_scalar_mix=True)])
        # case for SentenceTransformer models
        elif test_run in ['distilbert-base-nli-stsb-mean-tokens', 'roberta-large-nli-stsb-mean-tokens']:
            document_embeddings = SentenceTransformerDocumentEmbeddings(test_run)
        else:
            print("You need to define proper model name in the code"
                  " or choose from two predefined options: --test_run=fasttext or --test_run=roberta")

    # prepare df for output to csv
    # batch size for embedding tweet instances
    tweets_to_embed = data['text'].copy()
    print("beginning embedding")

    # prepare mini batches
    low_limits = list()
    for x in range(0, len(tweets_to_embed), bs):
        low_limits.append(x)
    up_limits = [x + bs for x in low_limits[:-1]]
    up_limits.append(len(tweets_to_embed))

    # a placeholder for embedded tweets and time of computation
    newEmbedings = list()
    embedding_times = list()

    # embeddings tweets
    for i in range(len(low_limits)):
        it = time.time()
        print(f"batch {math.ceil(up_limits[i] / bs)}")
        # get the list of current tweet instances
        slist = tweets_to_embed.iloc[low_limits[i]:up_limits[i]].to_list()

        # create a list of Sentence objects
        sentlist = list()
        for sent in slist:
            sentlist.append(Sentence(sent, use_tokenizer=True))

        # feed the list of Sentence objects to the model and output embeddings
        document_embeddings.embed(sentlist)

        # add embeddings of sentences to a new data frame
        for num, sentence in enumerate(sentlist):
            sent_emb = sentence.get_embedding()
            newEmbedings.append(sent_emb.squeeze().tolist())

        ft = time.time()
        embedding_times.append((ft - it) / bs)

    print("Average tweet embedding time: ", np.array(embedding_times).mean())
    print("Total tweet embedding time: ", len(tweets_to_embed)*np.array(embedding_times).mean())
    # save all embeddings in a DataFrame
    df = pd.DataFrame(newEmbedings)

    # add rows with target variable and dummy_id for identification of rows
    df = df.astype(np.float16)
    df[dependent_variable] = data[dependent_variable]
    df["dummy_id"] = data["dummy_id"].astype(str)

    print(df.head())

    # if trained embeddings
    if not pool:
        df.to_csv(f"./data/embeddings/{dependent_variable}_{data_file_name}_{test_run}_encoded_sentences_{fold}.csv")
    # if pooled embeddings
    else:
        df.to_csv(f"./data/embeddings/{dependent_variable}_{data_file_name}_{test_run}_encoded_sentences_pooled.csv")

# if universal sentence encoder (USE)
else:
    # a placeholder for embedded tweets and time of computation
    newEmbedings = list()
    embedding_times = list()

    # import and load USE
    import tensorflow_hub as hub
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    tweets_to_embed = data['text'].to_list()
    it = time.time()
    use_embeddings = embed(tweets_to_embed)

    for _, use_embedding in enumerate(np.array(use_embeddings).tolist()):
        # create a list of embeddings
        newEmbedings.append(use_embedding)

    ft = time.time()
    print("Average tweet embedding time: ", (ft-it)/len(tweets_to_embed))
    print("Total tweet embedding time: ", ft-it)

    # save all embeddings in a DataFrame
    df = pd.DataFrame(newEmbedings)

    # add rows with target variable and dummy_id for identification of rows
    df = df.astype(np.float16)
    df['sentiment'] = data['sentiment']
    df["dummy_id"] = data["dummy_id"].astype(str)

    # output USE embeddings
    df.to_csv(f"./data/embeddings/{dependent_variable}_{data_file_name}_USE_encoded_sentences.csv")
    print("USE embeddings saved to: ", f"./data/embeddings/"
                                       f"{dependent_variable}_{data_file_name}_USE_encoded_sentences.csv")
