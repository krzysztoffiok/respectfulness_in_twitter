start=`date +%s`

# define models to be trained: transformer models and other models
#declare -a transformer_list=('roberta-large' 'facebook/bart-large-cnn' 'xlm-mlm-en-2048' 'xlm-roberta-large'
# 'bert-large-cased' 'bert-large-uncased')

declare -a transformer_list=()

#declare -a other_model_list=('fasttext' 'roberta_lstm')
declare -a other_model_list=()
# define pretrained SentenceTransformer models to be tested
#declare -a pretrained_model_list=('distilbert-base-nli-stsb-mean-tokens' 'roberta-large-nli-stsb-mean-tokens')
declare -a pretrained_model_list=('albert-base-v2')

# declare dependent variable and number of folds
dependent_variable="respect"
input_file="input_file.xlsx"
folds=1 ### CHANGE FOLDS
let "embed_folds=$folds - 1"

# split the input data
python3 input_data_splitter.py --folds=$folds --filepath=$input_file

# train selected models
#for i in "${transformer_list[@]}"; do python3 ./model_train.py --filepath=$input_file \
# --dependent_variable=$dependent_variable --folds=$folds \
# --test_run="$i" --fine_tune; done

#for i in "${other_model_list[@]}"; do python3 ./model_train.py --filepath=$input_file \
# --dependent_variable=$dependent_variable --folds=$folds \
# --test_run="$i" --epochs=1; done ### CHANGE EPOCHS

# embed data with trained models
#for i in "${transformer_list[@]}"; do for j in $(seq 0 $embed_folds); do python3 embed_data.py \
#--dependent_variable=$dependent_variable --fold=$j --test_run="$i"  --filepath=$input_file; done; done
#for i in "${other_model_list[@]}"; do for j in $(seq 0 $embed_folds); do python3 embed_data.py \
# --dependent_variable=$dependent_variable --fold=$j --test_run="$i"  --filepath=$input_file; done; done

# embed data with pre-trained models
for i in "${pretrained_model_list[@]}"; do python3 embed_data.py \
 --dependent_variable=$dependent_variable --test_run="$i" --pool=True  --filepath=$input_file; done

# carry out the final ML predictions
#for i in "${transformer_list[@]}"; do python3 ./ml_single_model_procedure.py --test_run="$i" \
# --dependent_variable=$dependent_variable --folds=$folds --filepath=$input_file;done
#for i in "${other_model_list[@]}"; do python3 ./ml_single_model_procedure.py --test_run="$i" \
# --dependent_variable=$dependent_variable --folds=$folds --filepath=$input_file;done
for i in "${pretrained_model_list[@]}"; do python3 ./ml_single_model_procedure.py --test_run="$i" \
 --dependent_variable=$dependent_variable --folds=$folds --pooled --filepath=$input_file;done

### CHANGE BASELINE NOT TO COMPUTE AGAIN THE POOLED MODELS
 # compute results for baseline models (TF, LIWC, SEANCE)
#python3 baseline_models.py

end=`date +%s`

runtime=$((end-start))
echo $runtime
destdir=./data/partial_results/experiment_total_time.txt
echo $runtime > $destdir
