# %% [markdown]
# # Getting started
# 
# In this notebook, we illustrate how to use the Neural News Recommendation with Multi-Head Self-Attention ([NRMS](https://aclanthology.org/D19-1671/)). The implementation is taken from the [recommenders](https://github.com/recommenders-team/recommenders) repository. We have simply stripped the model to keep it cleaner.
# 
# We use a small dataset, which is downloaded from [recsys.eb.dk](https://recsys.eb.dk/). All the datasets are stored in the folder path ```~/ebnerd_data/*```.

# %% [markdown]
# ## Load functionality

# %%


from ebrec.utils._polars import split_df_chunks
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import tensorflow as tf


from wandb_logger import setup_wandb, WandbCallback

import polars as pl
import datetime
import sys 
import  ebrec
from ebrec.utils._constants import *

from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_prediction_scores,
    truncate_history,
    ebnerd_from_path,
)
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
from ebrec.utils._articles import convert_text2encoding_with_transformers
from ebrec.utils._polars import concat_str_columns, slice_join_dataframes
from ebrec.utils._articles import create_article_id_to_value_mapping
from ebrec.utils._nlp import get_transformers_word_embeddings
from ebrec.utils._python import write_submission_file, rank_predictions_by_score
from tensorflow.keras.models import Sequential
from ebrec.models.newsrec.dataloader import NRMSDataLoader
from ebrec.models.newsrec.model_config import hparams_nrms
from ebrec.models.newsrec import NRMSModel
from tensorflow.keras.backend import clear_session
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import tensorflow as tf
import datetime as dt
import polars as pl
import numpy as np
import gc
import os

from ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_IS_BEYOND_ACCURACY_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
)

print(sys.path)
sys.path.append("/dtu/blackhole/18/202456")
#/dtu/blackhole/18/202456/data/ebnerd_large/behaviors.parquet


# %%
import ebrec
print(ebrec.__file__)


# %%
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.models import Sequential
print("Keras imported successfully!")


# %%
# List all physical devices
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

physical_devices = tf.config.list_physical_devices()
print("Available devices:", physical_devices)

# %%
HISTORY_SIZE = 20
hparams_nrms.history_size = HISTORY_SIZE

# %%
COLUMNS = [
    DEFAULT_USER_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
]
# This notebook is just a simple 'get-started'; we down sample the number of samples to just run quickly through it.


# %% [markdown]
# ## Load dataset

# %%
# from pathlib import Path

# # Use raw string to avoid issues with backslashes
# PATH = Path(r"C:\Users\antot\Downloads\Pigasos\ebnerd-benchmark\examples\quick_start").expanduser()

# # Other configurations
# DATASPLIT = "ebnerd_demo"

# # Create a directory for dumping predictions
# DUMP_DIR = Path("ebnerd_predictions")
# DUMP_DIR.mkdir(exist_ok=True, parents=True)


# %%
from pathlib import Path

# Use raw string to avoid issues with backslashes
PATH = Path("/dtu/blackhole/18/202456").expanduser()
TRAIN = f"ebnerd_small"  # [ebnerd_demo, ebnerd_small, ebnerd_large]
VAL = f"ebnerd_small"
TEST = f"ebnerd_testset"#, "ebnerd_testset_gt"


# Create a directory for dumping predictions
#DUMP_DIR = Path("ebnerd_predictions")
#DUMP_DIR.mkdir(exist_ok=True, parents=True)

# %%
DUMP_DIR = Path("ebnerd_predictions_2")
DUMP_DIR.mkdir(exist_ok=True, parents=True)

# %%
# Load your train and validation datasets directly
df_train = ebnerd_from_path(
    PATH.joinpath("ebnerd_large/train"),
    history_size=HISTORY_SIZE,
    
).select(COLUMNS).pipe(
    sampling_strategy_wu2019,
    npratio=4,
    shuffle=True,
    with_replacement=True,
    seed=123,
).pipe(create_binary_labels_column)

df_validation = ebnerd_from_path(
    PATH.joinpath("ebnerd_large/validation"),
    history_size=HISTORY_SIZE
    
).select(COLUMNS).pipe(
    sampling_strategy_wu2019,
    npratio=4,
    shuffle=True,
    with_replacement=True,
    seed=123,
).pipe(create_binary_labels_column)

print(f"Train samples: {df_train.height}\nValidation samples: {df_validation.height}")

# Preview the datasets
print("Train Data Sample:")
print(df_train.head(2))

print("Validation Data Sample:")
print(df_validation.head(2))


# %%


# %% [markdown]
# ### Generate labels
# We sample a few just to get started. For testset we just make up a dummy column with 0 and 1 - this is not the true labels.

# %%


# %% [markdown]
# History size can often be a memory bottleneck; if adjusted, the NRMS hyperparameter ```history_size``` must be updated to ensure compatibility and efficient memory usage

# %%

#print(f"Model Directory: {MODEL_NAME}")

# Data preprocessing parameters
MAX_TITLE_LENGTH = 30
HISTORY_SIZE = 20
FRACTION = 1.0
EPOCHS = 5
FRACTION_TEST = 1.0
hparams_nrms.history_size = HISTORY_SIZE

# Batch sizes
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_VAL = 64
BATCH_SIZE_TEST_WO_B = 64
BATCH_SIZE_TEST_W_B = 64
N_CHUNKS_TEST = 10
CHUNKS_DONE = 0

# Columns to select
COLUMNS = [
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
]


# %%

# # Load and preprocess the train dataset
# df_train = (
#     ebnerd_from_path(TRAIN_PATH, history_size=HISTORY_SIZE)
#     .sample(fraction=FRACTION, seed=SEED)
#     .select(COLUMNS)
#     .pipe(
#         sampling_strategy_wu2019,
#         npratio=4,
#         shuffle=True,
#         with_replacement=True,
#         seed=SEED,
#     )
#     .pipe(create_binary_labels_column)
# )

# # Load and preprocess the validation dataset
# df_validation = (
#     ebnerd_from_path(VALIDATION_PATH, history_size=HISTORY_SIZE)
#     .sample(fraction=FRACTION, seed=SEED)
#     .select(COLUMNS)
#     .pipe(
#         sampling_strategy_wu2019,
#         npratio=4,
#         shuffle=True,
#         with_replacement=True,
#         seed=SEED,
#     )
#     .pipe(create_binary_labels_column)
# )



# %% [markdown]
# In this example we sample the dataset, just to keep it smaller. We'll split the training data into training and validation 

# %% [markdown]
# ### Test set
# We'll use the validation set, as the test set.

# %%
COLUMNSTEST = [
    DEFAULT_USER_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
]

# %%
df_test = (
   ebnerd_from_path(
       Path("/dtu/blackhole/18/202456/ebnerd_testset/test")
   )
   .sample(fraction=FRACTION)
)

print(f"Test samples: {df_test.height}")
print("Test Data Sample:")
print(df_test.head(2))
#
#
## %%
#df_test.columns


# %%


# %% [markdown]
# ## Load articles


# %%

df_articles_train = pl.read_parquet(PATH.joinpath("ebnerd_large/articles.parquet"))
df_articles_train.head()
# df_articles_test = pl.read_parquet(TEST_MAIN_PATH.joinpath("articles.parquet"))

# %%
df_articles_test = pl.read_parquet(PATH.joinpath(PATH, "ebnerd_testset/articles.parquet"))
df_articles_test.head()


# %%


# %% [markdown]
# ## Init model using HuggingFace's tokenizer and wordembedding
# In the original implementation, they use the GloVe embeddings and tokenizer. To get going fast, we'll use a multilingual LLM from Hugging Face. 
# Utilizing the tokenizer to tokenize the articles and the word-embedding to init NRMS.
# 

# %%
# TRANSFORMER_MODEL_NAME = "FacebookAI/xlm-roberta-base"
from transformers import AutoModel, AutoTokenizer


# Define Available Models
MODEL_CLASSES = {
    "xlm-roberta": "FacebookAI/xlm-roberta-base",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
}

# Select Model
SELECTED_MODEL = "bert"  # Change this to "bert" or "roberta" as needed
TRANSFORMER_MODEL_NAME = MODEL_CLASSES[SELECTED_MODEL] 

TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]
MAX_TITLE_LENGTH = 30

# LOAD HUGGINGFACE:
transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

# We'll init the word embeddings using the
word2vec_embedding = get_transformers_word_embeddings(transformer_model)
#
df_articles_train, cat_cal = concat_str_columns(df_articles_train, columns=TEXT_COLUMNS_TO_USE)
df_articles_train, token_col_title = convert_text2encoding_with_transformers(
    df_articles_train, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
)
# =>
article_mapping_train = create_article_id_to_value_mapping(
    df=df_articles_train, value_col=token_col_title
)



df_articles_test, cat_cal = concat_str_columns(df_articles_test, columns=TEXT_COLUMNS_TO_USE)
df_articles_test, token_col_title = convert_text2encoding_with_transformers(
   df_articles_test, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
)
# =>
article_mapping_test = create_article_id_to_value_mapping(
   df=df_articles_test, value_col=token_col_title
)

# %% [markdown]
# # Initiate the dataloaders
# In the implementations we have disconnected the models and data. Hence, you should built a dataloader that fits your needs.
# 
# Note, with this ```NRMSDataLoader``` the ```eval_mode=False``` is meant for ```model.model.fit()``` whereas ```eval_mode=True``` is meant for ```model.scorer.predict()```. 

# %%
# Initialize DataLoaders for train and validation
print("Initializing train and validation dataloaders...")

# %%
#trained with subsets
BATCH_SIZE = 64 # try with 64
#df_train_subset = df_train[:10000] 
#df_val_subset = df_validation[:10000]  
train_dataloader = NRMSDataLoader(
    behaviors=df_train,
    article_dict=article_mapping_train,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=BATCH_SIZE,
)
val_dataloader = NRMSDataLoader(
    behaviors=df_validation,
    article_dict=article_mapping_train,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=BATCH_SIZE,
)

# %%


# %%


# %% [markdown]
# ## Train the model
# 

# %%
# List all physical devices
# physical_devices = tf.config.list_physical_devices()
# print("Available devices:", physical_devices)

# %% [markdown]
# Initiate the NRMS-model:

# %%
model = NRMSModel(
    hparams=hparams_nrms,
    word2vec_embedding=word2vec_embedding,
    seed=42,
)
model.model.compile(
    optimizer=model.model.optimizer,
    loss=model.model.loss,
    metrics=["AUC"],
)



MODEL_NAME = model.__class__.__name__
MODEL_WEIGHTS = DUMP_DIR.joinpath(f"state_dict/{MODEL_NAME}/weights")
LOG_DIR = DUMP_DIR.joinpath(f"runs/{MODEL_NAME}")

from tensorflow.keras.callbacks import ModelCheckpoint

# Define checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=str(MODEL_WEIGHTS),
    save_weights_only=True,
    monitor='val_auc',
    mode='max',
    save_best_only=True,
    verbose=1
) 

# %% [markdown]
# ### Callbacks
# We will add some callbacks to model training.

# %%
# Tensorboard:
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=LOG_DIR,
    histogram_freq=1,
)

# Earlystopping:
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=3,
    restore_best_weights=True,
)

# ModelCheckpoint:
modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_WEIGHTS,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)

# Learning rate scheduler:
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_auc",
    mode="max",
    factor=0.2,
    patience=2,
    min_lr=1e-5,
)



USE_WANDB = True  # Set this to False to disable WandB logging

if USE_WANDB:
    setup_wandb(
        project_name="NRMS-Model",
        config={
            "learning_rate": model.model.optimizer.learning_rate.numpy(),
            "history_size": hparams_nrms.history_size,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "transformer_model": TRANSFORMER_MODEL_NAME,
        },
    )




# callbacks = [lr_scheduler]

callbacks = [lr_scheduler]

if USE_WANDB:
    callbacks.append(WandbCallback())


# %%
USE_CALLBACKS = True
EPOCHS = 5


hist = model.model.fit(
    train_dataloader,
    validation_data=val_dataloader,
    epochs=EPOCHS,
    callbacks=callbacks if USE_CALLBACKS else [],
)

# Save weights after training
MODEL_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
model.model.save_weights(MODEL_WEIGHTS)
print(f"Model weights saved at: {MODEL_WEIGHTS}")


import gc
from tensorflow.keras.backend import clear_session
from ebrec.utils._polars import split_df_chunks

# Set up validation directories
VAL_DF_DUMP = DUMP_DIR.joinpath("val_predictions", MODEL_NAME)
VAL_DF_DUMP.mkdir(parents=True, exist_ok=True)

# Split validation dataset into manageable chunks
df_val_chunks = split_df_chunks(df_validation, n_chunks=32)
df_pred_val = []

# Loop through each chunk
for i, df_val_chunk in enumerate(df_val_chunks):
    print(f"Processing validation chunk: {i + 1}/{len(df_val_chunks)}")

    # Initialize DataLoader for validation set
    val_dataloader = NRMSDataLoader(
        behaviors=df_val_chunk,
        article_dict=article_mapping_train,
        unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=True,
        batch_size=64,  # Batch size for validation
    )

    # Predict scores for validation chunk
    scores = model.scorer.predict(val_dataloader)
    clear_session()

    # Add predictions to the DataFrame
    df_val_chunk = add_prediction_scores(df_val_chunk, scores.tolist()).with_columns(
        pl.col("scores")
        .map_elements(lambda x: list(rank_predictions_by_score(x)))  # Rank predictions
        .alias("ranked_scores")
    )

    # Save predictions for this chunk
    df_val_chunk.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
        VAL_DF_DUMP.joinpath(f"val_pred_chunk_{i + 1}.parquet")
    )

    # Append processed chunk
    df_pred_val.append(df_val_chunk)

    # Cleanup to release memory
    del df_val_chunk, val_dataloader, scores
    gc.collect()

# Combine all validation chunks into a single DataFrame
df_pred_val_combined = pl.concat(df_pred_val)

# Save the full validation predictions
df_pred_val_combined.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
    VAL_DF_DUMP.joinpath("val_predictions_combined.parquet")
)

# Print the first few rows for inspection
print("Validation Predictions:")
print(df_pred_val_combined.head())



from ebrec.evaluation.metrics import (
    mean_squared_error,
    accuracy_score,
    roc_auc_score,
    ndcg_score,
    mrr_score,
    log_loss,
    f1_score,
)

evaluator = MetricEvaluator(
    labels=df_pred_val_combined["labels"].to_list(),
    predictions=df_pred_val_combined["scores"].to_list(),
    metric_functions=[
        AucScore(),
        MrrScore(),
        NdcgScore(k=5),
        NdcgScore(k=10),
    ],
)
results = evaluator.evaluate()
print(results)



#---------------TEST--------------------------

df_test = (
    ebnerd_from_path(
        PATH.joinpath(PATH, "ebnerd_testset/test")
    )
    .sample(fraction=FRACTION)
)

print(f"Test samples: {df_test.height}")
print("Test Data Sample:")
print(df_test.head(2))


df_test = (
    ebnerd_from_path(PATH.joinpath("ebnerd_testset", "test"), history_size=HISTORY_SIZE)
    .sample(fraction=FRACTION_TEST)
    .with_columns(
        pl.col(DEFAULT_INVIEW_ARTICLES_COL)
        .list.first()
        .alias(DEFAULT_CLICKED_ARTICLES_COL)
    )
    .select(COLUMNS + [DEFAULT_IS_BEYOND_ACCURACY_COL])
    .with_columns(
        pl.col(DEFAULT_INVIEW_ARTICLES_COL)
        .list.eval(pl.element() * 0)
        .alias(DEFAULT_LABELS_COL)
    )
)
df_test.head()


# Filter rows into two subsets
df_test_wo_beyond = df_test.filter(~pl.col("is_beyond_accuracy"))
df_test_w_beyond = df_test.filter(pl.col("is_beyond_accuracy"))

# Verify the split
print("Rows without beyond accuracy (False):", df_test_wo_beyond.shape[0])
print("Rows with beyond accuracy (True):", df_test_w_beyond.shape[0])






df_test_chunks = split_df_chunks(df_test_wo_beyond, n_chunks=N_CHUNKS_TEST)
df_pred_test_wo_beyond = []

BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL = 32
BATCH_SIZE_TEST_WO_B = 32
BATCH_SIZE_TEST_W_B = 4
N_CHUNKS_TEST = 10
CHUNKS_DONE = 0


import gc
from tensorflow.keras.backend import clear_session
TEST_DF_DUMP = DUMP_DIR.joinpath("test_predictions", MODEL_NAME)
TEST_DF_DUMP.mkdir(parents=True, exist_ok=True)

df_test_chunks = split_df_chunks(df_test_wo_beyond, n_chunks=N_CHUNKS_TEST)
df_pred_test_wo_beyond = []

for i, df_test_chunk in enumerate(df_test_chunks[CHUNKS_DONE:], start=1 + CHUNKS_DONE):
    print(f"Init test-dataloader: {i}/{len(df_test_chunks)}")
    # Initialize DataLoader
    test_dataloader_wo_b = NRMSDataLoader(
        behaviors=df_test_chunk,
        article_dict=article_mapping_test,
        unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=True,
        batch_size=BATCH_SIZE_TEST_WO_B,
    )
    # Predict and clear session
    scores = model.scorer.predict(test_dataloader_wo_b)
    clear_session()

    # Process the predictions
    df_test_chunk = add_prediction_scores(df_test_chunk, scores.tolist()).with_columns(
        pl.col("scores")
        .map_elements(lambda x: list(rank_predictions_by_score(x)))
        .alias("ranked_scores")
    )

    # Save the processed chunk
    df_test_chunk.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
        TEST_DF_DUMP.joinpath(f"pred_wo_ba_{i}.parquet")
    )

    # Append and clean up
    df_pred_test_wo_beyond.append(df_test_chunk)

    # Cleanup
    del df_test_chunk, test_dataloader_wo_b, scores
    gc.collect()


    import polars as pl

# Concatenate all DataFrame chunks into a single DataFrame
df_pred_test_wo_beyond = pl.concat(df_pred_test_wo_beyond)

# Now you can use the .select() method
df_pred_test_wo_beyond.select(DEFAULT_IMPRESSION_ID_COL, "scores").write_parquet(
    TEST_DF_DUMP.joinpath("pred_wo_ba.parquet")
)

# View the head of the DataFrame
print(df_pred_test_wo_beyond.head(30))


print("Init test-dataloader: beyond-accuracy")
test_dataloader_w_b = NRMSDataLoader(
    behaviors=df_test_w_beyond,
    article_dict=article_mapping_test,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=BATCH_SIZE_TEST_W_B,
)

scores = model.scorer.predict(test_dataloader_w_b)
df_pred_test_w_beyond = add_prediction_scores(
    df_test_w_beyond, scores.tolist()
).with_columns(
    pl.col("scores")
    .map_elements(lambda x: list(rank_predictions_by_score(x)))
    .alias("ranked_scores")
)
df_pred_test_w_beyond.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
    TEST_DF_DUMP.joinpath("pred_w_ba.parquet")
)

# Check the schemas of both DataFrames
print("Schema of df_pred_test_wo_beyond:")
print(df_pred_test_wo_beyond.schema)

print("Schema of df_pred_test_w_beyond:")
print(df_pred_test_w_beyond.schema)


# Check the schemas of both DataFrames
print("Schema of df_pred_test_wo_beyond:")
print(df_pred_test_wo_beyond.schema)

print("Schema of df_pred_test_w_beyond:")
print(df_pred_test_w_beyond.schema)

# Align column types
df_pred_test_wo_beyond = df_pred_test_wo_beyond.with_columns(
    [pl.col(column).cast(df_pred_test_w_beyond.schema[column]) for column in df_pred_test_w_beyond.schema]
)

# Combine both DataFrames
df_test = pl.concat([df_pred_test_wo_beyond, df_pred_test_w_beyond])

# Write to Parquet
df_test.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
    TEST_DF_DUMP.joinpath("pred_concat.parquet")
)


import polars as pl
import numpy as np

# Update the 'labels' column to match the length of 'ranked_scores' and assign 1 to the highest rank
df_test = df_test.with_columns(
    pl.struct(["ranked_scores", "scores"])
    .apply(lambda row: [1 if rank == 1 else 0 for rank in row["ranked_scores"]]
           if len(row["ranked_scores"]) == len(row["scores"]) else None)
    .alias("labels")
)

# Check for rows where labels are None (mismatched lengths)
invalid_rows = df_test.filter(pl.col("labels").is_null())

if invalid_rows.height > 0:
    print("Found rows with mismatched 'ranked_scores' and 'scores':")
    print(invalid_rows)

# Verify the updated 'labels' column
print(df_test.select(["scores", "labels"]))


# Importing the required library

#/dtu/blackhole/18/202456
DATASPLIT = "ebnerd_large_test"

write_submission_file(
    impression_ids=df_test[DEFAULT_IMPRESSION_ID_COL],
    prediction_scores=df_test["ranked_scores"],
    path=DUMP_DIR.joinpath("predictions2.txt"),
    filename_zip=f"{DATASPLIT}_predictions2-{MODEL_NAME}.zip",
)



