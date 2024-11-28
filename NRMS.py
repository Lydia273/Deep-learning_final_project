# %%
import pandas as pd
import sentencepiece as spmy
from sklearn.model_selection import train_test_split
import numpy as np
import torch

# %% [markdown]
# Preparing the GloVe embeddings and tokenizer 

# %%
import zipfile
from huggingface_hub import hf_hub_download
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List
import torch
import tokenizers
import rich

# Load GloVe embeddings
from pathlib import Path
import zipfile
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from typing import List, Tuple

def load_glove_vectors(filename="glove.840B.300d.txt") -> Tuple[List[str], torch.Tensor]:
    """
    Load the GloVe vectors and parse vocabulary and vectors.

    Args:
        filename (str): Name of the GloVe file to load.

    Returns:
        vocabulary (List[str]): List of words in the GloVe vocabulary.
        vectors (torch.Tensor): Tensor of embedding vectors.
    """
    # Download GloVe file from Hugging Face hub (ensure correct version)
    try:
        path = Path(hf_hub_download(repo_id="stanfordnlp/glove", filename="glove.840B.300d.zip"))
    except Exception as e:
        raise ValueError("Error downloading the GloVe file from Hugging Face: " + str(e))
    
    # Unzip if the file exists
    target_file = path.parent / filename
    if not target_file.exists():
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(path.parent)

        # Check if file was extracted correctly
        if not target_file.exists():
            print("Available files:")
            for p in path.parent.iterdir():
                print(p)
            raise ValueError(f"Target file `{filename}` can't be found. Check if `{filename}` was properly downloaded.")
    
    # Parse vocabulary and vectors
    vocabulary = []
    vectors = []
    with open(target_file, "r", encoding="utf8") as f:
        for line in tqdm(f.readlines(), desc=f"Parsing {target_file.name}..."):
            word, *vector = line.split()
            vocabulary.append(word)
            vectors.append(torch.tensor([float(v) for v in vector], dtype=torch.float32))
    vectors = torch.stack(vectors)
    return vocabulary, vectors



# %%
# Prepare data
glove_vocabulary, glove_vectors = load_glove_vectors() #glove voc is a list of words from the glove embeddings , glove vec is a tensor where each row represents the embedding of a word in glove voc 
rich.print(f"glove_vocabulary: type={type(glove_vocabulary)}, length={len(glove_vocabulary)}")
rich.print(f"glove_vectors: type={type(glove_vectors)}, shape={glove_vectors.shape}, dtype={glove_vectors.dtype}")

# Add special tokens
special_tokens = ['<|start|>', '<|unknown|>', '<|pad|>']  # add special toeksn , useful for seq processing
glove_vocabulary = special_tokens + glove_vocabulary # creates random embeddings for these tokens with the same dim as other glove embeddings
glove_vectors = torch.cat([torch.randn(len(special_tokens), glove_vectors.shape[1]), glove_vectors]) # concatenates these new embeddings to the glove vectors

# Tokenizer for GloVe
glove_tokenizer = tokenizers.Tokenizer( # A WordLevel tokenizer maps each word in the vocabulary to a unique integer (token ID).
    tokenizers.models.WordLevel(vocab={v: i for i, v in enumerate(glove_vocabulary)}, unk_token="<|unknown|>") # The tokenizer uses <|unknown|> as a fallback for words not found in the vocabulary.
)
glove_tokenizer.normalizer = tokenizers.normalizers.BertNormalizer(strip_accents=False)
glove_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

# %%
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Step 1: Load train and validation datasets
dataset_train_parquet = "final_df_train.parquet"
df_train = pd.read_parquet(dataset_train_parquet, engine="pyarrow")
df_train = df_train.drop(columns=['articles_num'])

# One-hot encode the 'sentiment_label' column
df_train = pd.get_dummies(df_train, columns=['sentiment_label'], prefix='sentiment')


dataset_validation_parquet = "final_df_valid.parquet"
df_validation = pd.read_parquet(dataset_validation_parquet, engine="pyarrow")

df_validation = df_validation.drop(columns=['articles_num'])

# One-hot encode the 'sentiment_label' column
df_validation = pd.get_dummies(df_validation, columns=['sentiment_label'], prefix='sentiment')


# Combine train and validation titles into a single process for tokenization and padding
all_titles = pd.concat(
    [df_train["title"].fillna(""), df_validation["title"].fillna("")], ignore_index=True
).tolist()


# Define padding length
MAX_TITLE_LENGTH = 10  # Adjust this to the max length of titles you want

# Modify tokenization function to pad or truncate titles
def tokenize_titles(titles, tokenizer, max_length=MAX_TITLE_LENGTH):
    tokenized_titles = []
    pad_token_id = tokenizer.token_to_id("<|pad|>")  # Get the padding token ID from the tokenizer

    for title in tqdm(titles, desc="Tokenizing titles"):
        encoding = tokenizer.encode(title)
        token_ids = encoding.ids
        token_ids = [min(id, len(glove_vocabulary) - 1) for id in token_ids]  # Clip the IDs if they are out of range

        # Truncate or pad the tokenized title to max_length
        if len(token_ids) < max_length:
            # Pad to the right
            token_ids += [pad_token_id] * (max_length - len(token_ids))
        else:
            # Truncate the sequence
            token_ids = token_ids[:max_length]

        tokenized_titles.append(token_ids)

    return tokenized_titles

# Tokenize and pad all titles
tokenized_titles = tokenize_titles(all_titles, glove_tokenizer, max_length=MAX_TITLE_LENGTH)

# Convert tokenized titles to tensor
tokenized_titles_tensor = torch.tensor(tokenized_titles, dtype=torch.long)

# You can now use tokenized_titles_tensor for embedding input


# You can now use tokenized_titles_tensor for embedding input


# Step 3: Prepare user history and candidate news
def prepare_data(df):
    user_histories = []
    candidate_news = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Preparing data"):
        # Assuming 'article_ids_clicked' contains a list of article IDs user has clicked
        user_history = row['article_ids_clicked']  # This should be a list of IDs
        candidate_news_id = row['article_id']  # Current article ID for prediction
        
        # If the history is less than the expected length, pad with -1 (or any other padding value)
        user_history_padded = user_history[:10] + [-1] * (10 - len(user_history))  # Max length 10 for history
        
        # Append the history and candidate news
        user_histories.append(user_history_padded)
        candidate_news.append([candidate_news_id] * 20)  # Assuming you have 20 candidate news articles

    # Convert lists to tensors
    user_histories_tensor = torch.tensor(user_histories, dtype=torch.long)
    candidate_news_tensor = torch.tensor(candidate_news, dtype=torch.long)

    return user_histories_tensor, candidate_news_tensor

# Prepare data for train set
train_user_history, train_candidate_news = prepare_data(df_train)

# Prepare data for validation set
valid_user_history, valid_candidate_news = prepare_data(df_validation)

# Check shapes (optional)
print("Train user history shape:", train_user_history.shape)
print("Train candidate news shape:", train_candidate_news.shape)
print("Validation user history shape:", valid_user_history.shape)
print("Validation candidate news shape:", valid_candidate_news.shape)



# %%

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# Word Embedding Layer
class WordEmbeddings(nn.Module):
    def __init__(self, embedding_matrix):
        super(WordEmbeddings, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=True  # Set to False to fine-tune embeddings
        )

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        return embeddings


# Multi-Head Self-Attention Layer (Vectorized)
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, and V
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            output (torch.Tensor): Output tensor of shape (batch_size, seq_len, embed_dim)
            attention_weights (torch.Tensor): Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, embed_dim = inputs.size()

        # Project inputs to queries, keys, and values
        Q = self.query_projection(inputs)  # (batch_size, seq_len, embed_dim)
        K = self.key_projection(inputs)  # (batch_size, seq_len, embed_dim)
        V = self.value_projection(inputs)  # (batch_size, seq_len, embed_dim)

        # Reshape to separate heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # Compute scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = F.softmax(attention_scores, dim=-1)  # Normalize over seq_len

        # Compute context
        context = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads and project the output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)
        output = self.out_projection(context)  # (batch_size, seq_len, embed_dim)

        return output, attention_weights


# Additive Word Attention Layer
class AdditiveWordAttention(nn.Module):
    def __init__(self, embed_dim, query_dim):
        super(AdditiveWordAttention, self).__init__()
        self.Vw = nn.Linear(embed_dim, query_dim, bias=True)
        self.qw = nn.Parameter(torch.randn(query_dim))

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): Tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            weighted_sum (torch.Tensor): Tensor of shape (batch_size, embed_dim)
            attention_weights (torch.Tensor): Tensor of shape (batch_size, seq_len)
        """
        projection = self.Vw(inputs)  # (batch_size, seq_len, query_dim)
        scores = torch.tanh(projection)  # Non-linear transformation
        attention_scores = torch.matmul(scores, self.qw)  # (batch_size, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)  # Normalize over seq_len
        weighted_sum = torch.sum(inputs * attention_weights.unsqueeze(-1), dim=1)  # Weighted sum of embeddings

        return weighted_sum, attention_weights


# News Encoder
class NewsEncoder(nn.Module):
    def __init__(self, embedding_matrix, embed_dim, num_heads, query_dim, dropout=0.2):
        super(NewsEncoder, self).__init__()
        self.word_embeddings = nn.Embedding.from_pretrained(
            embedding_matrix.clone().detach(), freeze=False
        )
        self.self_attention = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.additive_attention = AdditiveWordAttention(embed_dim=embed_dim, query_dim=query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len)

        Returns:
            news_representation (torch.Tensor): Tensor of shape (batch_size, embed_dim)
            attention_weights (torch.Tensor): Tensor of shape (batch_size, seq_len)
        """
        x = self.word_embeddings(inputs)  # Shape: (batch_size, seq_len, embed_dim)
        x = self.dropout(x)

        # Multi-Head Self-Attention
        attn_output, _ = self.self_attention(x)

        # Additive Attention
        news_representation, attention_weights = self.additive_attention(attn_output)

        return news_representation, attention_weights


# %%
df_train.columns

# %%
class MultiHeadAdditiveAttention(nn.Module):
    """
    Implements multi-head additive attention for user encoding,
    strictly following the provided formulas.
    """
    def __init__(self, embed_dim, num_heads):
        """
        Args:
        - embed_dim: Dimensionality of the input embeddings (news representations).
        - num_heads: Number of attention heads.
        """
        super(MultiHeadAdditiveAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Parameters for each head
        self.Q_n = nn.ParameterList([nn.Parameter(torch.randn(self.head_dim, self.head_dim)) for _ in range(num_heads)])
        self.V_n = nn.ParameterList([nn.Parameter(torch.randn(self.head_dim, self.head_dim)) for _ in range(num_heads)])

    def forward(self, news_embeddings):
        """
        Args:
        - news_embeddings: Tensor of shape (batch_size, num_news, embed_dim),
                           representing the news representations.

        Returns:
        - enhanced_news_embeddings: Tensor of shape (batch_size, num_news, embed_dim),
                                    enhanced news representations.
        - attention_weights: List of tensors of shape (batch_size, num_news, num_news) per head,
                             attention weights for each head.
        """
        batch_size, num_news, embed_dim = news_embeddings.size()

        # Split embeddings for each head
        news_per_head = news_embeddings.view(batch_size, num_news, self.num_heads, self.head_dim).transpose(1, 2)

        all_head_outputs = []
        all_attention_weights = []

        for h in range(self.num_heads):
            Q_n = self.Q_n[h]
            scores = torch.einsum('bnd,dk,bmk->bnm', news_per_head[:, h, :, :], Q_n, news_per_head[:, h, :, :])
            attention_weights = F.softmax(scores, dim=-1)
            V_n = self.V_n[h]
            head_output = torch.einsum('bnm,bmd,dk->bnd', attention_weights, news_per_head[:, h, :, :], V_n)

            all_head_outputs.append(head_output)
            all_attention_weights.append(attention_weights)

        concat_output = torch.cat(all_head_outputs, dim=-1)
        return concat_output, all_attention_weights
    

class UserAdditiveAttention(nn.Module):
    """
    Implements additive attention for user encoding based on the provided formulas.
    """
    def __init__(self, embed_dim):
        """
        Args:
        - embed_dim: Dimensionality of the input embeddings (news representations).
        """
        super(UserAdditiveAttention, self).__init__()
        self.V_n = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_n = nn.Parameter(torch.zeros(embed_dim))
        self.q_n = nn.Parameter(torch.randn(embed_dim))

    def forward(self, news_embeddings):
        """
        Args:
        - news_embeddings: Tensor of shape (batch_size, num_news, embed_dim),
                           representing the news representations.

        Returns:
        - user_representation: Tensor of shape (batch_size, embed_dim),
                               the final user representation.
        - attention_weights: Tensor of shape (batch_size, num_news),
                             attention weights for the news articles.
        """
        transformed_news = self.V_n(news_embeddings)
        scores = torch.tanh(transformed_news + self.v_n)
        scores = torch.einsum('bnd,d->bn', scores, self.q_n)
        attention_weights = F.softmax(scores, dim=1)
        user_representation = torch.einsum('bn,bnd->bd', attention_weights, news_embeddings)
        return user_representation, attention_weights
    

class UserEncoder(nn.Module):
    """
    Combines MultiHeadAdditiveAttention and UserAdditiveAttention
    to encode user representations based on news embeddings.
    """
    def __init__(self, embed_dim, num_heads):
        """
        Args:
        - embed_dim: Dimensionality of the input embeddings (news representations).
        - num_heads: Number of attention heads in the MultiHeadAdditiveAttention layer.
        """
        super(UserEncoder, self).__init__()
        self.multi_head_attention = MultiHeadAdditiveAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.user_attention = UserAdditiveAttention(embed_dim=embed_dim)

    def forward(self, news_embeddings):
        """
        Args:
        - news_embeddings: Tensor of shape (batch_size, num_news, embed_dim),
                           representing the news representations.

        Returns:
        - user_representation: Tensor of shape (batch_size, embed_dim),
                               the final user representation.
        - attention_weights: Dictionary containing:
            - 'multi_head_attention': List of tensors of shape (batch_size, num_news, num_news) per head,
                                      attention weights for each head from the MultiHeadAdditiveAttention layer.
            - 'user_attention': Tensor of shape (batch_size, num_news),
                                attention weights for the news articles from the UserAdditiveAttention layer.
        """
        enhanced_news_embeddings, multi_head_attention_weights = self.multi_head_attention(news_embeddings)
        user_representation, user_attention_weights = self.user_attention(enhanced_news_embeddings)

        attention_weights = {
            'multi_head_attention': multi_head_attention_weights,
            'user_attention': user_attention_weights
        }
        return user_representation, attention_weights

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClickPredictor(nn.Module):
    """
    Click Predictor Module: Predicts the probability of a user clicking on a candidate news article.
    Uses a dot product between the user representation and the candidate news representation.
    """
    def __init__(self):
        super(ClickPredictor, self).__init__()
        # No additional parameters are needed; the click probability
        # is computed using the dot product followed by a sigmoid activation.
    
    def forward(self, user_repr, candidate_news_repr):
        """
        Forward pass for click prediction.
        
        Args:
            user_repr (Tensor): User representation tensor of shape (batch_size, embedding_dim).
            candidate_news_repr (Tensor): Candidate news representation tensor of shape (batch_size, embedding_dim).
    
        Returns:
            click_prob (Tensor): Click probabilities tensor of shape (batch_size, 1), with values in [0, 1].
        """
        # Ensure the embedding dimensions match
        assert user_repr.size(1) == candidate_news_repr.size(1), "Embedding dimensions must match!"
        
        # Step 1: Compute the dot product between user and candidate news representations
        # This measures the similarity between user interests and news content
        click_score = torch.sum(user_repr * candidate_news_repr, dim=1, keepdim=True)  # Shape: (batch_size, 1)
    
        # Step 2: Apply a sigmoid activation function to get probabilities in [0, 1]
        click_prob = torch.sigmoid(click_score)  # Shape: (batch_size, 1)
    
        # Step 3: Return the click probabilities
        return click_prob

# Example usage
if __name__ == "__main__":
    # Simulate example inputs
    batch_size = 5      # Number of user-news pairs in a batch
    embedding_dim = 300 # Embedding dimension matching the NewsEncoder and UserEncoder outputs
    
    # Random user and news embeddings
    user_repr = torch.rand(batch_size, embedding_dim)          # Shape: (batch_size, embedding_dim)
    candidate_news_repr = torch.rand(batch_size, embedding_dim) # Shape: (batch_size, embedding_dim)
    
    # Instantiate the Click Predictor
    click_predictor = ClickPredictor()
    
    # Forward pass: Compute click probabilities
    click_probs = click_predictor(user_repr, candidate_news_repr)
    
    # Print the results
    print("User Representations Shape:", user_repr.shape)
    print("Candidate News Representations Shape:", candidate_news_repr.shape)
    print("Click Probabilities Shape:", click_probs.shape)
    print("Click Probabilities:\n", click_probs)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class NRMS(nn.Module):
    """
    NRMS Model: News Recommendation Model using Multi-Head Additive Attention
    and Click Prediction.
    """
    def __init__(self, embedding_matrix, embed_dim, num_heads, query_dim, dropout=0.2):
        super(NRMS, self).__init__()
        
        # News Encoder: Encodes the news articles
        self.news_encoder = NewsEncoder(
            embedding_matrix=embedding_matrix,
            embed_dim=embed_dim,
            num_heads=num_heads,
            query_dim=query_dim,
            dropout=dropout
        )
        
        # User Encoder: Encodes the user based on news interactions
        self.user_encoder = UserEncoder(embed_dim=embed_dim, num_heads=num_heads)
        
        # Click Predictor: Predicts the likelihood of a click based on user and news representations
        self.click_predictor = ClickPredictor()

    def forward(self, user_history, candidate_news):
        """
        Args:
            user_history (Tensor): Tensor of shape (batch_size, history_len),
                                    representing the IDs of news articles that the user has interacted with.
            candidate_news (Tensor): Tensor of shape (batch_size, num_news),
                                      representing the candidate news articles to predict clicks for.
        
        Returns:
            click_prob (Tensor): Tensor of shape (batch_size, 1), representing the click probabilities.
        """
        # Step 1: Get news representations
        news_repr, _ = self.news_encoder(candidate_news)  # Get the representation of the candidate news

        # Step 2: Get user representation
        user_repr, _ = self.user_encoder(user_history)  # Get the representation of the user based on their history
        
        # Step 3: Predict click probability
        click_prob = self.click_predictor(user_repr, news_repr)  # Get the click probability

        return click_prob


# Example usage:
if __name__ == "__main__":
    # Define the embedding matrix (this should be pre-trained word embeddings)
    embedding_matrix = torch.rand(10000, 300)  # Example: 10,000 words, each with 300-dimensional embeddings
    
    # Parameters
    batch_size = 5        # Number of users in a batch
    history_len = 10      # Number of news articles in user history
    num_news = 20         # Number of candidate news articles for each user
    embed_dim = 300       # Embedding dimension
    num_heads = 8         # Number of attention heads
    query_dim = 64        # Dimension for additive attention
    
    # Simulate example inputs
    user_history = torch.randint(0, 10000, (batch_size, history_len))  # User history (batch_size, history_len)
    candidate_news = torch.randint(0, 10000, (batch_size, num_news))  # Candidate news (batch_size, num_news)
    
    # Instantiate the NRMS model
    nrms_model = NRMS(
        embedding_matrix=embedding_matrix,
        embed_dim=embed_dim,
        num_heads=num_heads,
        query_dim=query_dim,
        dropout=0.2
    )
    
    # Forward pass: Compute click probabilities
    click_probs = nrms_model(user_history, candidate_news)
    
    # Print the results
    print("User History Shape:", user_history.shape)
    print("Candidate News Shape:", candidate_news.shape)
    print("Click Probabilities Shape:", click_probs.shape)
    print("Click Probabilities:\n", click_probs)


# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# Update hyperparameters based on the provided values
embed_dim = 300          # Dimension of word embeddings (GloVe)
num_heads = 15         # Number of self-attention heads
head_dim = 16            # Each attention head output dimension
query_dim = 200          # Dimension of additive attention query
dropout = 0.2            # 20% dropout on embeddings
learning_rate = 0.001    # Learning rate for Adam optimizer
batch_size = 64          # Batch size for training
negative_sampling_ratio = 4  # Negative sampling ratio K
epochs = 10              # Number of training epochs


# %%
class NRMS(nn.Module):
    def __init__(self, embedding_matrix, embed_dim, num_heads, query_dim, dropout=0.2):
        super(NRMS, self).__init__()
        self.news_encoder = NewsEncoder(embedding_matrix=embedding_matrix, embed_dim=embed_dim, num_heads=num_heads, query_dim=query_dim, dropout=dropout)
        self.user_encoder = UserEncoder(embed_dim=embed_dim, num_heads=num_heads)
        self.click_predictor = ClickPredictor()

    def forward(self, user_history, candidate_news):
        news_repr, _ = self.news_encoder(candidate_news)
        user_repr, _ = self.user_encoder(user_history)
        click_prob = self.click_predictor(user_repr, news_repr)
        return click_prob


# %%
# Custom tokenizer that handles OOV tokens
def tokenize_with_unk(text, tokenizer, unk_token_id=1):
    # Tokenize the text using the tokenizer
    encoding = tokenizer.encode(text)
    token_ids = encoding.ids  # Token IDs from the tokenizer
    
    # Replace any token not found in the vocabulary with the `<|unknown|>` token ID
    token_ids = [token_id if token_id in tokenizer.get_vocab() else unk_token_id for token_id in token_ids]
    
    return token_ids

# Clip token IDs to ensure they are within the bounds of the vocabulary size
def clip_token_ids(tensor, vocab_size, unk_token_id=1):
    # Ensure no index is out of range
    tensor = torch.clamp(tensor, min=0, max=vocab_size - 1)
    
    # If any token ID is out of bounds, replace it with the unknown token ID
    tensor[tensor >= vocab_size] = unk_token_id
    
    return tensor

# Modify the `prepare_data` function accordingly
def prepare_data(df, vocab_size, tokenizer, unk_token_id=1):
    user_histories = []
    candidate_news = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Preparing data"):
        user_history = row['article_ids_clicked']  # List of clicked article IDs
        candidate_news_id = row['article_id']  # Current candidate article ID
        
        # Tokenize user history: Assuming `article_ids_clicked` contains text that needs tokenization
        user_history_token_ids = [tokenize_with_unk(article_id, tokenizer, unk_token_id) for article_id in user_history]

        # Pad user history if it's shorter than the required length (max length = 10 for example)
        user_history_padded = user_history_token_ids[:10] + [unk_token_id] * (10 - len(user_history_token_ids))  # Max length 10 for history

        # Clip user history to be within bounds of the vocabulary
        user_history_padded = clip_token_ids(torch.tensor(user_history_padded, dtype=torch.long), vocab_size, unk_token_id)

        # Tokenize and pad candidate news article IDs
        candidate_news_token_id = tokenize_with_unk(candidate_news_id, tokenizer, unk_token_id)
        candidate_news_padded = [candidate_news_token_id] * 20  # Assuming you have 20 candidate news articles
        candidate_news_padded = clip_token_ids(torch.tensor(candidate_news_padded, dtype=torch.long), vocab_size, unk_token_id)

        # Append the history and candidate news
        user_histories.append(user_history_padded)
        candidate_news.append(candidate_news_padded)

    # Convert lists to tensors
    user_histories_tensor = torch.stack(user_histories)
    candidate_news_tensor = torch.stack(candidate_news)

    return user_histories_tensor, candidate_news_tensor

# Ensure the token IDs are within the correct range
print("Max token ID in user history:", train_user_history.max().item())
print("Max token ID in candidate news:", train_candidate_news.max().item())
print("Vocabulary size:", len(glove_vocabulary))  # Ensure this matches

# Assert that the max token IDs are within bounds
assert train_user_history.max().item() < len(glove_vocabulary), "User history contains out-of-vocabulary tokens!"
assert train_candidate_news.max().item() < len(glove_vocabulary), "Candidate news contains out-of-vocabulary tokens!"


# %%
# Check the max token ID in train_user_history
print("Max user history token ID:", train_user_history.max().item())
print("Vocabulary size:", len(glove_vocabulary))  # Check the vocabulary size


# %%
# Initialize model with GloVe embeddings (assume glove_vectors are loaded as done previously)
embedding_matrix = glove_vectors
model = NRMS(embedding_matrix=embedding_matrix, embed_dim=embed_dim, num_heads=num_heads, query_dim=query_dim, dropout=dropout)

# Optimizer and Loss Function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for click prediction

# Training Loop
for epoch in range(epochs):
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Zero gradients before backward pass
    
    # Forward pass: Compute click probabilities
    click_probs = model(train_user_history, train_candidate_news)
    
    # Assuming you have a column for actual clicks: 'clicked' or similar (1 for clicked, 0 for not clicked)
    click_labels = torch.tensor(df_train['clicked'].values, dtype=torch.float32).unsqueeze(1)  # Adjust if necessary

    # Compute loss
    loss = criterion(click_probs, click_labels)
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")



