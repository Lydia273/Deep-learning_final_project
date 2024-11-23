#!/usr/bin/env python
# coding: utf-8

# The demo and small datasets provided by EB-Nerd are subsets of the full dataset, designed for different levels of experimentation and prototypig.
# 
# 
# - We will use the demo dataset for the beggining in order to develop our model. and to quickly validate our core or preprocessing pipeline.
# 
# - Then we will use the small dataset to verify that our code works the demo dataset, because it is a more representative subset for training an evaluating the models .
# 
# - The large dataset, requires significant computational resources, and it is more time consuming to process so we should used it only after confirming the pipeline works correctly with smaller datasets.

# # Let's sarts with all the essential preprocessing steps required for the NRMS model and incorporates the data cleaning.

# In[1]:


# Step 1: Import Libraries
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Instantiate the LabelEncoder
label_encoder = LabelEncoder()


# In[2]:


# Define paths for the datasets based on the current working directory
base_dir = os.path.join(os.getcwd())
behavior_file = os.path.join(base_dir,'behaviors.parquet')
history_file = os.path.join(base_dir,'history.parquet')
articles_file = os.path.join(base_dir, 'articles.parquet')


# In[3]:


# Check if the paths exist and print them
paths = [behavior_file, history_file, articles_file]

for path in paths:
    if os.path.exists(path):
        print(f"Path exists: {path}")
    else:
        print(f"Path does NOT exist: {path}")


# Read the files

# In[4]:


# Step 3: Load Datasets
# Load each dataset from the specified paths
behavior_df = pd.read_parquet(behavior_file)
history_df = pd.read_parquet(history_file)
articles_df = pd.read_parquet(articles_file)

# Display the first few rows of each dataset to confirm loading worked
print("Behavior Data:")
print(behavior_df.head())

print("\nHistory Data:")
print(history_df.head())

print("\nArticles Data:")
print(articles_df.head())


# ### Check for missing values

# In[5]:


# Check for missing values
print(behavior_df.isnull().sum())


# In[6]:


print(behavior_df.columns)


# We observed that we have a lot of missing values for the gender , age and postcode attributes and since they are not relevant for our NRMS we drop them.

# In[7]:


# Now drop columns with high missing values
behavior_df = behavior_df.drop(columns=['impression_id', 'article_id','gender', 'postcode', 'age','scroll_percentage','device_type'])


# In[8]:


print(behavior_df.isnull().sum())


# In[9]:


print(behavior_df.columns)


# In[10]:


print(history_df.isnull().sum())


# In[11]:


print(articles_df.isnull().sum())


# In[12]:


# Now drop columns with high missing values
articles_df = articles_df.drop(columns=['image_ids'])


# In[13]:


print(articles_df.columns)


# In[14]:


print(articles_df.isnull().sum())


# In[15]:


print(articles_df.columns)


# Impute the missing values in the columns total_invies, total_pagevies, and total_read_time using K_Nearedt neighbors imputation method.

# In[16]:


# Calculate the percentage of missing values for each column
columns_to_check = ['total_inviews', 'total_pageviews', 'total_read_time']

missing_percentage = articles_df[columns_to_check].isna().mean() * 100

# Display the result
print("Percentage of missing values in the specified columns:")
print(missing_percentage)


# With approximately 36% missing values in these columns, imputing the missing values is reasonable as long as the imputation method aligns with the data's characteristics and intended usage.

# In[17]:


from sklearn.impute import KNNImputer
import pandas as pd

# Assuming `articles_df` is already loaded
print("Columns before imputing missing values:")
print(articles_df.isna().sum())

# Select the columns with missing values and prepare for KNN
columns_to_impute = ['total_inviews', 'total_pageviews', 'total_read_time']

# Ensure all columns are numeric before applying KNN
knn_data = articles_df[columns_to_impute].astype(float)

# Initialize KNN Imputer (using 5 nearest neighbors as default)
knn_imputer = KNNImputer(n_neighbors=5)

# Apply KNN Imputation
knn_imputed = knn_imputer.fit_transform(knn_data)

# Replace the imputed values back in the original dataframe
articles_df[columns_to_impute] = knn_imputed

# Verify that missing values are filled
print("Columns after imputing missing values:")
print(articles_df.isna().sum())


# ### Check how many rows our files have

# In[18]:


# Print the number of rows
print(f"Number of rows in behavior file: {len(behavior_df)}")
print(f"Number of rows in history file: {len(history_df)}")
print(f"Number of rows in articles file: {len(articles_df)}")


# # Modife the article dataset

# This code loads only the specified columns, converts subcategory_ids to a single string format, and applies LabelEncoder for model compatibility.

# In[19]:


# Now that we have the correct column names, let's proceed with the correct script

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Specify the correct columns based on the user's provided column names
articles_columns = ['article_id', 'title', 'subtitle', 'last_modified_time', 'premium',
       'body', 'published_time', 'article_type', 'ner_clusters',
       'entity_groups', 'topics', 'category', 'subcategory', 'category_str',
       'total_inviews', 'total_pageviews', 'total_read_time',
       'sentiment_score', 'sentiment_label']

# Load only the necessary columns from the articles file
articles_df = pd.read_parquet(articles_file, columns=articles_columns)

# Convert subcategory from list to a single string (assuming it's a list of IDs)
articles_df['subcategory'] = articles_df['subcategory'].apply(lambda x: ' '.join(map(str, x)))

# Apply LabelEncoder to transform 'subcategory' into a numeric format
label_encoder = LabelEncoder()
articles_df['subcategory_encoded'] = label_encoder.fit_transform(articles_df['subcategory'])

# Display the resulting DataFrame to verify changes
articles_df[['article_id', 'title', 'subtitle', 'last_modified_time', 'premium',
       'body', 'published_time', 'article_type', 'ner_clusters',
       'entity_groups', 'topics', 'category','subcategory_encoded', 'category_str',
       'total_inviews', 'total_pageviews', 'total_read_time',
       'sentiment_score', 'sentiment_label']].head()



# This code will:
# 
# Convert last_modified_time and published_time to milliseconds.
# Calculate the time_interval in milliseconds between the mod_time and pub_time.

# In[20]:


# Converting last_modified_time and published_time to milliseconds and calculating the interval

# Ensure the timestamps are in datetime format
articles_df['last_modified_time'] = pd.to_datetime(articles_df['last_modified_time'])
articles_df['published_time'] = pd.to_datetime(articles_df['published_time'])

# Calculate mod_time and pub_time in milliseconds
articles_df['mod_time'] = articles_df['last_modified_time'].astype('int64') // 10**6  # Convert to milliseconds
articles_df['pub_time'] = articles_df['published_time'].astype('int64') // 10**6  # Convert to milliseconds

# Calculate the time interval between last_modified_time and published_time in milliseconds
articles_df['time_interval'] = articles_df['mod_time'] - articles_df['pub_time']

# Display the resulting DataFrame to verify changes
articles_df.head()


# In[21]:


print(articles_df.columns)


# mod_time = the last modified time in milliseconds
# 
# pub_time = the published time in milliseconds
# 
# time_interval = mod_time - pub_time in milliseconds

# # Modify the behavior dataset

#  Add a feature (articles_num) for the count of in-view articles, and explode article_ids_inview to have one article per row for each impression.

# In[22]:


print(behavior_df.columns)


# ### Explanation of the script
# 1. subset of the data frame of the behavior_df (impression_time, article_ids_inview, user_id, and session_id)
# 2. Calculate articles_num , that represents the number of articles in the article_ids_invie list for each row.
# 3. Track row counts: Calculates the original row count of the dataset (original_row_count) and the expected number of rows after exploding (expected_row_count), which is the sum of the lengths of all article_ids_inview lists.
# 4. Define Chunk Size: Breaks the dataset into smaller chunks (chunk_size = 100,000) for efficient processing. The number of chunks is determined by dividing the total number of rows by the chunk size.
# 5. Process Each Chunk: For each chunk:
#     -    It explodes the article_ids_inview column, creating a new row for each article in the list.
#     -    Tracks the number of rows in the exploded chunk.
#     -    Saves the exploded chunk to a Parquet file with a unique filename.
# 6. Track Total Exploded Rows:
#     -    Tracks the total number of rows in all exploded chunks and verifies that it matches the expected_row_count.
# 
# 
# # Example: Exploding `article_ids_inview` Column in `behavior_df_new`
# 
# ## Input Data (`behavior_df_new`):
# 
# | impression_time | article_ids_inview     | user_id | session_id | articles_num |
# |------------------|------------------------|---------|------------|--------------|
# | 2023-11-01       | [101, 102, 103]       | A123    | S001       | 3            |
# | 2023-11-01       | [201]                 | A124    | S002       | 1            |
# | 2023-11-02       | [301, 302]            | A125    | S003       | 2            |
# 
# ---
# 
# ## Process: Exploding `article_ids_inview`
# 
# Each element in the `article_ids_inview` list becomes a new row, while other columns are duplicated for each new row.
# 
# ---
# 
# ## Output (Exploded Chunk):
# 
# | impression_time | article_ids_inview | user_id | session_id |
# |------------------|--------------------|---------|------------|
# | 2023-11-01       | 101                | A123    | S001       |
# | 2023-11-01       | 102                | A123    | S001       |
# | 2023-11-01       | 103                | A123    | S001       |
# | 2023-11-01       | 201                | A124    | S002       |
# | 2023-11-02       | 301                | A125    | S003       |
# | 2023-11-02       | 302                | A125    | S003       |
# 
# ---
# 
# ### Key Points:
# 1. **Original Data**:
#    - The dataset starts with 3 rows.
# 
# 2. **Exploded Data**:
#    - After exploding, the dataset has 6 rows since each article in the `article_ids_inview` list is moved to its own row.
# 
# 3. **Duplication**:
#    - Columns like `impression_time`, `user_id`, and `session_id` are duplicated for each new row corresponding to the articles.
# 
# ---
# 
# ### Why Explode the Data?
# - Exploding `article_ids_inview` helps in reshaping the dataset to make it easier to process each article individually for further analysis or modeling.
# 
# 

# In[23]:


import pandas as pd


# Step 1: Create a new DataFrame with only relevant columns and calculate articles_num for each row
behavior_df_new = behavior_df[['impression_time', 'article_ids_inview', 'read_time', 'user_id','article_ids_clicked','next_read_time', 'next_scroll_percentage', 'session_id']]
behavior_df_new['articles_num'] = behavior_df['article_ids_inview'].apply(len)  # Calculate articles_num first

# Track the original and expected row counts
original_row_count = len(behavior_df_new)
expected_row_count = behavior_df_new['articles_num'].sum()  # Expected row count after explosion
print(f"Original row count: {original_row_count}")
print(f"Expected row count after explosion: {expected_row_count}")

# Step 2: Define the chunk size
chunk_size = 100000
num_chunks = len(behavior_df_new) // chunk_size + 1  # Total number of chunks

# Step 3: Process and save each chunk
exploded_total_row_count = 0  # Track total rows after explosion
for i, start in enumerate(range(0, len(behavior_df_new), chunk_size)):
    # Select a chunk of the data
    chunk = behavior_df_new.iloc[start:start + chunk_size].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Explode the chunk to have one article per row
    exploded_chunk = chunk.explode('article_ids_inview')

    # Track row count for each exploded chunk
    exploded_chunk_row_count = len(exploded_chunk)
    exploded_total_row_count += exploded_chunk_row_count
    print(f"Chunk {i+1}: Original rows = {len(chunk)}, Exploded rows = {exploded_chunk_row_count}")

    # Save the exploded chunk to disk with a unique filename
    exploded_chunk.to_parquet(f"exploded_behavior_data_chunk_{i+1}.parquet", index=False)

print(f"Total exploded row count after processing all chunks: {exploded_total_row_count}")
print("All chunks have been processed and saved to disk.")



# - The original dataset is 24724 rows, which corresponds to the number of behavior_df= behavior_df_new before exploding.
# - The expected row count was calculated as 278,139, which is the sum of all articles_num values, indicating how many rows the dataset would have after exploding article_ids_inview.
# - After exploding, the chunk contained 278,139 rows, as expected.
# - Saved output

# In[24]:


import pyarrow.parquet as pq
import pyarrow as pa
import glob

# Define output file
output_file = "full_behavior_data.parquet"

# Initialize variables
first_chunk = True

# Initialize the Parquet writer only once, for the first chunk
for file in glob.glob("exploded_behavior_data_chunk_*.parquet"):
    # Read the chunk
    chunk = pq.read_table(file)

    # Write the first chunk to initialize the file with schema
    if first_chunk:
        # Open the Parquet writer with the schema of the first chunk
        writer = pq.ParquetWriter(output_file, chunk.schema)
        first_chunk = False

    # Write the current chunk to the output file
    writer.write_table(chunk)

# Close the writer after all chunks are written
writer.close()

print("All chunks have been merged into 'full_behavior_data.parquet'.")


# The above script merges all exploded chunks into a single parquet file, full_behavior_data .

# In[25]:


behavior_exploded_file = os.path.join(base_dir, 'full_behavior_data.parquet')
behavior_exploded = pd.read_parquet(behavior_exploded_file)


# In[26]:


# Print the number of rows
num_rows = len(behavior_exploded)
num_rows


# In[27]:


# Check row count before explosion (check if the data imported correctly)
original_rows = len(behavior_df)
print(f"Rows in original behavior_df: {original_rows}")

# Check row count after explosion
exploded_rows = len(behavior_exploded)
print(f"Rows in exploded behavior_df_new: {exploded_rows}")


# The merged full_behavior_data.parquet file contains 133,810,641 rows. This confirms that all chunks were successfully processed and combined into the final dataset.

# ### In the next cells i check if the article_ids_inview transformed correctly

# In[28]:


# here we just check the article_ids_inview to check the lists
#  print the first 100 rows for a sample view:
print(behavior_df['article_ids_inview'].head(100))  # Shows first 100 entries


# In[29]:


# Step-by-step code to calculate, sort, and display

# Calculate the number of articles in each list in 'article_ids_inview'
behavior_df['articles_num'] = behavior_df['article_ids_inview'].apply(len)

# Sort by 'articles_num' in ascending order
sorted_behavior_df = behavior_df.sort_values(by='articles_num')

# Display the first 100 rows of sorted results
print(sorted_behavior_df[['article_ids_inview', 'articles_num']].head(100))



# In[30]:


# Step 1: Verify original row count and the minimum value in articles_num
original_row_count = len(behavior_df)
min_articles_num = behavior_df['articles_num'].min()
print(f"Original row count: {original_row_count}")
print(f"Minimum articles_num value: {min_articles_num}")



# In[31]:


print(behavior_exploded.columns)


# Rename the article_ids_inview column of the behavior_exploded to article_id

# In[32]:


behavior_exploded.rename(columns={'article_ids_inview': 'article_id'}, inplace=True)


# Check if the names of the columns are correct

# In[33]:


print(behavior_exploded.columns)


# - Now we will create an impr_time that converts impression_time to an integer representation in milliseconds,
# - impr_pub_interval: calculates the difference between the impression and published time, giving the delay between publishing and the user's impression
# - impr_pub_hour that converts this interval into hours for easier groupin and analysis
# 

# How we do it:
# 1. Extract article_id from behavior_exploded ( after exploding article_ids_inview).
# 2. Merge behavior_exploded with articles_df on article_id.
# 3. Calculate impr_time, impr_pub_interval, and impr_pub_hour based on impression_time and published_time from the merged dataset.

# In[34]:


import pandas as pd

# Ensure the necessary columns are in datetime format
# Convert both 'impression_time' and 'published_time' to datetime if not already done
behavior_exploded['impression_time'] = pd.to_datetime(behavior_exploded['impression_time'])
articles_df['published_time'] = pd.to_datetime(articles_df['published_time'])


# Merge behavior_df with articles_df on article_id to bring in published_time information
merged_df = behavior_exploded.merge(articles_df[['article_id', 'published_time']], on='article_id', how='left')

# 1. Convert impression_time to an integer representation in milliseconds
merged_df['impr_time'] = merged_df['impression_time'].astype('int64') // 10**6  # Convert from nanoseconds to milliseconds

# 2. Calculate the delay (in milliseconds) between the impression and the published time
merged_df['impr_pub_interval_milliseconds'] = (merged_df['impression_time'] - merged_df['published_time']).dt.total_seconds() * 1000

# 3. Convert the impr_pub_interval to hours
merged_df['impr_pub_hour_interval'] = merged_df['impr_pub_interval_milliseconds'] / (1000 * 3600)  # Convert milliseconds to hours

# Display a sample to verify
print(merged_df[['impression_time', 'published_time', 'impr_time', 'impr_pub_interval_milliseconds', 'impr_pub_hour_interval']].head())


# ### Explaining the calculation of impr_pub_interval:
# 
# impr_pub_interval computes the delay in milliseconds between when the article was published (published_time) and when the impression occurred (impression_time).
# For example, if an article was published at 2023-11-13 08:00:00 and viewed at 2023-11-14 10:15:30, the impr_pub_interval might be 90090000 milliseconds (around 25 hours).
# 

# In[35]:


merged_df.head()


# ### Now i print the colmns of each dataset to check which columns i will merge.
# We have the behavior_df, the behavior_exploded, the articles_df, the history_df, and the merged_df.

# In[36]:


print(behavior_df.columns)


# In[37]:


print(behavior_exploded.columns)


# In[38]:


print(articles_df.columns)


# In[39]:


print(history_df.columns)


# In[40]:


print(merged_df.columns)


# In[41]:


# For behavior_df
print(f"Number of rows in behavior_df: {behavior_df.shape[0]}")

# For articles_df
print(f"Number of rows in articles_df: {articles_df.shape[0]}")

# For behavior_exploded
print(f"Number of rows in behavior_exploded: {behavior_exploded.shape[0]}")

# For history_df
print(f"Number of rows in history_df : {history_df.shape[0]}")

# For merged_df
print(f"Number of rows in merged_df: {merged_df.shape[0]}")


# ### Lets modify the history file a bit..

# In[42]:


history_df.head()


# - Current number of rows
# - Expected number of rows after the explosion
# - Perform the explosiojn in chunks for the history_df dataset.
# 

# In[43]:


import os
import pandas as pd

# Paths
output_folder = "history_chunks"
os.makedirs(output_folder, exist_ok=True)

# Step 1: Current number of rows
current_row_count = len(history_df)

# Step 2: Expected row count after explosion
expected_row_count = history_df['article_id_fixed'].apply(len).sum()

print(f"Original row count: {current_row_count}")
print(f"Expected row count after explosion: {expected_row_count}")

# Step 3: Chunk processing
chunk_size = 5000  # Adjust based on available memory
total_exploded_row_count = 0
num_chunks = (len(history_df) // chunk_size) + 1

for i, start in enumerate(range(0, len(history_df), chunk_size)):
    # Select chunk
    chunk = history_df.iloc[start:start + chunk_size].copy()

    # Explode the chunk
    exploded_chunk = chunk.explode(
        ['impression_time_fixed', 'scroll_percentage_fixed', 'article_id_fixed', 'read_time_fixed']
    )
    exploded_chunk.reset_index(drop=True, inplace=True)

    # Track rows
    total_exploded_row_count += len(exploded_chunk)
    print(f"Chunk {i+1}: Original rows = {len(chunk)}, Exploded rows = {len(exploded_chunk)}")

    # Save exploded chunk to file
    output_file = os.path.join(output_folder, f"history_chunk_{i+1}.parquet")
    exploded_chunk.to_parquet(output_file, index=False)

# Final stats
print(f"Total exploded row count after processing all chunks: {total_exploded_row_count}")
print("All chunks have been processed and saved to disk.")


# In[44]:


import os
import pandas as pd
import glob

# Define input folder and output file
input_folder = "history_chunks"
output_file = "merged_history_data.parquet"

# Find all chunk files in the folder
chunk_files = glob.glob(os.path.join(input_folder, "*.parquet"))

# Initialize an empty list to collect DataFrames
df_list = []

# Read each chunk file and append it to the list
for file in chunk_files:
    print(f"Reading chunk file: {file}")
    chunk_df = pd.read_parquet(file)
    df_list.append(chunk_df)

# Concatenate all DataFrames into a single DataFrame
merged_history_df = pd.concat(df_list, ignore_index=True)

# Save the merged DataFrame to a single Parquet file
merged_history_df.to_parquet(output_file, index=False)

print(f"All chunks have been merged into '{output_file}'.")


# In[45]:


import pandas as pd

# Define the file path
merged_history_file = "merged_history_data.parquet"

# Read the merged Parquet file
history_exploded = pd.read_parquet(merged_history_file)

# Display the number of rows and columns
print(f"Number of rows: {len(history_exploded)}")
print(f"Number of columns: {len(history_exploded.columns)}")

# Display the column names
print("Column names:", history_exploded.columns.tolist())

# Display the first few rows of the DataFrame
history_exploded.head()


# Certainly! Below is the content formatted for a Jupyter Notebook with markdown and code cells.
# 
# ---
# 
# ### Description of Datasets
# 
# #### 1. **`behaviors.parquet`**
# - **Purpose:** Captures the behavior logs for a specific 7-day period.
# - **Contents:**
#   - Articles shown to the user (`article_ids_inview`).
#   - Articles clicked by the user (`article_ids_clicked`).
#   - Impression time for when the articles were shown to the user (`impression_time`).
#   - Other metadata such as:
#     - `user_id`
#     - `session_id`
#     - Engagement details like `scroll_percentage`.
# 
# ---
# 
# #### 2. **`history.parquet`**
# - **Purpose:** Contains the click history for users over the 21 days prior to the behavior period.
# - **Contents:**
#   - Articles previously clicked by the user (`article_id_fixed`).
#   - Timestamp for when each article was clicked (`impression_time_fixed`).
#   - Engagement details such as:
#     - `read_time_fixed`
#     - `scroll_percentage_fixed`.
# 
# ---
# 
# ### Key Points
# 
# 1. **Timeframes:**
#    - `behaviors.parquet` covers a **7-day period**.
#    - `history.parquet` contains **21 days** of user click history prior to the behavior period.
#    - The click histories in `history.parquet` are **static** and are **not updated** during the 7-day behavior period.
# 
# 2. **Relationships:**
#    - The `behaviors.parquet` dataset reflects **user interactions within the 7-day split period**.
#    - The `history.parquet` dataset provides **context about a user’s prior behavior**.
# 
# ---
# 
# ### Why This is Important
# 
# 1. **Static Nature of `history.parquet`:**
#    - The `history.parquet` dataset serves as context for understanding user preferences **before the behavior period begins**.
#    - It is **not influenced** by or updated with interactions during the behavior period in `behaviors.parquet`.
# 
# 2. **Merging Behavior and History:**
#    - By combining these datasets, we can analyze:
#      - **What influenced a user's clicks** in the behavior logs (based on prior history).
#      - **Patterns in user engagement** over time.
# 
# ---
# 
# ### Expected Outputs
# 
# - **Columns in `behaviors.parquet`:**
#   - `impression_time`, `article_ids_inview`, `article_ids_clicked`, `user_id`, `session_id`, `scroll_percentage`, etc.
# 
# - **Columns in `history.parquet`:**
#   - `impression_time_fixed`, `article_id_fixed`, `scroll_percentage_fixed`, `read_time_fixed`, `user_id`.
# 
# ---
# 
# 

# In[46]:


# For behavior_df
print(f"Number of rows in behavior_df: {behavior_df.shape[0]}")

# For articles_df
print(f"Number of rows in articles_df: {articles_df.shape[0]}")

# For behavior_exploded
print(f"Number of rows in behavior_exploded: {behavior_exploded.shape[0]}")

# For history_df
print(f"Number of rows in history_df : {history_df.shape[0]}")

# For history_exploded
print(f"Number of rows in history_exploded : {history_exploded.shape[0]}")

# For merged_df
print(f"Number of rows in merged_df: {merged_df.shape[0]}")


# In[47]:


print(behavior_df.columns)


# In[48]:


print(behavior_exploded.columns)


# In[49]:


print(articles_df.columns)


# In[50]:


print(merged_df.columns)


# In[51]:


print(articles_df.columns)


# In[52]:


import pandas as pd

#Concatenate the specified columns from articles_df into merged_df based on article_id
merged_df = merged_df.merge(
     articles_df[['article_id','title','subtitle','mod_time','pub_time','time_interval','article_type','topics','category','subcategory_encoded','total_inviews', 'total_pageviews', 'total_read_time',
       'sentiment_score', 'sentiment_label']],
    on='article_id',
    how='left'
)

# Display a sample of the updated merged_df to verify
print(merged_df.head())


# In[53]:


print(merged_df.columns)


# In[54]:


# Now drop columns with high missing values
merged_df = merged_df.drop(columns=['impression_time','published_time','impr_time','subtitle', 'mod_time', 'pub_time','article_type',
       'topics', 'category'])


# In[55]:


print(merged_df.head())


# In[56]:


# Print the number of rows in merged_df
num_rows = len(merged_df)
print(f"The number of rows in merged_df: {num_rows}")


# In[57]:


print(history_exploded.columns)


# In[58]:


# Inspect lengths and unique user IDs
merged_df_length = len(merged_df)
history_exploded_length = len(history_exploded)

# Count unique user_ids in both datasets
unique_user_ids_merged = merged_df['user_id'].nunique()
unique_user_ids_history = history_exploded['user_id'].nunique()

# Find the intersection and differences of user_ids
common_user_ids = set(merged_df['user_id']).intersection(set(history_exploded['user_id']))
only_in_merged = set(merged_df['user_id']) - set(history_exploded['user_id'])
only_in_history = set(history_exploded['user_id']) - set(merged_df['user_id'])

expected_length = merged_df_length  # Left join means all rows in merged_df will remain

{
    "merged_df_length": merged_df_length,
    "history_exploded_length": history_exploded_length,
    "unique_user_ids_merged": unique_user_ids_merged,
    "unique_user_ids_history": unique_user_ids_history,
    "common_user_ids_count": len(common_user_ids),
    "only_in_merged_count": len(only_in_merged),
    "only_in_history_count": len(only_in_history),
    "expected_length": expected_length
}


# In[59]:


# Display information about the resulting dataset
print(f"Length of merged_df: {len(merged_df)}")
print(f"Length of history_exploded: {len(history_exploded)}")


# In[61]:


import os

# Function to process a chunk with a slice of history_exploded
def process_chunk_optimized(chunk, history_file, output_dir, chunk_index):
    print(f"Processing chunk {chunk_index}...")

    # Load the slice of history_exploded for the user_ids in the current chunk
    user_ids = chunk['user_id'].unique()
    history_slice = history_exploded[history_exploded['user_id'].isin(user_ids)]

    # Merge the current chunk with the sliced history_exploded
    merged_chunk = chunk.merge(
        history_slice,
        on='user_id',
        how='left'  # Perform a left join to preserve all rows from chunk
    )

    # Fill NaN values with 0
    merged_chunk = merged_chunk.fillna(0)

    # Write the result to a new Parquet file for the chunk
    output_file = os.path.join(output_dir, f"chunk_{chunk_index}.parquet")
    merged_chunk.to_parquet(output_file, engine='pyarrow')
    print(f"Chunk {chunk_index} written to {output_file}.")

# Directory to store individual chunk files
output_dir = "output_chunks"
os.makedirs(output_dir, exist_ok=True)

chunk_size = 200_000  # Adjust chunk size based on memory availability

# Process merged_df in smaller chunks
chunks = [
    merged_df.iloc[i:i + chunk_size]
    for i in range(0, len(merged_df), chunk_size)
]

# Process each chunk with a corresponding history slice
for idx, chunk in enumerate(chunks):
    process_chunk_optimized(chunk, history_exploded, output_dir, idx + 1)

print(f"All chunks processed and saved in {output_dir}.")



# In[62]:


import glob

# Get all the chunk files
chunk_files = glob.glob(os.path.join(output_dir, "*.parquet"))

# Load and concatenate all chunks
final_df = pd.concat([pd.read_parquet(file) for file in chunk_files])

# Save to a single Parquet file
final_output_file = "final_merged_df_optimized.parquet"
final_df.to_parquet(final_output_file, engine='pyarrow')
print(f"Final dataset saved to {final_output_file}.")


# In[64]:


import pandas as pd

# Path to the Parquet file
file_path = "final_merged_df_optimized.parquet"

# Read the Parquet file
df = pd.read_parquet(file_path, engine='pyarrow')

# Display the DataFrame
print(df.head())  # Show the first few rows


# In[65]:


# Perform one-hot encoding
df_one_hot = pd.get_dummies(df, columns=['sentiment_label'], prefix='sentiment')

# Example output columns: sentiment_negative, sentiment_neutral, sentiment_positive
print(df_one_hot.head())

