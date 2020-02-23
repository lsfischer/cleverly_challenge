# importing required libraries
import ast # for evaluating string representation of lists
import nltk
import torch
import string
import itertools # for chaining ranges 
import transformers # for using DistilBert
import numpy as np
import pandas as pd
from scipy import sparse
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# loading the dataset
df = pd.read_csv("../data/Clothing_Shoes_and_Jewelry_5.csv")

df = df.sample(frac=0.15, random_state=42)

# Convert unix timestamp to datetime object
df.drop(["unixReviewTime", "asin", "overall", "reviewTime", "reviewerID",
"reviewerName", "summary"], axis=1, inplace=True)

df = df.fillna("")

def clean_sentence(s):
    """
        Given a sentence remove its punctuation and stop words
        
        Arguments:
            s - the input string to clean
            
        Returns: The input string stripped out of punctuation and stopwords
    """
    ascii_ranges = itertools.chain(range(32), range(128, 257))

    # Create a lookup table specifying that non-ascii chars should be translated to empty strings
    # This table will be fed to str.translate
    # str.translate uses C-level table lookup so it's much faster than iterating over every char and replacing them
    tbl = dict.fromkeys(ascii_ranges, u"")
    
    stop_words = set(stopwords.words('english'))

    #porter = PorterStemmer()

    s = s.lower().translate(tbl) # Set the text to lower case and remove non-ascii chars
    s = s.translate(str.maketrans('','',string.punctuation)) # remove punctuation
    tokens = word_tokenize(s)
    cleaned_s = [w for w in tokens if w not in stop_words] # remove stop-words and stem regular words
    return " ".join(cleaned_s[:21])

# Apply clean_sentence to reviewText and summary; to reviewerName just remove non-ascii chars
#df["reviewText"] = df["reviewText"].apply(clean_sentence)
df["reviewText"] = df["reviewText"].apply(clean_sentence)

# Evaluating the string representation of the list to convert it to an actual list
df["helpful"] = df["helpful"].apply(lambda str_list: ast.literal_eval(str_list))

# Splitting this way is faster the applying a fuction to every row to get the first and second element
df[['num_yes_votes','num_votes']] = pd.DataFrame(df["helpful"].values.tolist(), index= df.index)

# Avoid dividing by zero with np.where
df["prop_yes_votes"] = np.where(df["num_votes"] == 0, 0, df["num_yes_votes"] / df["num_votes"])

# Creating the binary target label indicating if a review was helpful or not
df["helpful_review"] = ((df["num_votes"] >= 4) & (df["prop_yes_votes"] >= 0.5)).astype(int)

df.drop(["prop_yes_votes", "num_yes_votes", "num_votes", "helpful"], axis=1, inplace=True)


# Loading pretrained model/tokenizer
# This is the Distilled, base, uncased version of BERT 
tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")

# Tokenize the sentences adding the special "[CLS]" and "[SEP]" tokens
tokenized = df["reviewText"].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

# Get the length of the longest tokenized sentence
max_len = tokenized.apply(len).max() 

# Padd the rest of the sentence with zeros if the sentence is smaller than the longest sentence
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values]) 

# Create the attention mask so BERT knows to ignore the zeros used for padding
attention_mask = np.where(padded != 0, 1, 0)

# Create the input tensors
input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

# Pass the inputs through DistilBERT
with torch.no_grad():
    encoder_hidden_state = model(input_ids, attention_mask=attention_mask)

# Create a new dataframe with the encoded features
df_encoded = pd.DataFrame(encoder_hidden_state[0][:,0,:].numpy())

# Insert the original columns in the beginning of the encoded dataframe
df_encoded.insert(loc=0, column='original_reviewText', value=df["reviewText"])
df_encoded.insert(loc=0, column='helpful_review', value=df["helpful_review"])

# Download the encoded csv
df_encoded.to_csv("./bert_encoded_reviewText_15.csv", index=False)

####### CANT LOAD IT INTO MEMORY, JUST TAKE A SMALL SAMPLE LIKE 5% OR SOMETHING LIKE THAT