import ast # for evaluating string representation of lists
import nltk
import string
import itertools # for chaining ranges 
import transformers # for using DistilBert
import numpy as np
import pandas as pd
import pickle as pkl
from scipy import sparse
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# loading the dataset
df = pd.read_csv("../data/Clothing_Shoes_and_Jewelry_5.csv")

# Due to RAM constrainits it was not possible to get the embeddings for the entire
# dataframe so a sample of 50% of the dataset was used
# However note that this scrip was run on a Google Cloud VM with 406Gb of RAM
# to run this locall a smaller sample must be used (e.g. 1%)
df = df.sample(frac=0.50, random_state=42).reset_index(drop=True)

df = df.fillna("")

ascii_ranges = itertools.chain(range(32), range(128, 257)) # join both ranges together

# Create a lookup table specifying that non-ascii chars should be translated to empty strings
# This table will be fed to str.translate
# str.translate uses C-level table lookup so it's much faster than iterating over every char and replacing them
replacement_tbl = dict.fromkeys(ascii_ranges, u"")
    
def clean_sentence(sentence):
    """
        Given a sentence remove its punctuation, stop words and stem every word
        
        Arguments:
            sentence - the input string to clean
            
        Returns: The input string stripped out of punctuation and stopwords
    """
    
    stop_words = set(stopwords.words('english'))
    #porter = PorterStemmer()
    
    try:
        sentence = sentence.lower().translate(replacement_tbl) # Set the text to lower case and remove non-ascii chars
        sentence = sentence.translate(str.maketrans('','',string.punctuation)) # remove punctuation
        tokens = word_tokenize(sentence)
        cleaned_s = [w for w in tokens if w not in stop_words] # remove stop-words and stem words
        return " ".join(cleaned_s)
    
    except (AttributeError, TypeError):
        raise AssertionError("Input variable 's' should be of type string")

df["summary"] = df["summary"].apply(clean_sentence)

# Evaluating the string representation of the list to convert it to an actual list
df["helpful"] = df["helpful"].apply(lambda str_list: ast.literal_eval(str_list))

# Splitting this way is faster the applying a fuction to every row to get the first and second element
df[['num_yes_votes','num_votes']] = pd.DataFrame(df["helpful"].values.tolist(), index= df.index)

# Avoid dividing by zero with np.where
df["prop_yes_votes"] = np.where(df["num_votes"] == 0, 0, df["num_yes_votes"] / df["num_votes"])

df["num_words"] = df["reviewText"].str.count("\w+")

# Create new feature that is the deviation of the users overall review to the product
# from that products overall mean
product_overall_mean = df.groupby("asin")['overall'].mean().to_dict()
df['product_overall_mean'] = df['asin'].map(product_overall_mean).values
df['overall_deviation'] = df['overall'] - df['product_overall_mean']

# Creating the binary target label indicating if a review was helpful or not
df["helpful_review"] = ((df["num_votes"] >= 4) & (df["prop_yes_votes"] >= 0.5)).astype(int)

# ---DistilBERT PART---

# Loading pretrained model/tokenizer
# This is the Distilled, base, uncased version of BERT 
tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")

# Tokenize the sentences adding the special "[CLS]" and "[SEP]" tokens
tokenized = df["summary"].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

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

# Remove lines with empty strings
df_encoded = df_encoded[~(df["summary"] == "").any(axis=1)]
df = df[~(df["summary"] == "").any(axis=1)]

# Reset index as a sanity check
df_encoded = df_encoded.reset_index(drop=True)
df = df.reset_index(drop=True)

# Re-add columns
df_encoded["helpful_review"] = df["helpful_review"]
df_encoded["original_summary"] = df["summary"]
df_encoded["num_words"] = df["num_words"]
df_encoded["overall"] = df["overall"]
df_encoded["overall_deviation"] = df["overall_deviation"]

# Save file as pickle format
with open("bert_encoded_summary.pkl", "wb") as file_out:
    pkl.dump(df_encoded, file_out)