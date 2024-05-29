import pandas as pd
import os, pickle
import time
import re
import torch.nn as nn
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from collections import Counter
import re
import faiss
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import openai
#from data_augmentation import find_augmented_sentences_context
STOP_WORDS = set(stopwords.words('english'))
vectorizer = SentenceTransformer('all-mpnet-base-v2')

def encode_desirable_label(label):
    if label == "desirable":
        label = 1
    elif label == "undesirable":
        label = 0
    else:
        raise "encode desirable label error"
    return label

def clean_text(text):
    if not isinstance(text, str):
        text = ""

    text = re.sub(r'[\n|\r]', ' ', text)
    # remove space between ending word and punctuations
    text = re.sub(r'[ ]+([\.\?\!\,]{1,})', r'\1 ', text)
    # remove duplicated spaces
    text = re.sub(r' +', ' ', text)
    # add space if no between punctuation and words
    text = re.sub(r'([a-z|A-Z]{2,})([\.\?\!]{1,})([a-z|A-Z]{1,})', r'\1\2\n\3', text)
    # handle case "...\" that" that the sentence spliter cannot do
    text = re.sub(r'([\?\!\.]+)(\")([\s]+)([a-z|A-Z]{1,})', r'\1\2\n\3\4', text)
    # remove space between letter and punctuation
    text = re.sub(r'([a-z|A-Z]{2,})([ ]+)([\.\?\!])', r'\1\3', text)
    # handle case '\".word' that needs space after '.'
    text = re.sub(r'([\"\']+\.)([a-z|A-Z]{1,})', r'\1\n\2', text)
    # handle case '.\"word' that needs space after '\"'
    text = re.sub(r'(\.[\"\']+)([a-z|A-Z]{1,})', r'\1\n\2', text)
    # text = re.sub('\n', ' ', text)
    text = text.strip()
    text = text.lower()

    if len(text) > 0 and text[-1].isalpha():
        text += "."
    return text

def get_file_paths(path):
    paths = glob(os.path.join(path, "*.xlsx"))
    return paths

def clean_df_data(df):
    # remove space before or after strings
    for col_name in df.columns:
        df[col_name] = df[col_name].apply(lambda x: x.strip() if isinstance(x, str) else x)
    # replace typos
    if "Unnamed: 7" in df.columns:
        df["Unnamed: 7"] = df["Unnamed: 7"].apply(lambda x: x if isinstance(x, str) else x)
    if "Unnamed: 8" in df.columns:
        df["Unnamed: 8"] = df["Unnamed: 8"].apply(lambda x: x.replace("releveant", "relevant").replace("not relevant", "irrelevant").replace("CLE", "LCE") if isinstance(x, str) else x)
    if "coarse_labels" in df.columns:
        df["coarse_labels"] = df["coarse_labels"].apply(lambda x: x.replace("surace", "surface") if isinstance(x, str) else x)
    return df

def extract_sentence(text):
    text = text + "\n"
    re_template = r'[^a-zA-Z0-9]*(.*?\n)'
    matches = re.findall(re_template, text)
    matches = [clean_text(x) for x in matches]
    match_str = "\n".join(matches)
    # match_str = text.replace("\n", " ")
    if len(match_str) == 0:
        match_str = clean_text(text)
    return match_str

def mean_pooling(token_embeddings, attention_mask):
            input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)