from transformers import AutoTokenizer
from torch.utils.data import Dataset
from utilities import *
from nltk.tokenize import sent_tokenize
import pandas as pd
import re, time

class TextDataset(Dataset):
    def __init__(self, args, data):
        ## Double IDs because we are training on draft one and two
        self.essay_ids = data["essay_id"].apply(lambda x: int(x)).tolist()
        self.essay_ids += self.essay_ids

        ## Get both sets of drafts, clean the text
        # self.first_draft = data["first_draft"].apply(lambda x: sent_tokenize(clean_text(x))).tolist()
        # self.second_draft = data["second_draft"].apply(lambda x: sent_tokenize(clean_text(x))).tolist()
        self.first_draft = data["first_draft"].apply(lambda x: clean_text(x)).tolist()
        self.second_draft = data["second_draft"].apply(lambda x: clean_text(x)).tolist()
        self.drafts = self.first_draft + self.second_draft

        ## Get the article for comparison
        article = re.sub("\n", "", " ".join(open("test_data/article.txt").readlines()))

        ## Get gold scores
        self.gold_score_first = data["Evidence Before (Avg of R1 R2)"].apply(lambda x: x-1).tolist()
        self.gold_score_second = data["Evidence After (Avg of R1 R2)"].apply(lambda x: x-1).tolist()
        self.gold_scores = self.gold_score_first + self.gold_score_second

        ## Ready for tokenization
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")
        self.tokens = []

        for draft in self.drafts:
            tk = self.tokenizer(draft, padding="max_length", truncation=True, return_tensors="pt")
            self.tokens.append(tk)

    def __len__(self):
        return len(self.drafts)
    
    def __getitem__(self, idx):
        es_id = self.essay_ids[idx]
        token = self.tokens[idx]
        label = self.gold_scores[idx]

        return es_id, token, label
