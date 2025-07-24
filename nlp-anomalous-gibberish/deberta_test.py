import pandas as pd
import torch
from transformers import DebertaTokenizer, DebertaForSequenceClassification
 
# Load the tokenizer and model
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-v3-base')
model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-v3-base')