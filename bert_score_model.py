from transformers import AutoModel, AutoTokenizer

AutoModel.from_pretrained("bert-base-uncased")
AutoTokenizer.from_pretrained("bert-base-uncased")

import nltk
nltk.download('wordnet')
nltk.download('punkt')
