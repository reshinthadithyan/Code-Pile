import re
import gensim
from gensim.utils import simple_preprocess
import numpy as np
from nltk.corpus import stopwords
import spacy
import pickle
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'also', 'take', 'make'])
import json

def remove_stopwords(words):
    return [word for word in words if word not in stop_words]

def lemmatization(words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    doc = nlp(" ".join(words))
    return [token.lemma_ for token in doc if token.pos_ in allowed_postags]

def preprocess(example):
    text = re.sub('[,\.!?]', '', example['text']).lower()
    text = gensim.utils.simple_preprocess(str(text), deacc=True)
    text = remove_stopwords(text)
    text = lemmatization(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    return {"processed_text": text}

import random
import gensim.corpora as corpora
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

id2word = pickle.load(open('lda_save/id2word_Enwiki', 'rb'))
lda_model = gensim.models.LdaMulticore.load('lda_save/lda_model_Enwiki')

all_cats = ['arXiv', 'PubMed', 'Books3', 'Gutenberg', 'FreeLaw_Options', 'UbuntuIRC', 'EuroParliamentProceedings',
            'DMMath', 'USPTO', 'PileOfLaw', 'ASFPublicMail', 'StackExchange',
            'CPDataset', 'USENET', 'AI4Code', 'Discourse', 'AMPS', 'LeetCode', 'OtherWiki', 'S2ORC', 'PileCC', 
            'USENET', 'OtherWiki', 'PubMed', 'UbuntuIRC']

all_cats = ['AI4Code']


# topic_perpl_dict = {"FreeLaw_Options": 8.978133340013256, "EuroParliamentProceedings": 15.162733092355857,
#                     "arXiv": 9.895235844017671, "DMMath": 8.766861598182608, "USPTO": 8.898743946087595, 'ASFPublicMail': 10.025503540132409, 
#                     'StackExchange': 9.367101029915574,  'USENET': 9.328392523746125, 'OtherWiki': 9.357722072244057, 'Discourse': 9.382440919569483,
#                     "PubMed": 9.29481637396142, "S2ORC": 9.053711915732272}
topic_perpl_dict = {}

for subset in all_cats:
    if subset in topic_perpl_dict:
        continue
    print("Run topic modeling for: " + subset)
    dataset = load_from_disk(f'hf_data_pilev2_by_cats/{subset}')
    idxs = random.sample(range(len(dataset)), min(1000000, len(dataset)))
    dataset = dataset.select(idxs)
    dataset = dataset.map(
        preprocess, batched=False, 
        num_proc=128, remove_columns=dataset.column_names
    )
    texts = dataset['processed_text']
    corpus = [id2word.doc2bow(text) for text in texts]
    temp = lda_model.log_perplexity(corpus)
    perplexity = np.exp(-1. * temp)
    topic_perpl_dict[subset] = np.log(perplexity)
    print("Completed: ", topic_perpl_dict)
    json.dump(topic_perpl_dict, open('new_topic_perpl_dict.json', 'w'))
    
print("Perplexity: ", topic_perpl_dict)