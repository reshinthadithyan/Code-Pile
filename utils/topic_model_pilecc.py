import re
import gensim
from gensim.utils import simple_preprocess
import numpy as np
# NLTK Stop words
# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(words):
    return [word for word in words if word not in stop_words]

def make_bigrams(words):
    return bigram_mod[words]

def make_trigrams(words):
    return trigram_mod[bigram_mod[words]]

def lemmatization(words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    doc = nlp(" ".join(words))
    return [token.lemma_ for token in doc if token.pos_ in allowed_postags]

import spacy
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])


def normalize_text(example):
    text = re.sub('[,\.!?]', '', example['text']).lower()
    text = gensim.utils.simple_preprocess(str(text), deacc=True)
    return {"processed_text": text}

def preprocess(example):
    data_words_nostops = remove_stopwords(example['processed_text'])
    data_words_bigrams = make_bigrams(data_words_nostops)
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    return {"data_lemmatized": data_lemmatized}
    

from datasets import load_dataset, load_from_disk
pilecc = load_from_disk('hf_data_pilev2_by_cats/Enwiki')
# import random
# indexs = random.sample(range(0, len(pilecc)), 10000)
# pilecc = pilecc.select(indexs)
pilecc = pilecc.map(normalize_text, batched=False, num_proc=128, remove_columns=pilecc.column_names)
data_words = pilecc['processed_text']
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
# trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
# trigram_mod = gensim.models.phrases.Phraser(trigram)

pilecc = pilecc.map(preprocess, batched=False, num_proc=128, remove_columns=pilecc.column_names)
data_lemmatized = pilecc['data_lemmatized']
import gensim.corpora as corpora
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)
# Create Corpus
texts = data_lemmatized
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
if 0:

    lda_model = gensim.models.ldamodel.LdaModel.load('lda_model_Enwiki_20_1M')
    print("Perplexity: ", np.log(np.exp(-1. * lda_model.log_perplexity(corpus))))
    exit()
    
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=20, 
                                       random_state=100,
                                       chunksize=10000,
                                       iterations=20,
                                       passes=10,
                                       per_word_topics=True)
# Print the Keyword in the 10 topics
print(lda_model.print_topics())
perplexity = lda_model.log_perplexity(corpus)
print("Perplexity: ", np.log(np.exp(-1. * perplexity)))
lda_model.save('lda_save/lda_model_Enwiki')
