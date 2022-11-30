from datasets import load_dataset, load_from_disk, concatenate_datasets
import random
import os

import logging
import os
from matplotlib import pyplot as plt
from rich.logging import RichHandler
import sys
import json
import numpy as np
sys.path.append("hf_tools")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(RichHandler(rich_tracebacks=True))

from filtering import LoadParameters,  ModifyingDocuments, Filtering
from parameters_filtering import parameters_filtering_default
from flagged_words import flagged_words

lang_dataset_id = 'en'
path_sentencepiece_model = '/fsx/home-duyphung/pilev2/hf_tools/en.sp.model'
path_kenlm_model = '/fsx/home-duyphung/pilev2/hf_tools/en.arpa.bin'
sentencepiece_model = LoadParameters.load_sentencepiece_model(
    lang_dataset_id, path_sentencepiece_model
)
kenlm_model = LoadParameters.load_kenlm_model(
    lang_dataset_id, path_kenlm_model
)

def stats_func(document):
    perplexity_score = Filtering.compute_perplexity_score(
        document['text'],
        sentencepiece_model,
        kenlm_model,
    )
    perplexity_score = round(min(perplexity_score, 20000), 3)
    flag_words_ratio = Filtering.compute_flagged_words_ratio(
        document['text'],
        sentencepiece_model,
        parameters_filtering_default['strip_characters'],
        parameters_filtering_default['cond_words_augmentation'],
        parameters_filtering_default['words_augmentation_group_sizes'],
        parameters_filtering_default['words_augmentation_join_char'],
        flagged_words
    )
    
    return {
        "flag_words_ratio": flag_words_ratio,
        "word_count": len(document['text'].split()),
        "perplexity_score": perplexity_score,
    }
    
text_datasets = [
    'Enwiki', 'arXiv', 'DMMath', 'PubMed', 'Books3', 'Gutenberg', 'FreeLaw_Options', 'UbuntuIRC','AMPS', 
    'EuroParliamentProceedings', 'USPTO', 'PileOfLaw', 'ASFPublicMail',  'USENET',  'OtherWiki', "S2ORC",
    'RedditNonProgramming_0', 'RedditNonProgramming_1', 'RedditNonProgramming_2', 'RedditNonProgramming_3',
    'RedditNonProgramming_4', 'RedditNonProgramming_5', 'RedditNonProgramming_6', 'RedditNonProgramming_7',
    'RedditNonProgramming_8', 'RedditNonProgramming_9', 'RedditNonProgramming_11',
    'RedditNonProgramming_12', 'RedditNonProgramming_13', 'RedditNonProgramming_14', 'RedditNonProgramming_15',
    'RedditNonProgramming_16', 'RedditNonProgramming_17', 'RedditNonProgramming_18', 
    'RedditNonProgramming_19', 'RedditNonProgramming_10', 'RedditNonProgramming_20'  ,  'RedditProgramming_0', 'RedditProgramming_1', 'StackExchange_ver2' 
]

code_datasets = [
    'LeetCode',  'AI4Code', 'Discourse', "GithubIssues", "GithubDiff_0", "GithubDiff_3", "GithubDiff_2", "GithubDiff_1",
    'RedditProgramming_0', 'RedditProgramming_1', 'RedditProgramming_2'
]


small_pilev2_dict = {}
stats_dict = {}
for subset in text_datasets:
    print("Sampling: ", subset)
    data = load_from_disk(f'/fsx/shared/hf_data_pilev2_by_cats/{subset}')
    idxs = random.sample(range(len(data)), min(1000000, len(data)))
    small_pilev2_dict[subset] = data.select(idxs)
    ds = small_pilev2_dict[subset].map(
        stats_func, 
        num_proc=128,
        remove_columns=small_pilev2_dict[subset].column_names
    )
    print("Length: ", len(ds))
    lst_word_count = ds['word_count']
    lst_perplexity_score = ds['perplexity_score']
    lst_flag_words_ratio = ds['flag_words_ratio']
    names = ['word_count', 'perplexity', 'flag_words_ratio']
    stats_dict[subset] = {}
    for (name, lst) in zip(names, [lst_word_count, lst_perplexity_score, lst_flag_words_ratio]):
        lst = np.array(lst)
        mean, median, std, min_, max_ = lst.mean(), np.median(lst), lst.std(), lst.min(), lst.max()
        quantiles = list(np.quantile(lst, [0.25, 0.5, 0.75]))
        stats_dict[subset][name] = {
            "mean": mean,
            "median": median,
            "std": std,
            "min": min_,
            "max": max_,
            "quantiles": quantiles,
            "lst": lst
        }
        # plot
        if name == "perplexity":
            plt.hist(lst, bins=1000)
        else:
            plt.hist(lst, bins=100)
        plt.title(f"{subset} - {name}")
        plt.savefig(f"/fsx/home-duyphung/pilev2/result_stats/{subset}_{name}.png")
        plt.clf()
        import pickle
        pickle.dump(stats_dict, open("/fsx/home-duyphung/pilev2/result_stats/stats_dict2.pkl", "wb"))