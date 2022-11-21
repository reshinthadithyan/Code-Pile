from datasets import load_dataset, load_from_disk
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

all_cats = ['Enwiki', 'arXiv', 'LeetCode', 'DMMath', 'PubMed', 'Books3', 'Gutenberg', 'FreeLaw_Options',
            'UbuntuIRC', 'EuroParliamentProceedings', 'DMMath',
            'USPTO', 'PileOfLaw', 'ASFPublicMail', 'StackExchange', 'CPDataset',
            'USENET', 'AI4Code', 'Discourse', 'AMPS', 'OtherWiki', "S2ORC", "TheStack", "GithubIssues",
            "GithubDiff_0", "GithubDiff_3", "GithubDiff_2", "GithubDiff_1"]

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

def count_tokens_batch(example):
    return {"length": tokenizer(example['text'], return_length=True)['length']}

def count_tokens_single(example):
    return {"length": tokenizer(example['text'], return_length=True)['length'][0]}

import json
counter = {}
for subset in all_cats:
    if subset in counter:
        continue
    print("Processing", subset)
    try:
        data = load_from_disk(f'hf_data_pilev2_by_cats/{subset}')
        tmp = data.map(count_tokens_single, batched=False, num_proc=128, remove_columns=data.column_names)
        counter[subset] = sum(tmp['length'])
        json.dump(counter, open('token_counting.json', 'w'))
    except Exception as e:
        print(e)
        print(f'Failed to load {subset}')
        continue
print("Total tokens: ", sum(counter.values()))