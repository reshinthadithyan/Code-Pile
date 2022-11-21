import os
from datasets import load_dataset
from sklearn.model_selection import train_test_split


all_cats = ['LeetCode', 'DMMath'
            ,'arXiv', 'PubMed', 'Books3', 'Gutenberg', 'FreeLaw_Options',
            'UbuntuIRC', 'Enwiki', 'EuroParliamentProceedings', 'DMMath',
            'USPTO', 'PileOfLaw', 'ASFPublicMail', 'StackExchange', 'CPDataset',
            'USENET', 'AI4Code', 'Discourse', 'AMPS', 'TheStack', 
            'GithubDiff_0', 'GithubDiff_1', 'GithubDiff_2', 'GithubDiff_3']

if not os.path.exists('hf_data_pilev2_by_cats'):
    os.mkdir('hf_data_pilev2_by_cats')

for subset in all_cats:
    print("Processing: ", subset)
    try:
        if not os.path.exists('cache_temp'):
            os.mkdir('cache_temp')
        hf_subset = load_dataset('hf_pilev2.py', cache_dir='hf_data_cats', subsets=[subset])
        hf_subset['train'].save_to_disk(f'hf_data_pilev2_by_cats/{subset}')
        os.rmdir('cache_temp')
    except Exception as e:
        print(e)
        print(f'Failed to load {subset}')