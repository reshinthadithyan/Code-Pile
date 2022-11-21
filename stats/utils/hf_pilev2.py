"""The Pile V2 dataset."""

import json

import datasets
from boto3.session import Session
import boto3
import os

_ROOT_DIR = "s3_pilev2"
_DESCRIPTION = "PileV2"


_DATA_URLS = {
    "all": [
        'data_lm_arxiv/data_0_time1667490666_arXiv_batch1.jsonl.zst',
        'data_lm_arxiv/data_0_time1667461348_arXiv_batch2.jsonl.zst',
        'data_lm_arxiv/data_0_time1667479574_arXiv_batch4.jsonl.zst',
        'data_lm_arxiv/data_0_time1667481621_arXiv_batch5.jsonl.zst',
        'data_lm_arxiv/data_0_time1667466895_arXiv_batch3.jsonl.zst',
        'data_0_time1666921283_PubMedDataset.jsonl.zst',
        'data_0_time1666905734_Books3.jsonl.zst',
        'gutenberg.jsonl.zst',
        'FreeLaw_Opinions.jsonl.zst',
        'ubuntu_irc_until_2020_9_1.jsonl.zst',
        'enwiki_20221006_lmdataformat.jsonl.zst',
        'EuroParliamentProceedings_1996_2011.jsonl.zst',
        'data_0_time1666879910_DMMathDataset.jsonl.zst',
        'data_lm_USPTO_Application/data_3_time1667983523_USPTO_Application_9.jsonl.zst',
        'data_lm_USPTO_Application/data_3_time1667982316_USPTO_Application_3.jsonl.zst',
        'data_lm_USPTO_Application/data_5_time1667982737_USPTO_Application_5.jsonl.zst',
        'data_lm_USPTO_Application/data_5_time1667983787_USPTO_Application_11.jsonl.zst',
        'data_lm_USPTO_Application/data_0_time1667981812_USPTO_Application_0.jsonl.zst',
        'data_lm_USPTO_Application/data_0_time1667983098_USPTO_Application_6.jsonl.zst',
        'data_lm_USPTO_Application/data_1_time1667983284_USPTO_Application_7.jsonl.zst',
        'data_lm_USPTO_Application/data_4_time1667982512_USPTO_Application_4.jsonl.zst',
        'data_lm_USPTO_Application/data_4_time1667983723_USPTO_Application_10.jsonl.zst',
        'data_lm_USPTO_Application/data_1_time1667982006_USPTO_Application_1.jsonl.zst',
        'data_lm_USPTO_Application/data_2_time1667982188_USPTO_Application_2.jsonl.zst',
        'data_lm_USPTO_Application/data_2_time1667983398_USPTO_Application_8.jsonl.zst',
        'data_lm_USPTO_Grant/data_5_time1667976082_USPTO_Grant_5.jsonl.zst',
        'data_lm_USPTO_Grant/data_10_time1667976806_USPTO_Grant_10.jsonl.zst',
        'data_lm_USPTO_Grant/data_4_time1667975959_USPTO_Grant_4.jsonl.zst',
        'data_lm_USPTO_Grant/data_1_time1667975615_USPTO_Grant_1.jsonl.zst',
        'data_lm_USPTO_Grant/data_7_time1667976401_USPTO_Grant_7.jsonl.zst',
        'data_lm_USPTO_Grant/data_3_time1667975902_USPTO_Grant_3.jsonl.zst',
        'data_lm_USPTO_Grant/data_9_time1667976610_USPTO_Grant_9.jsonl.zst',
        'data_lm_USPTO_Grant/data_0_time1667975451_USPTO_Grant_0.jsonl.zst',
        'data_lm_USPTO_Grant/data_6_time1667976208_USPTO_Grant_6.jsonl.zst',
        'data_lm_USPTO_Grant/data_2_time1667975770_USPTO_Grant_2.jsonl.zst',
        'data_lm_USPTO_Grant/data_8_time1667976518_USPTO_Grant_8.jsonl.zst',
        'data_lm_USPTO_Grant/data_11_time1667976888_USPTO_Grant_11.jsonl.zst',
        'pile_of_law_lmdataformat_filtered_2022_11_04.jsonl.zst',
        'data_0_time1666883223_ASFPublicMail.jsonl.zst',
        'data_0_time1666432830_StackExchangeDataset.jsonl.zst',
        'codepile/data_0_time1667836475_CPDataset_without_TopCoder.jsonl.zst',
        'data_0_time1668326801_USENET.jsonl.zst',
        'codepile/data_0_time1668223934_AI4Code.jsonl.zst',
        'codepile/data_0_time1668405609_Discourse.jsonl.zst',
        'data_0_time1667715493_AMPS.jsonl.zst',
        'codepile/data_0_time1666600580_LeetCodeDataset.jsonl.zst'
    ],
    "arXiv": [
        'data_lm_arxiv/data_0_time1667490666_arXiv_batch1.jsonl.zst',
        'data_lm_arxiv/data_0_time1667461348_arXiv_batch2.jsonl.zst',
        'data_lm_arxiv/data_0_time1667479574_arXiv_batch4.jsonl.zst',
        'data_lm_arxiv/data_0_time1667481621_arXiv_batch5.jsonl.zst',
        'data_lm_arxiv/data_0_time1667466895_arXiv_batch3.jsonl.zst',
    ],
    "PubMed": ['data_0_time1666921283_PubMedDataset.jsonl.zst'],
    "Books3": ['data_0_time1666905734_Books3.jsonl.zst'],
    "Gutenberg":  ['gutenberg.jsonl.zst'],
    "FreeLaw_Options": ['FreeLaw_Opinions.jsonl.zst'],
    "UbuntuIRC": ['ubuntu_irc_until_2020_9_1.jsonl.zst'],
    "Enwiki": [ 'enwiki_20221006_lmdataformat.jsonl.zst'],
    "EuroParliamentProceedings": ['EuroParliamentProceedings_1996_2011.jsonl.zst'],
    "DMMath":  ['data_0_time1666879910_DMMathDataset.jsonl.zst'],
    "USPTO":[
        'data_lm_USPTO_Application/data_3_time1667983523_USPTO_Application_9.jsonl.zst',
        'data_lm_USPTO_Application/data_3_time1667982316_USPTO_Application_3.jsonl.zst',
        'data_lm_USPTO_Application/data_5_time1667982737_USPTO_Application_5.jsonl.zst',
        'data_lm_USPTO_Application/data_5_time1667983787_USPTO_Application_11.jsonl.zst',
        'data_lm_USPTO_Application/data_0_time1667981812_USPTO_Application_0.jsonl.zst',
        'data_lm_USPTO_Application/data_0_time1667983098_USPTO_Application_6.jsonl.zst',
        'data_lm_USPTO_Application/data_1_time1667983284_USPTO_Application_7.jsonl.zst',
        'data_lm_USPTO_Application/data_4_time1667982512_USPTO_Application_4.jsonl.zst',
        'data_lm_USPTO_Application/data_4_time1667983723_USPTO_Application_10.jsonl.zst',
        'data_lm_USPTO_Application/data_1_time1667982006_USPTO_Application_1.jsonl.zst',
        'data_lm_USPTO_Application/data_2_time1667982188_USPTO_Application_2.jsonl.zst',
        'data_lm_USPTO_Application/data_2_time1667983398_USPTO_Application_8.jsonl.zst',
        'data_lm_USPTO_Grant/data_5_time1667976082_USPTO_Grant_5.jsonl.zst',
        'data_lm_USPTO_Grant/data_10_time1667976806_USPTO_Grant_10.jsonl.zst',
        'data_lm_USPTO_Grant/data_4_time1667975959_USPTO_Grant_4.jsonl.zst',
        'data_lm_USPTO_Grant/data_1_time1667975615_USPTO_Grant_1.jsonl.zst',
        'data_lm_USPTO_Grant/data_7_time1667976401_USPTO_Grant_7.jsonl.zst',
        'data_lm_USPTO_Grant/data_3_time1667975902_USPTO_Grant_3.jsonl.zst',
        'data_lm_USPTO_Grant/data_9_time1667976610_USPTO_Grant_9.jsonl.zst',
        'data_lm_USPTO_Grant/data_0_time1667975451_USPTO_Grant_0.jsonl.zst',
        'data_lm_USPTO_Grant/data_6_time1667976208_USPTO_Grant_6.jsonl.zst',
        'data_lm_USPTO_Grant/data_2_time1667975770_USPTO_Grant_2.jsonl.zst',
        'data_lm_USPTO_Grant/data_8_time1667976518_USPTO_Grant_8.jsonl.zst',
        'data_lm_USPTO_Grant/data_11_time1667976888_USPTO_Grant_11.jsonl.zst'
    ],
    "PileOfLaw": ['pile_of_law_lmdataformat_filtered_2022_11_04.jsonl.zst'],
    "ASFPublicMail": ['data_0_time1666883223_ASFPublicMail.jsonl.zst'],
    "StackExchange": ['codepile/data_0_time1667803157_StackExchangeDataset.jsonl.zst'],
    "CPDataset":   ['codepile/data_0_time1667836475_CPDataset_without_TopCoder.jsonl.zst'],
    "USENET":  ['data_0_time1668326801_USENET.jsonl.zst'],
    "AI4Code": ['codepile/data_0_time1668223934_AI4Code.jsonl.zst'],
    "Discourse":  ['codepile/data_0_time1668405609_Discourse.jsonl.zst'],
    "AMPS":   ['data_0_time1667715493_AMPS.jsonl.zst'],
    "LeetCode":  ['codepile/data_0_time1667808070_LeetCodeDataset.jsonl.zst'],
    "OtherWiki": ['data_0_time1667876060_others_wiki_dataset.jsonl.zst'],
    'S2ORC': [
        'data_lm_s2orc/data_3_time1667975610_S2ORC_3.jsonl.zst',
        'data_lm_s2orc/data_6_time1667976311_S2ORC_6.jsonl.zst',
        'data_lm_s2orc/data_5_time1667976071_S2ORC_5.jsonl.zst',
        'data_lm_s2orc/data_1_time1667975093_S2ORC_1.jsonl.zst',
        'data_lm_s2orc/data_2_time1667975339_S2ORC_2.jsonl.zst',
        'data_lm_s2orc/data_0_time1667974864_S2ORC_0.jsonl.zst',
        'data_lm_s2orc/data_8_time1667976795_S2ORC_8.jsonl.zst',
        'data_lm_s2orc/data_4_time1667975829_S2ORC_4.jsonl.zst',
        'data_lm_s2orc/data_0_time1667977212_S2ORC_9.jsonl.zst',
        'data_lm_s2orc/data_7_time1667976567_S2ORC_7.jsonl.zst'
    ],
    'PileCC': [
        'pile_cc_filtered_deduped.jsonl.zst'
    ],
    "TheStack": [
        f"codepile/the_stack_perm_license/{file}" for file in os.listdir(os.path.join(_ROOT_DIR, "codepile/the_stack_perm_license")) 
        if file.endswith(".jsonl.zst")
    ],
    "GithubDiff": [
        f"codepile/diffs/lmd_github_diffs/{file}" for file in os.listdir(os.path.join(_ROOT_DIR, "codepile/diffs/lmd_github_diffs/")) if file.endswith(".jsonl.zst")
    ],
    "GithubIssues": [
        "data_0_time1668691347_GithubIssues.jsonl.zst"
    ],
    "USPTO_V1": [
        f"pile_uspto/{file}" for file in os.listdir(os.path.join(_ROOT_DIR, "pile_uspto")) if file.endswith(".jsonl.zst")
    ]
}

_FEATURES = {
    "all": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "arXiv": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "PubMed": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "Gutenberg": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "FreeLaw_Options": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "Books3": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "UbuntuIRC": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "Enwiki": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "EuroParliamentProceedings": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "DMMath": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "USPTO": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "PileOfLaw": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "ASFPublicMail": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "StackExchange": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "CPDataset": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "USENET": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "AI4Code": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "Discourse": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "AMPS": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "LeetCode": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "S2ORC": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string")
        }
    ),
    "OtherWiki": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string")
        }
    ),
    "PileCC": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string")
        }
    ),
    "TheStack": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string")
        }
    ),
    "GithubDiff": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string")
        }
    ),
    "GithubIssues": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string")
        }
    ),
    "USPTO_V1": datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string")
        }
    )
}

batch_size = 400
diff_files = [
        f"codepile/diffs/lmd_github_diffs/{file}" for file in 
            os.listdir(os.path.join(_ROOT_DIR, "codepile/diffs/lmd_github_diffs/")) if file.endswith(".jsonl.zst")
]

for (i, count) in enumerate(range(0, len(diff_files), batch_size)):
    _DATA_URLS[f"GithubDiff_{i}"] = diff_files[count:count+batch_size]
    _FEATURES[f"GithubDiff_{i}"] = datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "meta": datasets.Value("string")
        }
    )


class ThePileConfig(datasets.BuilderConfig):
    """BuilderConfig for The Pile."""

    def __init__(self, *args, subsets, **kwargs):
        """BuilderConfig for The Pile.
        Args:
            subsets (:obj:`List[str]`): List of subsets to load.
            **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(
            *args,
            name="+".join(subsets),
            **kwargs,
        )
        self.subsets = subsets


class ThePile(datasets.GeneratorBasedBuilder):
    """The Pile dataset."""

    VERSION = datasets.Version("2.0.0")

    BUILDER_CONFIG_CLASS = ThePileConfig
    BUILDER_CONFIGS = [ThePileConfig(subsets=[subset]) for subset in _DATA_URLS]
    DEFAULT_CONFIG_NAME = "all"

    def _info(self):
        """Give information and typings for the dataset."""
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=_FEATURES.get(self.config.name),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=None,
            # License for the dataset if available
            license=None,#_LICENSES.get(self.config.name, "Multiple: see each subset license"),
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name='train',
                gen_kwargs={
                    "files": [os.path.join(_ROOT_DIR, x) for x in _DATA_URLS[self.config.name]],
                },
            )
        ]


    def _generate_examples(self, files):
        """Yield examples as (key, example) tuples."""
        key = 0
        if isinstance(files, list):
            import zstandard as zstd
            from tqdm import tqdm
            for path in tqdm(files):
                print("Loading", path)
                with zstd.open(open(path, "rb"), "rt", encoding="utf-8") as f:
                    for row in f:
                        try:
                            try:
                                data = json.loads(row)
                            except:
                                data = json.loads(row[:-1])
                            if len(data['text'].split()) > 8192: # truncate long documents
                                data['text'] = ' '.join(data['text'].split()[:8192])
                            if len(data['meta']) == 0: # prevent empty meta when write to lm_dataformat
                                data['meta'] = str({"source": path})
                            data['meta'] = str(data['meta'])
                            data['id'] = str(key)
                        except:
                            continue
                        yield key, data
                        key += 1