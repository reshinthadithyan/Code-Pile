from typing import Iterable
from codepile.dataset import DatasetInfo, DatasetSources, RawDataset, Scraper, Processor, Analyser, Dataset
import os
import json
import urllib.request
import pandas as pd
from unidiff import PatchSet
from tqdm import tqdm
import logging
import ast

logger = logging.getLogger(__name__)


#START:Hypernion's GitHub Gist

def process_ind_patch(patch_diff) -> dict:
    """Process patch to get diff data."""
    patch_parsed_diff: dict = {
        "src_file": [],
        "tgt_file": [],
        "hunks": [],
        "addition_count": [],
        "deletion_count": [],
    }

    patch_parsed_diff["addition_count"] = patch_diff.added
    patch_parsed_diff["src_file"] = patch_diff.source_file
    patch_parsed_diff["tgt_file"] = patch_diff.target_file
    patch_parsed_diff["patch_info"] = patch_diff.patch_info
    patch_parsed_diff["deletion_count"] = patch_diff.removed
    patch_diff_list = list(patch_diff)
    for patch_diff_ind in patch_diff_list:
        patch_diff_ind = str(patch_diff_ind)
        patch_diff_split = patch_diff_ind.split("@@")
        patch_diff_line = patch_diff_split[2].split("\n")
        patch_diff_line_numbers = [list(map(int, hunk.strip("-+").split(",")))
                                   for hunk in patch_diff_split[1].strip().split(" ")]
        patch_parsed_diff["hunks"].append(patch_diff_line_numbers + patch_diff_line[:-1])
    return patch_parsed_diff


def patch_parse(patch_url: str) -> list:
    """Parse a patch to get diff data."""
    diff_list: list = []
    if ".diff" not in patch_url:
        patch_url = patch_url + ".diff"
    diff = urllib.request.urlopen(patch_url)
    encoding = diff.headers.get_charsets()[0]
    patch = PatchSet(diff, encoding=encoding)
    for patch_ind in patch:
        # Skip if the file is not a python file.
        if not patch_ind.target_file.endswith(".py"):
            continue
        patch_parsed = process_ind_patch(patch_ind)
        diff_list.append(patch_parsed)
    return diff_list


def apply_reverse_patch(diff_list: list, repo_data: tuple, length_threshold: int = 4096) -> list:
    """Apply reverse patch to get before files. Returns list of modified files."""
    files_list: list = []
    for diff in diff_list:
        # Get raw after file.
        file_raw_url = (f"https://raw.githubusercontent.com/{repo_data[0]}/"
                        f"{repo_data[1]}/{repo_data[2]}/{diff['tgt_file'][2:]}")
        raw_file = urllib.request.urlopen(file_raw_url)
        raw_file_encoding = raw_file.headers.get_charsets()[0]
        raw_file = [line.decode(raw_file_encoding) for line in raw_file.readlines()]
        # file_length = sum(1 for _ in raw_file)
        # if file_length < length_threshold:
        files_list.append(raw_file)
        # Iterate over hunks for this file and apply the reverse patch.
        for hunk in diff_list[0]["hunks"]:
            hunk_list = []
            for line in hunk[3:]:
                if line.startswith("-") or line.startswith(" "):
                    hunk_list.append(line[1:] + "\n")
            files_list[0][hunk[0][0]:hunk[0][0] + hunk[0][1]] = hunk_list

    return files_list


def process_commit(commit_url: str, commit_message: str) -> dict:
    """Process a commit url to get the before files and diff dict."""
    # Get dict containing diff data.
    diff_list = patch_parse(commit_url)
    patch_url_split = commit_url.split("/")
    # author, repo name, commit hash
    repo_data = (patch_url_split[3], patch_url_split[4], patch_url_split[6])
    # Get list of files, each of which is a list of strings, one for each line.
    files_list = apply_reverse_patch(diff_list, repo_data)
    data_dict = {
        "before_file": files_list,
        "commit_message": commit_message,
        "diff": diff_list,
    }
    return data_dict
#END:Hypernion's GitHub Gist

# class GitHubDatasetInfo(DatasetInfo):
#         identifier = "github_diff"
#         description = "Diffs from GitHub commits"
#         # the last time when new information was incorporated into the dataset
#         # aka when was the latest sample collected
#         data_end = None
#         # the beginning of the datasets data
#         data_start = None
#         # estimated size in bits
#         size = None

#         # compute cost needed for processing
#         # usefull information for rebuilding
#         cpu_hours = None 
#         gpu_hours = None 
#         ram_requirement = None 
#         tempfile_requirement = None 

#         # the main sources website/description/entrypoint/domain
#         source_uri = None 

#         # what are the advantages of including this dataset
#         # like a good fit for the downstream modelling tasks
#         dataset_pros = "Would be a very useful resource for GitHub diffs."
#         # what are the disadvantages of including this dataset
#         # like biases
#         dataset_cons  = "A lot of commits might be not passing the test cases"

#         # the languages that are present from the source download
#         languages = ["en"]
#         # the programming languages that are present from the source download
#         coding_languages = []
#         # the language modalities that are present in the dataset:
#         # like discussion, code_review, source_code, unittest
#         modalities = []
#         # to track copyright 
#         source_license = []
#         # a citation for acknowledging the data source
#         # as this is convention in academia
#         source_citation = []
#         # a single person responsible for the dataset
#         data_owner = []
#         contributers = []



class GitHubDiff(Dataset):
    def __init__(self, tempdir, target_dir, *args, **kwargs):
        self.tempdir = tempdir
        self.target_dir = target_dir

        self.info : DatasetInfo

    def make_commit_url(self,row):
        """
        Function to make a commit url given the repo name and commit_id
        """
        repo_name = row["repo_name"]
        #repo_name = ast.literal_eval(row["repo_name"])
        output_commit_urls = []
        # if isinstance(repo_name,list):
        #     for ind_repo in repo_name:
        #         repo_name = ind_repo
        #         commit_url =  f"https://github.com/{repo_name}/commit/{row['commit']}"
        #         output_commit_urls.append(commit_url)
        if isinstance(repo_name,str):
                commit_url =  f"https://github.com/{repo_name}/commit/{row['commit']}"
                output_commit_urls.append(commit_url)
        return output_commit_urls
            

    def fetch_raw(self,return_df=True):
        if return_df:
            raw_path = os.path.join("data","commit_dataset_1.csv")
            return pd.read_csv(raw_path)
        else:
            raise NotImplementedError("Can only return DataFrame")

    def download(self, return_df=True):
        df = self.fetch_raw(return_df)
        return df
    def process_ind_datapoint(self,commit_url:str, commit_message:str):
        """
        Process function for individual datapoint which would be useful for 
        """
        return process_commit(commit_url,commit_message)
    def process(self,dataset:Iterable):
        dataset_processed = []
        logger.info(f"Length of the dataset : {len(dataset)}")
        success_count = 0
        for ind,row in tqdm(dataset.iterrows(),total=len(dataset),desc=f"{success_count}"):
            success_count += 1
            commit_url_list = self.make_commit_url(row)
            for commit_url in commit_url_list:
                commit_message = row["message"]
                processed_diff = process_commit(commit_url,commit_message)
                dataset_processed.append(processed_diff)
            with open(os.path.join(self.target_dir,"trial_parsed.json"),"w") as f:
                json.dump(dataset_processed,f,indent=2)




if __name__ == "__main__":
    diff_dataset = GitHubDiff("data/","data/")
    diff_dataset_df = diff_dataset.download(True)
    diff_dataset.process(diff_dataset_df)
    print(diff_dataset_df.iloc[[0]].columns)
    print(diff_dataset_df.iloc[[0]]["commit"])

