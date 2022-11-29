import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class Viz:
    def __init__(self) -> None:
        self.config = None 

    def add_config(self, config: dict[object]):
        self.config = config
    
    def process_stats_dataset(self,dataset_path:str):
        """
        process stats dataset and return a pandas dataframe
        """
        dataset = datasets.load_from_disk(dataset_path).to_pandas()
        self.dataset = dataset
        return dataset

    def find_uniques(self,config:dict[str],dataset:pd.DataFrame):
        """
        
        """
        dist_dict = {"dist":{}}
        for label in config["to_pivot_meta"]:
            dataset["meta"][label] = dataset[label].unique()
            print(dist_dict)
            if label not in dist_dict:
                dist_dict[label].append(label)


    def __call__(self,dataset_path:str):
        self.process_stats_dataset(dataset_path)
        if self.config is None:
            logger.info("No config found, no metadata pivot")
        else:
            dist_dict = self.find_uniques(self.config,self.dataset)
            print(dist_dict)


def custom_stats(stats_path:str):
	"""
	Given a path loads a stats dataset and checks basic configs
	"""
	dataset = datasets.load_from_disk(stats_path)
	return dataset

if __name__ == "__main__":
    TEST_STAT_PATH = "codepile/stats/stat_dump/StackExchange"
    viz = Viz()
    DUMMY_CONFIG = {"to_pivot_meta" : [
        "source"
    ]}
    dataset = viz.process_stats_dataset(TEST_STAT_PATH)
    print(dataset.head().columns)
    viz.add_config(DUMMY_CONFIG)
    viz(TEST_STAT_PATH)
