from .base import AbstractDataset
from .utils import *

from datetime import date
from pathlib import Path
import pickle
import shutil
import tempfile
import os

import gzip
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class Dataset2023(AbstractDataset):
    @classmethod
    def code(cls):
        raise NotImplementedError("Dataset code not implemented")
    
    @classmethod
    def url(cls):
        assert False, 'This dataset is not available for download'
        
    @classmethod
    def zip_file_content_is_folder(cls):
        assert False, 'This dataset is not available for download'

    @classmethod
    def all_raw_file_names(cls):
        assert False, 'This dataset is not available for download'

    def maybe_download_raw_dataset(self):
        assert False, 'This dataset is not available for download'

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        print('dataset_path')
        print(dataset_path)
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        else:
            assert False, 'This dataset is not available for download'
            
    def load_ratings_df(self):
        assert False, 'This dataset is not available for download'
    
    def load_meta_dict(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath(self.all_raw_file_names()[1])

        meta_dict = {}
        with gzip.open(file_path, 'rb') as f:
            for line in f:
                item = eval(line)
                if 'title' in item and len(item['title']) > 0:
                    meta_dict[item['asin'].strip()] = item['title'].strip()
        
        return meta_dict


class AllBeauty2023Dataset(Dataset2023):
    @classmethod
    def code(cls):
        return 'all_beauty_2023'


class SportsAndOutdoors2023Dataset(Dataset2023):
    @classmethod
    def code(cls):
        return 'sports_and_outdoors_2023'


class ToysAndGames2023Dataset(Dataset2023):
    @classmethod
    def code(cls):
        return 'toys_and_games_2023'

class BeautyAndPersonalCare2023Dataset(Dataset2023):
    @classmethod
    def code(cls):
        return 'Beauty_and_Personal_Care_2023'