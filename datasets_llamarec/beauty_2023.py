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


class SmallBeauty2023Dataset(AbstractDataset):
    
    @classmethod
    def code(cls):
        return 'small_beauty_2023'
    
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