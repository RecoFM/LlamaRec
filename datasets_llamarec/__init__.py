from .ml_100k import ML100KDataset
from .beauty import BeautyDataset
from .games import GamesDataset
from .beauty_2023 import SmallBeauty2023Dataset

DATASETS = {
    ML100KDataset.code(): ML100KDataset,
    BeautyDataset.code(): BeautyDataset,
    GamesDataset.code(): GamesDataset,
    SmallBeauty2023Dataset.code(): SmallBeauty2023Dataset
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
