from .ml_100k import ML100KDataset
from .beauty import BeautyDataset
from .games import GamesDataset
from .datasets_2023 import AllBeauty2023Dataset, SportsAndOutdoors2023Dataset, ToysAndGames2023Dataset

DATASETS = {
    ML100KDataset.code(): ML100KDataset,
    BeautyDataset.code(): BeautyDataset,
    GamesDataset.code(): GamesDataset,
    AllBeauty2023Dataset.code(): AllBeauty2023Dataset,
    SportsAndOutdoors2023Dataset.code(): SportsAndOutdoors2023Dataset,
    ToysAndGames2023Dataset.code(): ToysAndGames2023Dataset
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
