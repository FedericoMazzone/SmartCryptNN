from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

import numpy as np

from data_utils.load_location import load_location
from data_utils.load_mnist import load_MNIST
from data_utils.load_purchase100 import load_purchase100
from data_utils.load_texas100 import load_texas100


@dataclass
class Dataset:
    name: str
    loading_function: Callable[[str, int, int, bool, bool],
                               Tuple[np.ndarray, np.ndarray, np.ndarray,
                                     np.ndarray]]
    feature_shape: Union[int, Tuple[int, ...]]
    num_classes: int
    normalize: bool

    def load(
        self,
        data_path: Path=None,
        train_size: int=None,
        test_size: int=None,
        to_categorical: bool=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        if data_path is None:
            return self.loading_function()

        return self.loading_function(
            file_path=Path(data_path).joinpath(self.name),
            train_size=train_size,
            test_size=test_size,
            to_categorical=to_categorical,
            normalize=self.normalize
        )


DATASETS = {
    "location": Dataset(
        name="location",
        loading_function=load_location,
        feature_shape=446,
        num_classes=30,
        normalize=False
    ),
    "mnist": Dataset(
        name="mnist",
        loading_function=load_MNIST,
        feature_shape=64,
        num_classes=10,
        normalize=False
    ),
    "purchase100": Dataset(
        name="purchase100",
        loading_function=load_purchase100,
        feature_shape=600,
        num_classes=100,
        normalize=False
    ),
    "texas100": Dataset(
        name="texas100",
        loading_function=load_texas100,
        feature_shape=6169,
        num_classes=100,
        normalize=False
    )
}
