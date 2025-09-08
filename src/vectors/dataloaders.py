from abc import abstractmethod, ABC
from pathlib import Path
from typing import Iterator
import numpy as np
from numpy.typing import NDArray
from data_loaders import BaseDataLoader, get_data_loader
from embedder import Embedder

class VectorDataLoader(ABC):
    @abstractmethod
    def load(self) -> Iterator[tuple[NDArray[np.float64], str]]:
        pass

    @staticmethod
    def from_config(data_config: dict):
        """
        Factory function that reads the config and returns the appropriate
        data loader instance.
        """
        dataset_name = data_config.get("name")  # TODO rename to dataset_type?
        data_path = data_config.get("path")
        limit = data_config.get("limit")

        if not dataset_name or not data_path:
            raise ValueError("Dataset 'name' and 'path' must be specified in the config.")

        if dataset_name == "np_files":
            return VectorFileLoader(data_path, limit)
        else:
            return VectorConvertorLoader(get_data_loader(data_config), Embedder())


class VectorFileLoader(VectorDataLoader):
    def __init__(self, directory: Path, limit:int|None=None, file_pattern: str = "*.npy"):
        self.directory = directory
        self.limit = limit
        self.file_pattern = file_pattern

    def find_numpy_files(self) -> list[Path]:
        files = sorted(list(self.directory.rglob(self.file_pattern)))
        return files[:self.limit] if self.limit else files

    @staticmethod
    def load_numpy_files(
        npy_files: list[Path],
        max_rows: int | None = None
    ) -> Iterator[tuple[NDArray[np.float64], str]]:
        max_rows = max_rows or 60
        for file_path in npy_files:
            yield np.load(file_path)[:max_rows], file_path.stem

    def load(self) -> Iterator[tuple[NDArray[np.float64], str]]:
        yield from self.load_numpy_files(self.find_numpy_files())


class VectorConvertorLoader(VectorDataLoader):
    def __init__(self, base_dataloader: BaseDataLoader, embedder: Embedder):
        self.base_dataloader: BaseDataLoader = base_dataloader
        self.embedder: Embedder = embedder

    def load(self) -> Iterator[tuple[NDArray[np.float64], str]]:
        for x in self.base_dataloader.load():
            yield np.array(self.embedder.get_embeddings(x.video_id, x.get_texts()), dtype=np.float64), x.video_id
