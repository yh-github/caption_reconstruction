from abc import abstractmethod, ABC
from pathlib import Path
from typing import Iterator
import numpy as np
import yaml
from numpy.typing import NDArray
from data_loaders import BaseDataLoader, get_data_loader
from embedder import Embedder


class VectorDataLoader(ABC):

    @abstractmethod
    def get_data_type_name(self):
        pass

    @abstractmethod
    def load(self) -> Iterator[tuple[NDArray[np.float64], str]]:
        pass

    @staticmethod
    def from_config(data_config: dict):
        """
        Factory function that reads the config and returns the appropriate
        data loader instance.
        """
        dataset_name = data_config["name"]  # TODO rename to dataset_type?
        data_path = Path(data_config["path"])
        limit = data_config.get("limit")

        if dataset_name == "np_files":
            return VectorFileLoader(
                directory=data_path,
                data_type_name=VectorFileLoader.data_type_name(data_path),
                limit=limit
            )
        else:
            return VectorConvertorLoader(
                get_data_loader(data_config),
                Embedder() #TODO extern
            )


class VectorFileLoader(VectorDataLoader):
    def __init__(self, directory: Path, data_type_name:str, limit:int|None=None, file_pattern: str = "*.npy"):
        self.directory = directory
        self._data_type_name = data_type_name
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

    def get_data_type_name(self) -> str:
        return self._data_type_name

    @staticmethod
    def data_type_name(data_path):
        with open(Path(data_path / "metadata.yaml")) as f:
            return yaml.safe_load(f)["type"]


class VectorConvertorLoader(VectorDataLoader):
    def __init__(self, base_dataloader: BaseDataLoader, embedder: Embedder):
        self.base_dataloader: BaseDataLoader = base_dataloader
        self.embedder: Embedder = embedder

    def load(self) -> Iterator[tuple[NDArray[np.float64], str]]:
        for x in self.base_dataloader.load():
            yield np.array(self.embedder.get_embeddings(x.video_id, x.get_texts()), dtype=np.float64), x.video_id

    def get_data_type_name(self) -> str:
        return f"text_embeddings({self.base_dataloader.get_data_type_name()})"
