from abc import abstractmethod, ABC
from pathlib import Path
from typing import Iterator, Any, Self
import numpy as np
import yaml
from numpy.typing import NDArray
from data.data_loaders import BaseDataLoader, get_data_loader
from llm.embedder import Embedder


class VectorDataLoader(ABC):

    @abstractmethod
    def get_data_type_name(self):
        pass

    @abstractmethod
    def load(self) -> Iterator[tuple[NDArray[np.float64], str]]:
        pass

    def count(self) -> int:
        return len(list(self.load()))

    @classmethod
    def from_config(cls, data_config: dict[str, Any], llm_client=None) -> Self:
        """
        Factory function that reads the config and returns the appropriate
        data loader instance.
        """
        dataset_name = data_config["name"]  # TODO rename to dataset_type?
        limit = data_config.get("limit")

        if dataset_name == "np_files":
            data_path = Path(data_config["path"])
            return VectorFileLoader(
                directory=data_path,
                data_type_name=VectorFileLoader.data_type_name(data_path),
                limit=limit
            )
        elif dataset_name == "toy_vectors":
            return ToyVectorsLoader(
                data_type_name="video_embeddings",
                number_of_matrices=limit or 5,
                row_num=data_config.get("row_num", 8),
                col_num=data_config.get("col_num", 32),
                seed=42
            )
        else:
            emb_model_name = data_config.get('embedding_model', 'gemini')
            if emb_model_name.startswith('local:'):
                from llm.local_embedder import LocalEmbedder
                model_id = emb_model_name.split('local:', 1)[1] or "all-MiniLM-L6-v2"
                embedder = LocalEmbedder(model_name=model_id)
            else:
                 embedder = Embedder(client=llm_client)

            return VectorConvertorLoader(
                get_data_loader(data_config),
                embedder
            )

class ToyVectorsLoader(VectorDataLoader):
    """
    A data loader that generates a deterministic, random toy dataset of vector matrices.
    Perfect for system tests and reproducibility.
    """
    def __init__(
            self,
            data_type_name: str,
            number_of_matrices: int,
            row_num: int,
            col_num: int,
            seed: int
    ):
        """
        Initializes the toy data loader with specified dimensions and seed.

        Args:
            number_of_matrices: The total number of matrices to generate.
            row_num: The number of rows (vectors) in each matrix.
            col_num: The number of columns (embedding dimensions) in each vector.
            seed: The seed for the random number generator to ensure reproducibility.
        """
        self.data_type_name = data_type_name
        self.number_of_matrices = number_of_matrices
        self.row_num = row_num
        self.col_num = col_num
        self._rng = np.random.default_rng(seed)

    def load(self) -> Iterator[tuple[NDArray[np.float64], str]]:
        """
        Yields a specified number of random matrices and their unique IDs.
        """
        for i in range(self.number_of_matrices):
            matrix_id = f'toy_matrix_{i}'

            # Generate a random matrix with values between 0 and 1
            matrix = self._rng.random(size=(self.row_num, self.col_num), dtype=np.float64)
            norm = np.linalg.norm(matrix, axis=1, keepdims=True)
            # Avoid division by zero for any zero-vectors
            norm[norm == 0] = 1
            normalized_matrix = matrix / norm
            yield normalized_matrix, matrix_id

    def get_data_type_name(self):
        return self.data_type_name

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
