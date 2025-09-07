import logging
from abc import abstractmethod, ABC
from typing import Any
from typing import Iterator
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from data_loaders import BaseDataLoader
from embedder import Embedder
from evaluation import metrics_to_json
from vectors.eval_vectors import VectorReconstructionEvaluator

NPY_FILE_PATTERN = "*.npy"

from masking import MaskingStrategy
from vectors.reconstruction_startegies import VectorReconstructionStrategy

class VectorDataLoader(ABC):
    @abstractmethod
    def load(self) -> Iterator[tuple[NDArray[np.float64], str]]:
        pass

class VectorFileLoader(VectorDataLoader):
    def __init__(self, directory: Path, file_pattern: str = NPY_FILE_PATTERN):
        self.directory = directory
        self.file_pattern = file_pattern

    @staticmethod
    def find_numpy_files(directory: Path, file_pattern: str = NPY_FILE_PATTERN) -> list[Path]:
        return list(directory.rglob(file_pattern))

    @staticmethod
    def load_numpy_files(
        npy_files: list[Path],
        max_rows: int | None = None
    ) -> Iterator[tuple[NDArray[np.float64], str]]:
        max_rows = max_rows or 60
        for file_path in npy_files:
            yield np.load(file_path)[:max_rows], file_path.stem

    def load(self) -> Iterator[tuple[NDArray[np.float64], str]]:
        yield from self.load_numpy_files(self.find_numpy_files(self.directory, self.file_pattern))

class VectorConvertorLoader(VectorDataLoader):
    def __init__(self, base_dataloader:BaseDataLoader, embedder:Embedder):
        self.base_dataloader:BaseDataLoader=base_dataloader
        self.embedder:Embedder=embedder

    def load(self) -> Iterator[tuple[NDArray[np.float64], str]]:
        for x in self.base_dataloader.load():
            yield np.array(self.embedder.get_embeddings(x.video_id, x.get_texts()), dtype=np.float64), x.video_id


class VectorRunner:
    """
    Encapsulates and runs a single, atomic experiment.
    It is a pure "doer" that receives all its dependencies via injection.
    """
    def __init__(
        self,
        run_name: str,
        data_loader: VectorDataLoader,
        masking_strategy: MaskingStrategy,
        reconstruction_strategy: VectorReconstructionStrategy,
        evaluator: VectorReconstructionEvaluator,
        result_path: Path,
        conf_for_log:dict[str, Any]
    ):
        self.run_name = run_name
        self._data_loader = data_loader
        self._masking_strategy = masking_strategy
        self._reconstruction_strategy = reconstruction_strategy
        self._evaluator = evaluator
        self._result_path = result_path/run_name
        self.conf_for_log = conf_for_log

    def run(self) -> list[dict]:
        """Runs the full experiment from data loading to evaluation."""
        all_metrics:list[dict] = []

        for m, video_id in self._data_loader.load():
            logging.debug(f"--- Processing Video: {video_id} ---")

            masked_indices_set = self._masking_strategy.get_indices_to_mask(len(m))
            masked_indices = np.array(sorted(list(masked_indices_set)))
            masked_video = m.copy()
            masked_video[masked_indices] = np.nan
            reconstructed_vectors = self._reconstruction_strategy.reconstruct(masked_video)

            if len(reconstructed_vectors) != len(masked_indices):
                logging.warning(f'Bad indices found in reconstructed_video {video_id}, {masked_indices=}, skipping')
                continue

            video_metrics = self._evaluator.evaluate(reconstructed_vectors, m[masked_indices])

            video_metrics.update({
                "video_id": video_id,
                "num_captions": len(m),
                "masked": list(masked_indices)
            })

            all_metrics.append(video_metrics)

            logging.info(f"Evaluation complete metrics={metrics_to_json(video_metrics)}")

        return all_metrics
