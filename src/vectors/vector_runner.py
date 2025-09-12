import logging
from typing import Any
import numpy as np
from pathlib import Path
from evaluations.evaluation import metrics_to_json, MetricsRecordRaw, MetricsMetadata, round_metrics, VectorReconstructionEvaluator
from data.vector_dataloaders import VectorDataLoader
from reconstruction.masking import MaskingStrategy
from reconstruction.vector_reconstruction import VectorReconstructionStrategy

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
        save_path: Path,
        conf_for_log:dict[str, Any]
    ):
        self.run_name = run_name
        self.data_loader = data_loader
        self._masking_strategy = masking_strategy
        self._reconstruction_strategy = reconstruction_strategy
        self.evaluator = evaluator
        self._result_path = save_path/run_name
        self.conf_for_log = conf_for_log

    def run(self) -> list[MetricsRecordRaw]:
        """Runs the full experiment from data loading to evaluation."""
        all_metrics:list[MetricsRecordRaw] = []

        for m, video_id in self.data_loader.load():
            logging.debug(f"--- Processing Video: {video_id} ---")

            masked_indices_set = self._masking_strategy.get_indices_to_mask(len(m))
            masked_indices_list = sorted(list(masked_indices_set))
            masked_indices = np.array(masked_indices_list)
            masked_video = m.copy()
            masked_video[masked_indices] = np.nan
            reconstructed_vectors = self._reconstruction_strategy.reconstruct(masked_video)

            if len(reconstructed_vectors) != len(masked_indices):
                logging.warning(f'Bad indices found in reconstructed_video {video_id}, {masked_indices=}, skipping')
                continue

            video_metrics = self.evaluator.evaluate(reconstructed_vectors, m[masked_indices])

            raw_record = MetricsRecordRaw(
                raw_metrics=video_metrics,
                metadata=MetricsMetadata(
                    video_id=video_id,
                    size=len(m),
                    masked=masked_indices_list,
                    recon_strategy=str(self._reconstruction_strategy),
                    data_type=self.data_loader.get_data_type_name()
                )
            )

            all_metrics.append(raw_record)

            logging.info(f"Evaluation complete metrics={metrics_to_json(round_metrics(video_metrics))}")

        return all_metrics
