# src/experiment_runner.py
import logging
import mlflow
from data_loaders import get_data_loader
from masking import MaskingStrategy
from reconstruction_strategies import ReconstructionStrategy

class ExperimentRunner:
    """
    Encapsulates all components required to run a single experiment.
    This object is built in the main script and its run() method is called.
    """
    def __init__(self, config: dict, strategy: ReconstructionStrategy):
        self.config = config
        self.strategy = strategy
        self.data_loader = get_data_loader(config)
        self.masker = MaskingStrategy(config)

    def run(self):
        """Runs the full experiment from data loading to evaluation."""
        ground_truth_captions = self.data_loader.load()
        masked_captions = self.masker.apply(ground_truth_captions)
        
        reconstructed_captions = self.strategy.reconstruct(masked_captions)
        
        if reconstructed_captions:
            # We would call our final evaluation module here
            # metrics = evaluate(...)
            # mlflow.log_metrics(metrics)
            logging.info("Experiment finished successfully!")
        else:
            logging.error("Reconstruction failed.")
            mlflow.log_metric("reconstruction_failed", 1)
