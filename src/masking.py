# src/masking.py
import logging
import random
from abc import ABC, abstractmethod
from data_models import CaptionedClip
from constants import DATA_MISSING

class MaskingStrategy(ABC):
    """Abstract base class for all masking strategies."""
    def __init__(self, scheme: str):
        self.scheme = scheme

    @abstractmethod
    def _get_indices_to_mask(self, num_clips: int) -> set:
        pass

    def apply(self, caption: list[CaptionedClip]) -> tuple[list[CaptionedClip], set]:
        indices_to_mask: set = self._get_indices_to_mask(len(caption))
        
        masked_captions = []
        for i, clip in enumerate(caption):
            if i in indices_to_mask:
                masked_clip = clip.model_copy()
                masked_clip.data = DATA_MISSING
                masked_captions.append(masked_clip)
            else:
                masked_captions.append(clip)
        
        logging.debug(f"Masked {len(indices_to_mask)} of {len(caption)} clips using '{self.scheme}'.")
        return masked_captions, indices_to_mask

    def __repr__(self) -> str:
        """Generates a descriptive string for the strategy and its parameters."""
        params = self._get_params_for_repr()
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.scheme}({param_str})"

    @abstractmethod
    def _get_params_for_repr(self) -> dict:
        """Returns a dictionary of parameters for the string representation."""
        pass

class RandomMasking(MaskingStrategy):
    """Masks a random selection of clips based on a ratio."""
    def __init__(self, ratio: float, prn_generator: random.Random):
        super().__init__("random")
        self.ratio = ratio
        self.prn = prn_generator

    def _get_indices_to_mask(self, num_clips: int) -> set:
        num_to_mask = int(num_clips * self.ratio)
        return set(self.prn.sample(range(num_clips), k=num_to_mask))

    def _get_params_for_repr(self) -> dict:
        return {"ratio": self.ratio}

class PartitionMasking(MaskingStrategy):
    """A generic strategy that divides a sequence into partitions and masks a block."""
    def __init__(self, num_partitions: int, start_partition: int, num_parts_to_mask: int):
        super().__init__("partition")
        self.num_partitions = num_partitions
        self.start_partition = start_partition
        self.num_parts_to_mask = num_parts_to_mask

    def _get_indices_to_mask(self, num_clips: int) -> set:
        if self.num_partitions > num_clips:
            return set() # Cannot partition if there are more partitions than items

        base_size = num_clips // self.num_partitions
        remainder = num_clips % self.num_partitions
        
        partitions = []
        current_index = 0
        for i in range(self.num_partitions):
            part_size = base_size + 1 if i < remainder else base_size
            partitions.append(list(range(current_index, current_index + part_size)))
            current_index += part_size

        indices_to_mask = set()
        end_partition = self.start_partition + self.num_parts_to_mask
        for i in range(self.start_partition, end_partition):
            if i < len(partitions):
                indices_to_mask.update(partitions[i])
        return indices_to_mask

    def _get_params_for_repr(self) -> dict:
        return {"num_partitions": self.num_partitions, "start_partition": self.start_partition, "num_parts_to_mask": self.num_parts_to_mask}

def get_masking_strategies(masking_configs: list, master_seed: int) -> list[MaskingStrategy]:
    """
    Factory function that reads a list of masking configurations and generates
    a list of all specified masking strategy instances.
    """
    strategies = []

    for config in masking_configs:
        scheme = config.get("scheme")
        if scheme == "random":
            seed = config.get("seed", 0) # TODO: if "seed" is a list, iterate over all values
            for ratio in config.get("ratio", []):
                strategies.append(RandomMasking(ratio=ratio, prn_generator=random.Random(master_seed+seed) ))

        elif scheme == "partition":
            num_partitions = config["num_partitions"]

            for num_to_mask in config.get("num_parts_to_mask", []):
                if num_to_mask > num_partitions:
                    continue

                max_start_part = num_partitions - num_to_mask
                for start_part in range(max_start_part + 1):
                    strategies.append(PartitionMasking(
                        num_partitions=num_partitions,
                        start_partition=start_part,
                        num_parts_to_mask=num_to_mask
                    ))
        else:
            raise NotImplementedError(f"Masking scheme '{scheme}' is not implemented.")

    return strategies

