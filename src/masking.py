# src/masking.py
import logging
import random
from abc import ABC, abstractmethod
from data_models import CaptionedClip
from constants import DATA_MISSING

class BaseMaskingStrategy(ABC):
    """Abstract base class for all masking strategies."""
    def __init__(self, prn_generator: random.Random):
        self.prn = prn_generator # Store the dedicated generator instance

    @abstractmethod
    def _get_indices_to_mask(self, num_clips: int) -> set:
        pass

    def apply(self, caption: list[CaptionedClip]) -> list[CaptionedClip]:
        indices_to_mask = self._get_indices_to_mask(len(caption))
        
        masked_captions = []
        for i, clip in enumerate(caption):
            if i in indices_to_mask:
                masked_clip = clip.model_copy()
                masked_clip.data = DATA_MISSING
                masked_captions.append(masked_clip)
            else:
                masked_captions.append(clip)
        
        logging.info(f"Masked {len(indices_to_mask)} of {len(caption)} clips.")
        return masked_captions

class RandomMasking(BaseMaskingStrategy):
    """Masks a random selection of clips based on a ratio."""
    def __init__(self, ratio: float, prn_generator: random.Random):
        super().__init__(prn_generator)
        self.ratio = ratio

    def _get_indices_to_mask(self, num_clips: int) -> set:
        num_to_mask = int(num_clips * self.ratio)
        return set(self.prn.sample(range(num_clips), k=num_to_mask))

class ContiguousRandomMasking(BaseMaskingStrategy):
    """Masks a single, randomly placed contiguous block of clips."""
    def __init__(self, ratio: float, prn_generator: random.Random):
        super().__init__(prn_generator)
        self.ratio = ratio

    def _get_indices_to_mask(self, num_clips: int) -> set:
        num_to_mask = int(num_clips * self.ratio)
        if num_to_mask == 0: return set()
        start_index = self.prn.randint(0, num_clips - num_to_mask)
        return set(range(start_index, start_index + num_to_mask))

class ContiguousControlledMasking(BaseMaskingStrategy):
    """Masks a specific, contiguous block of clips."""
    def __init__(self, start_index: int, num_to_mask: int, prn_generator: random.Random):
        super().__init__(prn_generator)
        self.start_index = start_index
        self.num_to_mask = num_to_mask

    def _get_indices_to_mask(self, num_clips: int) -> set:
        return set(range(self.start_index, self.start_index + self.num_to_mask))

class SystematicMasking(BaseMaskingStrategy):
    """Masks clips at a regular interval."""
    def __init__(self, ratio: float, prn_generator: random.Random):
        super().__init__(prn_generator)
        self.ratio = ratio

    def _get_indices_to_mask(self, num_clips: int) -> set:
        num_to_mask = int(num_clips * self.ratio)
        if num_to_mask == 0: return set()
        step = num_clips // num_to_mask
        start_offset = self.prn.randint(0, step - 1)
        return set(range(start_offset, num_clips, step))

def get_masking_strategy(masking_config: dict, seed: int) -> BaseMaskingStrategy:
    """Factory function that reads the config and builds the correct masking strategy."""
    masking_prn = random.Random(seed) # Create the dedicated PRN generator
    scheme = masking_config.get("scheme")

    if scheme == "random":
        return RandomMasking(ratio=masking_config['ratio'], prn_generator=masking_prn)
    elif scheme == "contiguous_random":
        return ContiguousRandomMasking(ratio=masking_config['ratio'], prn_generator=masking_prn)
    elif scheme == "contiguous_controlled":
        return ContiguousControlledMasking(
            start_index=masking_config['mask_index'],
            num_to_mask=masking_config['num_to_mask'],
            prn_generator=masking_prn
        )
    elif scheme == "systematic":
        return SystematicMasking(ratio=masking_config['ratio'], prn_generator=masking_prn)
    else:
        raise NotImplementedError(f"Masking scheme '{scheme}' is not implemented.")
