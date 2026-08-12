import logging
import random
from abc import ABC, abstractmethod

from data_models.captions_only import CaptionedClip
from data_models.captions_only import CaptionedVideo


class MaskingStrategy(ABC):
    """Abstract base class for all masking strategies."""
    def __init__(self, scheme: str):
        self.scheme = scheme

    @abstractmethod
    def get_indices_to_mask(self, num_clips: int) -> set[int]:
        pass

    @staticmethod
    def mask_list(clips:list[CaptionedClip], indices_to_mask:set):
        masked_captions = []
        for clip in clips:
            if clip.index in indices_to_mask:
                masked_captions.append(clip.masked_copy())
            else:
                masked_captions.append(clip)
        return masked_captions

    def apply(self, captions: list[CaptionedClip]) -> tuple[list[CaptionedClip], set]:
        indices_to_mask: set = self.get_indices_to_mask(len(captions))
        masked_captions = self.mask_list(captions, indices_to_mask)
        return masked_captions, indices_to_mask

    def __repr__(self) -> str:
        """Generates a descriptive string for the strategy and its parameters."""
        params = self.get_params_for_repr()
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.scheme}({param_str})"

    @abstractmethod
    def get_params_for_repr(self) -> dict:
        """Returns a dictionary of parameters for the string representation."""
        pass

    def mask_video(self, video: CaptionedVideo) -> tuple[None, None] | tuple[CaptionedVideo, set[int]]:
        try:
            indices_to_mask: set = self.get_indices_to_mask(len(video.clips))
        except ValueError as e:
            logging.warning(f"Masking strategy {self} cannot be applied to video {video.video_id} (num_clips={len(video.clips)}): {e}")
            return None, None
        if not indices_to_mask:
            return None, None
        masked_clips = self.mask_list(video.clips, indices_to_mask)
        masked_video = video.model_copy(update={'clips': masked_clips})
        return masked_video, indices_to_mask


class RandomMasking(MaskingStrategy):
    """Masks a random selection of clips based on a ratio."""
    def __init__(self, ratio: float, prn_generator: random.Random):
        super().__init__("random")
        self.ratio = ratio
        self.prn = prn_generator

    def get_indices_to_mask(self, num_clips: int) -> set[int]:
        num_to_mask = int(num_clips * self.ratio)
        return set(self.prn.sample(range(num_clips), k=num_to_mask))

    def get_params_for_repr(self) -> dict:
        return {"ratio": self.ratio}


class FixedFillMasking(MaskingStrategy):

    def __init__(self, width:int, start_ind:int):
        super().__init__("fixed_fill")
        self.width:int = width
        self.start_ind:int = start_ind

    def get_params_for_repr(self) -> dict:
        return {"w": self.width, "i": self.start_ind}

    def get_indices_to_mask(self, num_clips: int) -> set[int]:
        """
        Calculates a set of indices to mask, starting from start_ind and
        expanding symmetrically until 'width' indices are collected.
        Handles boundaries by continuing expansion in the valid direction.

        Args:
            num_clips: The total number of available clips (indices are 0 to num_clips-1).

        Returns:
            A set of integer indices to be masked.
        """
        if self.start_ind >= num_clips:
            raise ValueError(f"start_ind ({self.start_ind}) must be less than num_clips ({num_clips}).")

        indices: set[int] = {self.start_ind}
        offset = 1

        # Clamp the width to not exceed the total number of clips
        target_width = min(self.width, num_clips) # Fixed off-by-one

        while len(indices) < target_width:
            right_idx = self.start_ind + offset
            if right_idx < num_clips:
                indices.add(right_idx)
                if len(indices) == target_width:
                    break

            left_idx = self.start_ind - offset
            if left_idx >= 0:
                indices.add(left_idx)
                if len(indices) == target_width:
                    break

            offset += 1

            # This safety break ensures the loop terminates if both directions
            # are exhausted. The bug was likely using 'or' here instead of 'and'.
            if (self.start_ind + offset > num_clips) and (self.start_ind - offset < 0):
                break

        return indices


class ContiguousMasking(MaskingStrategy):
    """
    A masking strategy that masks a single, contiguous block of clips.
    """

    def __init__(self, seed: int, width: int):
        super().__init__(scheme="contiguous")
        if not width > 0:
            raise ValueError("Masking width must be greater than 0.")

        self.seed = seed
        self.prn_generator = random.Random(seed)
        self.width = width

    def get_params_for_repr(self) -> dict:
        return {"seed": self.seed, "width": self.width}

    def get_indices_to_mask(self, num_clips: int) -> set[int]:
        """
        Determines the start index and returns the set of indices to be masked.
        """
        if self.width >= num_clips:
            return set()

        # The last possible starting position for the mask
        last_possible_start = num_clips - self.width

        # Choose a random starting index for the contiguous block
        start_index = self.prn_generator.randint(0, last_possible_start)

        # Create the set of indices to mask
        return set(range(start_index, start_index + self.width))

class PartitionMasking(MaskingStrategy):
    """A generic strategy that divides a sequence into partitions and masks a block."""
    def __init__(self, num_partitions: int, start_partition: int, num_parts_to_mask: int):
        super().__init__("partition")
        self.num_partitions = num_partitions
        self.start_partition = start_partition
        self.num_parts_to_mask = num_parts_to_mask

    def get_indices_to_mask(self, num_clips: int) -> set[int]:
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

    def get_params_for_repr(self) -> dict:
        return {"num_partitions": self.num_partitions, "start_partition": self.start_partition, "num_parts_to_mask": self.num_parts_to_mask}

def get_masking_strategies(masking_configs: list, master_seed: int) -> list[MaskingStrategy]:
    """
    Factory function that reads a list of masking configurations and generates
    a list of all specified masking strategy instances.
    """
    strategies = []

    for config in masking_configs:
        def get_list(fieldname:str) -> list:
            res = config.get(fieldname, [])
            if not isinstance(res, list):
                res = [res]
            return res

        scheme = config.get("scheme")
        if scheme == "random":
            seed = config.get("seed", 0) # TODO: if "seed" is a list, iterate over all values
            for ratio in config.get("ratio", []):
                strategies.append(RandomMasking(ratio=ratio, prn_generator=random.Random(master_seed+seed) ))
        elif scheme == "fixed_fill":
            for start_ind in get_list("start_ind"):
                for width in get_list("width"):
                    strategies.append(FixedFillMasking(width=width, start_ind=start_ind))
        elif scheme == "contiguous":
            for seed in get_list("seed"):
                for width in get_list("width"):
                    strategies.append(ContiguousMasking(seed=master_seed+seed, width=width))
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

