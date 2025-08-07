import random
from abc import ABC, abstractmethod
from data_models import CaptionedClip
from data_models import CaptionedVideo


class MaskingStrategy(ABC):
    """Abstract base class for all masking strategies."""
    def __init__(self, scheme: str):
        self.scheme = scheme

    @abstractmethod
    def _get_indices_to_mask(self, num_clips: int) -> set:
        pass

    def mask_list(self, clips:list[CaptionedClip], indices_to_mask:set):
        masked_captions = []
        for clip in clips:
            if clip.index in indices_to_mask:
                masked_clip = clip.model_copy(update={"caption": None})
                masked_captions.append(masked_clip)
            else:
                masked_captions.append(clip)
        return masked_captions

    def apply(self, captions: list[CaptionedClip]) -> tuple[list[CaptionedClip], set]:
        indices_to_mask: set = self._get_indices_to_mask(len(captions))
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
        indices_to_mask: set = self._get_indices_to_mask(len(video.clips))
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

    def _get_indices_to_mask(self, num_clips: int) -> set:
        num_to_mask = int(num_clips * self.ratio)
        return set(self.prn.sample(range(num_clips), k=num_to_mask))

    def get_params_for_repr(self) -> dict:
        return {"ratio": self.ratio}


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

    def _get_indices_to_mask(self, num_clips: int) -> set[int]|None:
        """
        Determines the start index and returns the set of indices to be masked.
        """
        if self.width >= num_clips:
            return None

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

