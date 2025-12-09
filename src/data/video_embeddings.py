
import numpy as np
from pathlib import Path
from collections import defaultdict
import math
import yaml
import logging

logger = logging.getLogger(__name__)

class VideoEmbedder:
    """
    A class to process video files and generate clip-based embeddings.
    """
    def __init__(self, model_name: str = 'vit_small_patch16_224', device: str = None):
        """
        Initializes the VideoEmbedder, loading the model and setting up the device.
        """
        try:
            global cv2, torch, timm, Image, transforms
            import cv2
            import torch
            import timm
            from PIL import Image
            from torchvision import transforms
        except ImportError as e:
            logger.error("Missing required libraries for VideoEmbedder. Please install 'opencv-python-headless' and 'timm'.")
            raise e

        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load the pre-trained model and move it to the appropriate device
        self.model: torch.nn.Module = timm.create_model(self.model_name, pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # Get the necessary image transformations for the model
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform: transforms.Compose = timm.data.create_transform(**data_config, is_training=False)

    def _extract_timestamped_frames(self, video_path: Path, fps: int) -> list[tuple[float, Image.Image]]:
        """
        Extracts frames from a video file at a given rate, returning them with their
        actual timestamps.
        """
        timestamped_frames: list[tuple[float, Image.Image]] = []
        vidcap = cv2.VideoCapture(str(video_path))
        if not vidcap.isOpened():
            logger.error(f"Error: Could not open video file {video_path.name}")
            return timestamped_frames

        video_fps: float = vidcap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0:
            logger.warning(f"Warning: Could not get FPS for {video_path.name}. Assuming 30.")
            video_fps = 30

        frame_interval = video_fps / fps
        current_frame_pos = 0.0

        while True:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, int(current_frame_pos))
            success, image = vidcap.read()
            if not success:
                break

            # Get the actual timestamp from the video capture, which is more accurate
            timestamp_msec = vidcap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_sec = timestamp_msec / 1000.0

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            timestamped_frames.append((timestamp_sec, Image.fromarray(image)))

            current_frame_pos += frame_interval

        vidcap.release()
        logger.debug(f"Extracted {len(timestamped_frames)} frames from {video_path.name} at {fps} FPS.")
        return timestamped_frames

    def _get_frame_embeddings(self, frames: list[Image.Image]) -> list[np.ndarray]:
        """
        Generates a list of vector embeddings, one for each frame.
        (Private method for internal use)
        """
        if not frames:
            return []

        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for frame in frames:
                img_tensor: torch.Tensor = self.transform(frame).unsqueeze(0).to(self.device)
                embedding: torch.Tensor = self.model.forward_features(img_tensor)
                embeddings.append(embedding[:, 0].cpu().numpy().flatten())

        return embeddings

    def _group_and_average_embeddings(self, timestamps: list[float], embeddings: list[np.ndarray], clip_size: int) -> np.ndarray:
        """
        Groups embeddings by clip and averages them to get one vector per clip.
        (Private method for internal use)
        """
        if not embeddings:
            return np.array([])

        grouped_embeddings = defaultdict(list)
        for timestamp, embedding in zip(timestamps, embeddings):
            clip_index = int(timestamp // clip_size)
            grouped_embeddings[clip_index].append(embedding)

        averaged_embeddings = [np.mean(grouped_embeddings[key], axis=0) for key in sorted(grouped_embeddings.keys())]

        return np.array(averaged_embeddings)

    def process_directory(self, video_dir: Path, output_dir: Path, fps: int, clip_size: int = 1):
        """
        Finds MP4s, generates embeddings, and saves one averaged vector per clip_size.
        """
        if not video_dir.exists():
            logger.error(f"Error: Video directory not found at '{video_dir}'")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir/"metadata.yaml",  'w') as f:
            yaml.dump({
                "type": "video_embeddings",
                "model_name": self.model_name,
                "fps": fps,
                "clip_size": clip_size,
                "input": video_dir.name
            }, f, default_flow_style=False, sort_keys=False)


        video_files = list(video_dir.glob("*.mp4"))
        logger.info(f"Found {len(video_files)} videos to process.")

        for video_path in video_files:
            output_filepath = output_dir / f"{video_path.stem}.npy"
            if output_filepath.exists():
                logger.info(f"Skipping {video_path.name}, output already exists.")
                continue

            logger.info(f"--- Processing: {video_path.name} ---")

            timestamped_frames = self._extract_timestamped_frames(video_path, fps=fps)
            if not timestamped_frames:
                logger.warning(f"Skipping video {video_path.name} as no frames were extracted.")
                continue

            timestamps, frames = zip(*timestamped_frames)

            frame_embeddings = self._get_frame_embeddings(list(frames))
            if not frame_embeddings:
                logger.warning(f"Skipping video {video_path.name} as no embeddings were generated.")
                continue

            # Optimization: If fps and clip_size are both 1, we can use the frame embeddings directly.
            if fps == 1 and clip_size == 1:
                clip_embeddings = np.array(frame_embeddings)
            else:
                clip_embeddings = self._group_and_average_embeddings(list(timestamps), frame_embeddings, clip_size)

            # --- VALIDATION STEP ---
            # Base the expected number of clips on the timestamp of the LAST extracted frame.
            last_timestamp = timestamps[-1]
            expected_clips = math.ceil(last_timestamp / clip_size)

            # We allow a small tolerance (e.g., 1) for rounding issues at the very end of the video.
            if abs(clip_embeddings.shape[0] - expected_clips) > 1:
                 logger.warning(
                    f"Discrepancy in '{video_path.name}': "
                    f"Expected ~{expected_clips} vectors based on extracted frames, but generated {clip_embeddings.shape[0]}. "
                    "This is likely due to video encoding/timestamp irregularities."
                )

            np.save(output_filepath, clip_embeddings)
            logger.info(f"Saved {clip_embeddings.shape[0]} vectors to {output_filepath} with shape {clip_embeddings.shape}")
