import sys
import types
import importlib.machinery
from pathlib import Path
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from transformers import AutoTokenizer, SiglipTextModel

from data_models.captions_only import CaptionedClip, CaptionedVideo, TimestampRange
from reconstruction.text_reconstruction import Reconstructed
from evaluations.evaluation import (
    ReconstructionEvaluator,
    ReconstructionEvaluator_CrossModal,
    ReconstructionEvaluator_EmbSimilarity,
)
from data.vector_dataloaders import VectorDataLoader
from data.video_embeddings import VideoEmbedder
from llm.local_embedder import SiglipTextEmbedder


class TestSiglipFactoryRouting:
    def test_evaluation_factory_siglip_text_sim(self):
        mock_tok = MagicMock()
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768

        with patch.object(AutoTokenizer, "from_pretrained", return_value=mock_tok), \
             patch.object(SiglipTextModel, "from_pretrained", return_value=mock_model):

            conf = {
                "type": "emb_sim",
                "embedding_model": "local:siglip",
            }
            evaluator = ReconstructionEvaluator.from_config(conf)
            assert isinstance(evaluator, ReconstructionEvaluator_EmbSimilarity)
            assert isinstance(evaluator._embedder, SiglipTextEmbedder)
            assert evaluator._embedder.model_name == "google/siglip-base-patch16-224"

    def test_evaluation_factory_cross_modal(self, tmp_path):
        mock_tok = MagicMock()
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768

        with patch.object(AutoTokenizer, "from_pretrained", return_value=mock_tok), \
             patch.object(SiglipTextModel, "from_pretrained", return_value=mock_model):

            conf = {
                "type": "cross_modal_sim",
                "embedding_model": "local:siglip",
                "video_embs_path": str(tmp_path),
            }
            evaluator = ReconstructionEvaluator.from_config(conf)
            assert isinstance(evaluator, ReconstructionEvaluator_CrossModal)
            assert isinstance(evaluator._embedder, SiglipTextEmbedder)
            assert evaluator.video_embs_path == tmp_path

    def test_vector_dataloader_factory_siglip(self):
        mock_tok = MagicMock()
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768

        with patch.object(AutoTokenizer, "from_pretrained", return_value=mock_tok), \
             patch.object(SiglipTextModel, "from_pretrained", return_value=mock_model), \
             patch("data.vector_dataloaders.get_data_loader") as mock_get_loader:

            mock_base_loader = MagicMock()
            mock_base_loader.get_data_type_name.return_value = "dummy"
            mock_get_loader.return_value = mock_base_loader

            data_conf = {
                "name": "wild_captions",
                "embedding_model": "local:siglip",
            }
            loader = VectorDataLoader.from_config(data_conf)
            assert isinstance(loader.embedder, SiglipTextEmbedder)
            assert loader.embedder.model_name == "google/siglip-base-patch16-224"


class TestReconstructionEvaluatorCrossModal:
    def setup_method(self):
        self.mock_embedder = MagicMock()
        self.video_id = "test_vid_001"
        self.video_embs_dir = Path("/tmp/test_siglip_embs")
        self.video_embs_dir.mkdir(parents=True, exist_ok=True)

        self.evaluator = ReconstructionEvaluator_CrossModal(
            embedder=self.mock_embedder,
            video_embs_path=str(self.video_embs_dir),
        )

        # Create dummy video with 3 clips
        self.orig_clips = [
            CaptionedClip(index=0, timestamp=TimestampRange(start=0.0, duration=1.0), caption="A red car"),
            CaptionedClip(index=1, timestamp=TimestampRange(start=1.0, duration=1.0), caption="A dog running"),
            CaptionedClip(index=2, timestamp=TimestampRange(start=2.0, duration=1.0), caption="A blue sky"),
        ]
        self.orig_video = CaptionedVideo(video_id=self.video_id, clips=self.orig_clips)

    def test_cross_modal_evaluation_success(self):
        video_embs = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float64)
        np.save(self.video_embs_dir / f"{self.video_id}.npy", video_embs)

        recon = Reconstructed(
            video_id=self.video_id,
            reconstructed_captions={1: "A puppy sprinting"},
        )

        self.mock_embedder.get_embeddings.return_value = [[0.0, 1.0, 0.0, 0.0]]

        metrics = self.evaluator.evaluate(recon, self.orig_video)
        assert "cos_sim" in metrics
        assert np.isclose(metrics["cos_sim"][0], 1.0, atol=1e-5)
        assert "cos_sim_residual" in metrics

    def test_cross_modal_missing_npy_file(self):
        recon = Reconstructed(
            video_id="non_existent_video",
            reconstructed_captions={0: "Some text"},
        )
        video = CaptionedVideo(video_id="non_existent_video", clips=[self.orig_clips[0]])
        metrics = self.evaluator.evaluate(recon, video)
        assert metrics == {}

    def test_cross_modal_empty_reconstructed(self):
        recon = Reconstructed(
            video_id=self.video_id,
            reconstructed_captions={},
        )
        metrics = self.evaluator.evaluate(recon, self.orig_video)
        assert metrics == {}

    def test_cross_modal_index_out_of_bounds(self):
        video_embs = np.zeros((2, 4), dtype=np.float64)
        np.save(self.video_embs_dir / f"{self.video_id}.npy", video_embs)

        recon = Reconstructed(
            video_id=self.video_id,
            reconstructed_captions={5: "Out of bounds text"},
        )
        metrics = self.evaluator.evaluate(recon, self.orig_video)
        assert metrics == {}


class TestSiglipTextEmbedder:
    def test_embed_new_padding_and_normalization(self):
        mock_tok = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.config.hidden_size = 768

        mock_tok.return_value = {
            "input_ids": torch.zeros((2, 64), dtype=torch.long),
            "attention_mask": torch.ones((2, 64), dtype=torch.long),
        }

        raw_output = torch.tensor([
            [3.0] * 768,
            [4.0] * 768,
        ], dtype=torch.float32)
        mock_model_output = MagicMock()
        mock_model_output.pooler_output = raw_output
        mock_model.return_value = mock_model_output

        with patch.object(AutoTokenizer, "from_pretrained", return_value=mock_tok), \
             patch.object(SiglipTextModel, "from_pretrained", return_value=mock_model):

            embedder = SiglipTextEmbedder(device="cpu")
            texts = ["caption one", "caption two"]
            results = embedder._embed_new("vid1", texts)

            mock_tok.assert_called_once_with(
                texts,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt",
            )

            assert set(results.keys()) == set(texts)
            for t in texts:
                vec = np.array(results[t])
                assert vec.shape == (768,)
                norm = np.linalg.norm(vec)
                assert np.isclose(norm, 1.0, atol=1e-5)


@pytest.fixture
def mock_video_dependencies():
    """Provides mock modules for timm, cv2, and torchvision on CPU test env."""
    mock_timm = types.ModuleType("timm")
    mock_timm.__spec__ = importlib.machinery.ModuleSpec("timm", None)
    mock_timm.create_model = MagicMock()
    mock_timm_data = types.ModuleType("timm.data")
    mock_timm_data.__spec__ = importlib.machinery.ModuleSpec("timm.data", None)
    mock_timm_data.resolve_model_data_config = MagicMock()
    mock_timm_data.create_transform = MagicMock()
    mock_timm.data = mock_timm_data

    mock_cv2 = types.ModuleType("cv2")
    mock_cv2.__spec__ = importlib.machinery.ModuleSpec("cv2", None)

    mock_tv = types.ModuleType("torchvision")
    mock_tv.__spec__ = importlib.machinery.ModuleSpec("torchvision", None)
    mock_tv.__path__ = []
    mock_tv_transforms = types.ModuleType("torchvision.transforms")
    mock_tv_transforms.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", None)
    mock_tv.transforms = mock_tv_transforms

    with patch.dict(sys.modules, {
        "timm": mock_timm,
        "timm.data": mock_timm_data,
        "cv2": mock_cv2,
        "torchvision": mock_tv,
        "torchvision.transforms": mock_tv_transforms,
    }):
        yield mock_timm


class TestVideoEmbedderSigLIP:
    def test_video_embedder_init_siglip(self, mock_video_dependencies):
        mock_timm = mock_video_dependencies
        mock_model = MagicMock()
        mock_timm.create_model.return_value = mock_model

        embedder = VideoEmbedder(model_name="vit_base_patch16_siglip_224", device="cpu")
        assert embedder.is_siglip is True
        mock_timm.create_model.assert_called_once_with("vit_base_patch16_siglip_224", pretrained=True, num_classes=0)

    def test_video_embedder_init_vit_small(self, mock_video_dependencies):
        mock_timm = mock_video_dependencies
        mock_model = MagicMock()
        mock_timm.create_model.return_value = mock_model

        embedder = VideoEmbedder(model_name="vit_small_patch16_224", device="cpu")
        assert embedder.is_siglip is False
        mock_timm.create_model.assert_called_once_with("vit_small_patch16_224", pretrained=True)

    def test_get_frame_embeddings_siglip_vs_vit(self, mock_video_dependencies):
        from PIL import Image
        mock_timm = mock_video_dependencies
        mock_model = MagicMock()
        mock_timm.create_model.return_value = mock_model
        mock_timm.data.create_transform.return_value = lambda img: torch.ones((3, 224, 224))

        dummy_frame = Image.new("RGB", (224, 224))

        # 1. SigLIP mode: calls model(img_tensor) and normalizes
        embedder_siglip = VideoEmbedder(model_name="vit_base_patch16_siglip_224", device="cpu")
        mock_model.return_value = torch.tensor([[2.0, 0.0, 0.0, 0.0]])

        embs = embedder_siglip._get_frame_embeddings([dummy_frame])
        assert len(embs) == 1
        assert np.isclose(np.linalg.norm(embs[0]), 1.0, atol=1e-5)
        mock_model.assert_called()

        # 2. ViT-Small mode: calls model.forward_features(img_tensor)[:, 0]
        mock_model.reset_mock()
        embedder_vit = VideoEmbedder(model_name="vit_small_patch16_224", device="cpu")
        mock_features = torch.zeros((1, 197, 384))
        mock_features[0, 0, :] = 5.0
        mock_model.forward_features.return_value = mock_features

        embs_vit = embedder_vit._get_frame_embeddings([dummy_frame])
        assert len(embs_vit) == 1
        assert embs_vit[0].shape == (384,)
        assert embs_vit[0][0] == 5.0
        mock_model.forward_features.assert_called_once()
