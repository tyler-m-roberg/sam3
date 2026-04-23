# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Tests for io_utils extensionless video file handling (D99228861)."""

import pickle
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

from sam3.model.io_utils import (
    IncrementalVideoFrameLoader,
    load_video_frames,
    load_video_frames_from_video_file,
)


class TestLoadVideoFramesRouting(unittest.TestCase):
    """Test that load_video_frames routes paths correctly based on extension."""

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_mp4_extension_routes_to_video_loader(self, mock_load_video):
        """Paths with .mp4 extension should route to load_video_frames_from_video_file."""
        mock_load_video.return_value = ("frames", 480, 640)
        result = load_video_frames(
            video_path="/tmp/test_video.mp4",
            image_size=256,
            offload_video_to_cpu=True,
        )
        mock_load_video.assert_called_once()
        self.assertEqual(result, ("frames", 480, 640))

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_mov_extension_routes_to_video_loader(self, mock_load_video):
        """Paths with .mov extension should route to load_video_frames_from_video_file."""
        mock_load_video.return_value = ("frames", 480, 640)
        load_video_frames(
            video_path="/tmp/test_video.mov",
            image_size=256,
            offload_video_to_cpu=True,
        )
        mock_load_video.assert_called_once()

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_extensionless_oil_path_routes_to_video_loader(self, mock_load_video):
        """Extensionless OIL paths should attempt video loading (D99228861 fix)."""
        mock_load_video.return_value = ("frames", 480, 640)
        result = load_video_frames(
            video_path="oil://fb_permanent/abc123def456",
            image_size=256,
            offload_video_to_cpu=True,
        )
        mock_load_video.assert_called_once()
        self.assertEqual(result, ("frames", 480, 640))

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_extensionless_bare_hash_routes_to_video_loader(self, mock_load_video):
        """Bare hash paths without extension should attempt video loading."""
        mock_load_video.return_value = ("frames", 480, 640)
        result = load_video_frames(
            video_path="/data/videos/abc123def456",
            image_size=256,
            offload_video_to_cpu=True,
        )
        mock_load_video.assert_called_once()
        self.assertEqual(result, ("frames", 480, 640))

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_extensionless_path_raises_on_decode_failure(self, mock_load_video):
        """Extensionless path that fails to decode should raise NotImplementedError."""
        mock_load_video.side_effect = RuntimeError("Could not decode video")
        with self.assertRaises(NotImplementedError) as ctx:
            load_video_frames(
                video_path="oil://fb_permanent/corrupted_file",
                image_size=256,
                offload_video_to_cpu=True,
            )
        self.assertIn("failed to load", str(ctx.exception))
        self.assertIn("oil://fb_permanent/corrupted_file", str(ctx.exception))

    @patch("sam3.model.io_utils.load_video_frames_from_image_folder")
    def test_directory_routes_to_image_folder_loader(self, mock_load_folder):
        """Directory paths should route to load_video_frames_from_image_folder."""
        mock_load_folder.return_value = ("frames", 480, 640)
        with tempfile.TemporaryDirectory() as tmpdir:
            load_video_frames(
                video_path=tmpdir,
                image_size=256,
                offload_video_to_cpu=True,
            )
            mock_load_folder.assert_called_once()

    def test_dummy_video_pattern(self):
        """<load-dummy-video-N> pattern should return dummy frames."""
        frames, h, w = load_video_frames(
            video_path="<load-dummy-video-5>",
            image_size=64,
            offload_video_to_cpu=True,
        )
        self.assertEqual(frames.shape[0], 5)  # 5 frames
        self.assertEqual(h, 480)
        self.assertEqual(w, 640)

    @patch("sam3.model.io_utils.load_video_frames_from_video_file")
    def test_unknown_extension_routes_to_video_loader(self, mock_load_video):
        """Paths with unrecognized extensions should attempt video loading."""
        mock_load_video.return_value = ("frames", 480, 640)
        result = load_video_frames(
            video_path="/tmp/video.xyz",
            image_size=256,
            offload_video_to_cpu=True,
        )
        mock_load_video.assert_called_once()
        self.assertEqual(result, ("frames", 480, 640))


class _FakeVideoReader:
    """Minimal stand-in for decord.VideoReader used in tests."""

    def __init__(self, num_frames=20, height=32, width=48):
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.access_count = {}

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        self.access_count[idx] = self.access_count.get(idx, 0) + 1
        # Each frame gets a unique constant value so we can detect mis-caching.
        return torch.full(
            (self.height, self.width, 3), fill_value=idx, dtype=torch.uint8
        )


class TestIncrementalVideoFrameLoader(unittest.TestCase):
    """Tests for the decord-backed incremental frame loader."""

    @patch("sam3.model.io_utils.IncrementalVideoFrameLoader")
    def test_decord_backend_routes_to_incremental_loader(self, mock_loader_cls):
        """video_loader_type='decord' should instantiate IncrementalVideoFrameLoader."""
        instance = MagicMock()
        instance.video_height = 480
        instance.video_width = 640
        mock_loader_cls.return_value = instance
        frames, h, w = load_video_frames_from_video_file(
            video_path="/tmp/test_video.mp4",
            image_size=256,
            offload_video_to_cpu=True,
            img_mean=(0.5, 0.5, 0.5),
            img_std=(0.5, 0.5, 0.5),
            async_loading_frames=False,
            video_loader_type="decord",
        )
        mock_loader_cls.assert_called_once()
        self.assertIs(frames, instance)
        self.assertEqual((h, w), (480, 640))

    def test_invalid_loader_type_raises(self):
        """Unknown video_loader_type values should raise a helpful error."""
        with self.assertRaises(RuntimeError) as ctx:
            load_video_frames_from_video_file(
                video_path="/tmp/test.mp4",
                image_size=256,
                offload_video_to_cpu=True,
                img_mean=(0.5, 0.5, 0.5),
                img_std=(0.5, 0.5, 0.5),
                async_loading_frames=False,
                video_loader_type="nope",
            )
        self.assertIn("decord", str(ctx.exception))

    def _make_loader(self, num_frames=20, cache_size=4):
        fake_reader = _FakeVideoReader(num_frames=num_frames)
        with patch.object(
            IncrementalVideoFrameLoader, "_open_reader", return_value=fake_reader
        ):
            loader = IncrementalVideoFrameLoader(
                video_path="<fake>",
                image_size=16,
                offload_video_to_cpu=True,
                img_mean=(0.5, 0.5, 0.5),
                img_std=(0.5, 0.5, 0.5),
                cache_size=cache_size,
            )
        return loader, fake_reader

    def test_basic_metadata(self):
        loader, _ = self._make_loader(num_frames=10)
        self.assertEqual(len(loader), 10)
        self.assertEqual(loader.video_height, 32)
        self.assertEqual(loader.video_width, 48)

    def test_getitem_returns_preprocessed_tensor(self):
        loader, _ = self._make_loader()
        frame = loader[0]
        self.assertEqual(frame.shape, (3, 16, 16))
        self.assertEqual(frame.dtype, torch.float16)

    def test_lru_cache_bounds_memory(self):
        loader, _ = self._make_loader(num_frames=20, cache_size=4)
        for i in range(20):
            _ = loader[i]
            self.assertLessEqual(len(loader._cache), 4)
        self.assertEqual(len(loader._cache), 4)
        # The most recent 4 frames should be resident.
        self.assertEqual(list(loader._cache.keys()), [16, 17, 18, 19])

    def test_repeated_access_hits_cache(self):
        loader, fake = self._make_loader(num_frames=10, cache_size=4)
        _ = loader[5]
        _ = loader[5]
        _ = loader[5]
        # Warmup opens the reader by reading frame 0 once for metadata.
        self.assertEqual(fake.access_count.get(5, 0), 1)

    def test_out_of_range_raises(self):
        loader, _ = self._make_loader(num_frames=5)
        with self.assertRaises(IndexError):
            _ = loader[10]

    def test_negative_index_supported(self):
        loader, _ = self._make_loader(num_frames=5)
        last = loader[-1]
        explicit = loader[4]
        self.assertTrue(torch.equal(last, explicit))

    def test_getstate_drops_reader_and_cache(self):
        loader, _ = self._make_loader(num_frames=5, cache_size=4)
        _ = loader[0]
        self.assertGreater(len(loader._cache), 0)
        state = loader.__getstate__()
        self.assertIsNone(state["_reader"])
        self.assertEqual(len(state["_cache"]), 0)

    def test_roundtrip_pickle_rebuilds_reader(self):
        loader, _ = self._make_loader(num_frames=5, cache_size=4)
        _ = loader[0]
        blob = pickle.dumps(loader)
        restored = pickle.loads(blob)
        self.assertIsNone(restored._reader)
        # Accessing a frame should lazily reopen via _open_reader.
        fake = _FakeVideoReader(num_frames=5)
        with patch.object(
            IncrementalVideoFrameLoader, "_open_reader", return_value=fake
        ):
            frame = restored[2]
        self.assertEqual(frame.shape, (3, 16, 16))
