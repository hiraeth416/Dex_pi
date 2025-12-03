"""Data loader for DexCanvas dataset converted to pi0 format.

This module provides a PyTorch Dataset and DataLoader for loading the converted
DexCanvas dataset. The dataset should be in the format created by
dexcanvas/scripts/convert_to_pi0_format.py.

Dataset Structure:
    dataset_dir/
        dataset_summary.json
        episode_000000/
            metadata.json
            states.npy          # [num_frames, robot_dof] - current frame's urdf_dof
            actions.npy         # [num_frames, robot_dof] - next frame's urdf_dof
            side_images/
                frame_000000.png
                frame_000001.png
                ...
            ego_images/
                frame_000000.png
                frame_000001.png
                ...
            mano/
                mano_rotations.npy
                mano_shape.npy
                ...
        episode_000001/
            ...
"""

import json
import logging
from pathlib import Path
from typing import Literal, SupportsIndex

import cv2
import jax
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)


class DexCanvasDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for DexCanvas data in pi0 format.
    
    The dataset uses DexCanvas camera naming conventions:
    - ego camera -> ego_rgb (main camera view)
    - side camera -> side_rgb (additional view, optional)
    
    Note: When using with pi0 models, you need to remap these keys to pi0's expected
    keys (base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb) in the transforms or
    pass custom image_keys to preprocess_observation_pytorch.
    
    Args:
        dataset_dir: Path to the dataset directory containing episode folders
        action_horizon: Number of future actions to return
        use_side_camera: If True, loads side camera as side_rgb, else fills with zeros
        load_images: Whether to load images (default: True)
    """
    
    def __init__(
        self,
        dataset_dir: str | Path,
        action_horizon: int = 50,
        use_side_camera: bool = False,
        load_images: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.action_horizon = action_horizon
        self.use_side_camera = use_side_camera
        self.load_images = load_images
        
        # Load dataset summary
        summary_path = self.dataset_dir / "dataset_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                self.summary = json.load(f)
        else:
            self.summary = {}
        
        # Find all episode directories
        self.episode_dirs = sorted([
            d for d in self.dataset_dir.iterdir()
            if d.is_dir() and d.name.startswith('episode_')
        ])
        
        if len(self.episode_dirs) == 0:
            raise ValueError(f"No episode directories found in {dataset_dir}")
        
        # Build index: (episode_idx, frame_idx) for each valid sample
        self.sample_indices = []
        self._build_index()
        
        logger.info(f"Loaded DexCanvas dataset from {dataset_dir}")
        logger.info(f"  Total episodes: {len(self.episode_dirs)}")
        logger.info(f"  Total samples: {len(self.sample_indices)}")
        logger.info(f"  Action horizon: {action_horizon}")
        logger.info(f"  Use side camera: {use_side_camera}")
    
    def _build_index(self):
        """Build an index of all valid samples (episode, frame) pairs."""
        for episode_idx, episode_dir in enumerate(self.episode_dirs):
            metadata_path = episode_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            num_frames = metadata['num_frames']
            
            # Each frame can be a starting point for a trajectory
            # We need at least action_horizon frames ahead
            for frame_idx in range(num_frames - self.action_horizon + 1):
                self.sample_indices.append((episode_idx, frame_idx))
    
    def __len__(self) -> int:
        return len(self.sample_indices)
    
    def __getitem__(self, index: SupportsIndex) -> dict:
        """Get a single sample.
        
        Returns a dictionary with:
            - image: dict with DexCanvas keys:
                - ego_rgb: [h, w, 3] uint8 RGB image (ego camera)
                - side_rgb: [h, w, 3] uint8 RGB image (side camera or zeros if not available)
            - image_mask: dict with {key: bool} for each camera
            - state: [robot_dof] float32 array (current frame's urdf_dof)
            - actions: [action_horizon, robot_dof] float32 array (next frame's urdf_dof)
            - prompt: str (text prompt)
        """
        episode_idx, frame_idx = self.sample_indices[index.__index__()]
        episode_dir = self.episode_dirs[episode_idx]
        
        # Load metadata
        metadata_path = episode_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load states and actions
        # States: current frame's urdf_dof [num_frames, robot_dof]
        # Actions: next frame's urdf_dof [num_frames, robot_dof]
        states = np.load(episode_dir / "states.npy")
        actions = np.load(episode_dir / "actions.npy")
        
        # Get current state
        state = states[frame_idx].astype(np.float32)
        
        # Get action sequence (next action_horizon steps)
        # Since actions[frame_idx] already contains the next frame's urdf_dof,
        # we can directly use actions[frame_idx:frame_idx + action_horizon]
        action_sequence = actions[frame_idx:frame_idx + self.action_horizon].astype(np.float32)
        
        # Pad if necessary (if we're near the end of the episode)
        if len(action_sequence) < self.action_horizon:
            padding = np.tile(action_sequence[-1:], (self.action_horizon - len(action_sequence), 1))
            action_sequence = np.concatenate([action_sequence, padding], axis=0)
        
        # Load image(s) if requested
        # Use DexCanvas camera naming: ego_rgb, side_rgb
        image_dict = {}
        image_mask_dict = {}
        
        if self.load_images:
            # Load ego camera (main camera)
            ego_image_path = episode_dir / "ego_images" / f"frame_{frame_idx:06d}.png"
            ego_img = cv2.imread(str(ego_image_path))
            if ego_img is None:
                raise ValueError(f"Failed to load ego image: {ego_image_path}")
            
            ego_rgb = cv2.cvtColor(ego_img, cv2.COLOR_BGR2RGB)
            image_dict["ego_rgb"] = ego_rgb
            image_mask_dict["ego_rgb"] = True
            
            # Load side camera if requested
            if self.use_side_camera:
                side_image_path = episode_dir / "side_images" / f"frame_{frame_idx:06d}.png"
                side_img = cv2.imread(str(side_image_path))
                if side_img is None:
                    logger.warning(f"Failed to load side image: {side_image_path}, using zeros")
                    image_dict["side_rgb"] = np.zeros_like(ego_rgb)
                    image_mask_dict["side_rgb"] = False
                else:
                    image_dict["side_rgb"] = cv2.cvtColor(side_img, cv2.COLOR_BGR2RGB)
                    image_mask_dict["side_rgb"] = True
            else:
                # Provide dummy image for side_rgb
                image_dict["side_rgb"] = np.zeros_like(ego_rgb)
                image_mask_dict["side_rgb"] = False
        
        return {
            "image": image_dict,
            "image_mask": image_mask_dict,
            "state": state,
            "actions": action_sequence,
            "prompt": metadata['text_prompt'],
        }


def create_dex_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    use_side_camera: bool = False,
) -> DexCanvasDataset:
    """Create a DexCanvas dataset for training.
    
    Args:
        data_config: Data configuration
        action_horizon: Number of future actions to return
        model_config: Model configuration
        use_side_camera: If True, loads side camera as left_wrist_0_rgb
    
    Returns:
        DexCanvasDataset instance
    """
    dataset_path = data_config.repo_id
    if dataset_path is None:
        raise ValueError("repo_id must be set to the path of the DexCanvas dataset directory")
    
    # Check if it's a fake dataset
    if dataset_path == "fake":
        from openpi.training.data_loader import FakeDataset
        return FakeDataset(model_config, num_samples=1024)
    
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    
    return DexCanvasDataset(
        dataset_dir=dataset_dir,
        action_horizon=action_horizon,
        use_side_camera=use_side_camera,
        load_images=True,
    )


def create_dex_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
):
    """Create a data loader for the DexCanvas dataset.
    
    This function creates a data loader that is compatible with the pi0 training pipeline.
    Images are mapped to standard pi0 camera keys:
    - ego camera -> base_0_rgb
    - side camera -> left_wrist_0_rgb (optional, controlled by use_side_camera in config)
    - right_wrist_0_rgb -> zeros (not available in DexCanvas)
    
    Args:
        config: Training configuration
        sharding: JAX sharding for distributed training
        shuffle: Whether to shuffle the dataset
        num_batches: Maximum number of batches to return
        skip_norm_stats: Whether to skip normalization
        framework: Framework to use ("jax" or "pytorch")
    
    Returns:
        DataLoader instance
    """
    from openpi.training.data_loader import DataLoaderImpl, TorchDataLoader, TransformedDataset
    
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"DexCanvas data_config: {data_config}")
    
    # Extract use_side_camera from the data config if it's a DexDataConfig
    use_side_camera = False
    if isinstance(config.data, _config.DexDataConfig):
        use_side_camera = config.data.use_side_camera
    
    # Create the base dataset
    dataset = create_dex_dataset(
        data_config, 
        config.model.action_horizon, 
        config.model, 
        use_side_camera=use_side_camera
    )
    
    # Apply transforms
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            logger.warning(
                "Normalization stats not found. Training without normalization. "
                "Run `scripts/compute_norm_stats.py --config-name=<your-config>` to compute them."
            )
        else:
            norm_stats = data_config.norm_stats
    
    dataset = TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )
    
    # Create PyTorch DataLoader
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            sampler = torch.utils.data.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
            )
            local_batch_size = config.batch_size // world_size
        else:
            local_batch_size = config.batch_size
    else:
        local_batch_size = config.batch_size // jax.process_count()
    
    logging.info(f"DexCanvas local_batch_size: {local_batch_size}")
    
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        framework=framework,
    )
    
    return DataLoaderImpl(data_config, data_loader)
