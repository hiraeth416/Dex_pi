"""Direct data loader for DexCanvas VLA dataset without intermediate conversion.

This module provides a PyTorch Dataset and DataLoader for loading the DexCanvas VLA
dataset directly, computing states and actions on-the-fly from mano_urdf_dof.
This eliminates the need for the intermediate conversion step.

Usage:
    dataset = DexCanvasDirectDataset(
        dataset_path="DEXROBOT/DexCanvas",  # or None to use default HF Hub
        action_horizon=50,
        max_len=256,
        load_active_only=True,
    )
"""

import io
import logging
from pathlib import Path
from typing import Literal, SupportsIndex

import jax
import numpy as np
import torch
from PIL import Image

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

# Import DexCanvas VLA dataset
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "dexcanvas"))
from dexcanvas import DexCanvasDataset_VLA, InfiniteDataset_VLA

logger = logging.getLogger(__name__)


def convert_to_robot_state_and_actions(
    urdf_dof: np.ndarray, action_horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert URDF DOF to robot states and actions for entire trajectory.
    
    States: Current frame's urdf_dof
    Actions: Next frame's urdf_dof (for last frame, use current frame)
    
    Args:
        urdf_dof: [seq_len, robot_dof] URDF degrees of freedom
        action_horizon: Number of future actions to return
    
    Returns:
        Tuple of (states, actions):
            - states: [seq_len, robot_dof] current frames' urdf_dof
            - actions: [seq_len, action_horizon, robot_dof] future actions
    """
    seq_len, robot_dof = urdf_dof.shape
    
    # States are the current frame's urdf_dof
    states = urdf_dof.copy()
    
    # Actions are the next action_horizon frames' urdf_dof
    actions = np.zeros((seq_len, action_horizon, robot_dof), dtype=np.float32)
    
    for i in range(seq_len):
        # Get next action_horizon frames
        end_idx = min(i + action_horizon, seq_len)
        num_actions = end_idx - i
        actions[i, :num_actions] = urdf_dof[i:end_idx]
        
        # Pad with last available action if needed
        if num_actions < action_horizon:
            actions[i, num_actions:] = urdf_dof[end_idx - 1]
    
    return states.astype(np.float32), actions


def decode_images(image_data: list) -> list[np.ndarray]:
    """
    Decode image bytes to numpy arrays in RGB format.
    
    Args:
        image_data: List of image bytes (possibly wrapped in tuples)
    
    Returns:
        List of numpy arrays in RGB format [H, W, 3]
    """
    images = []
    for frame_item in image_data:
        # Handle tuple wrapping if present (e.g. (bytes,))
        img_bytes = frame_item[0] if isinstance(frame_item, (list, tuple)) else frame_item
        img = Image.open(io.BytesIO(img_bytes))
        img = np.array(img)  # Keep as RGB
        images.append(img)
    
    return images


class DexCanvasDirectDataset(torch.utils.data.IterableDataset):
    """PyTorch IterableDataset for DexCanvas VLA data.
    
    This dataset loads DexCanvas VLA data directly and converts it on-the-fly
    to the format required for pi0 training. Each trajectory is processed frame by frame,
    yielding one sample per frame.
    
    Camera naming convention:
    - camera1/video_color[0] -> side_rgb (side camera view)
    - camera2/video_color[1] -> ego_rgb (ego camera view)
    
    Args:
        dataset_path: Path to DexCanvas dataset or HuggingFace Hub name (default: None, uses HF Hub)
        action_horizon: Number of future actions to return
        max_len: Maximum sequence length for trajectory loading
        load_active_only: If True, loads only active manipulation frames
        use_side_camera: If True, loads side camera as side_rgb
        min_rating: Minimum trajectory rating filter
        objects: Filter by object names
        manipulation_types: Filter by manipulation types
    """
    
    def __init__(
        self,
        dataset_path: str | None = None,
        action_horizon: int = 50,
        max_len: int = 1024,
        load_active_only: bool = True,
        use_side_camera: bool = False,
        min_rating: float | None = None,
        objects: list[str] | None = None,
        manipulation_types: list[str] | None = None,
    ):
        self.action_horizon = action_horizon
        self.use_side_camera = use_side_camera
        
        # Create the base DexCanvas VLA dataset
        self.base_dataset = DexCanvasDataset_VLA(
            dataset_path=dataset_path,
            min_rating=min_rating,
            objects=objects,
            manipulation_types=manipulation_types,
            max_len=max_len,
            load_active_only=load_active_only,
            device='cpu',
        )
        
        logger.info(f"Initialized DexCanvas direct dataset")
        logger.info(f"  Action horizon: {action_horizon}")
        logger.info(f"  Max length: {max_len}")
        logger.info(f"  Load active only: {load_active_only}")
        logger.info(f"  Use side camera: {use_side_camera}")
    
    def __len__(self) -> int:
        """Return number of trajectories in the dataset."""
        try:
            return len(self.base_dataset)
        except Exception as e:
            raise TypeError("DexCanvasDirectDataset length is not available.") from e
    
    def __iter__(self):
        """Iterate over the dataset, yielding one sample per frame."""
        for traj_data in self.base_dataset:
            # Extract data
            mano_urdf_dof = traj_data['mano_urdf_dof'].numpy()  # [seq_len, robot_dof]
            actual_length = int(traj_data['lengths'].item())
            metadata = traj_data['metadata']
            
            # Decode video frames
            video_color = traj_data['image']['video_color']
            side_images = decode_images(video_color[0]['color'])  # camera1 = side
            ego_images = decode_images(video_color[1]['color'])   # camera2 = ego
            
            # Truncate to actual length
            mano_urdf_dof = mano_urdf_dof[:actual_length]
            side_images = side_images[:actual_length]
            ego_images = ego_images[:actual_length]
            
            # Create text prompt
            object_name = metadata.get('object', 'unknown')
            text_prompt = f"grasp the {object_name}"
            
            # Convert URDF DOF to states and actions for entire trajectory
            states, actions = convert_to_robot_state_and_actions(
                mano_urdf_dof, self.action_horizon
            )
            
            # Yield one sample per frame
            for frame_idx in range(actual_length):
                # Get images for this frame
                ego_img = ego_images[frame_idx]
                
                # Build image dict
                image_dict = {
                    "ego_rgb": ego_img,
                }
                image_mask_dict = {
                    "ego_rgb": True,
                }
                
                if self.use_side_camera:
                    side_img = side_images[frame_idx]
                    image_dict["side_rgb"] = side_img
                    image_mask_dict["side_rgb"] = True
                else:
                    # Fill with zeros if side camera not requested
                    image_dict["side_rgb"] = np.zeros_like(ego_img)
                    image_mask_dict["side_rgb"] = False
                
                yield {
                    "image": image_dict,
                    "image_mask": image_mask_dict,
                    "state": states[frame_idx],
                    "actions": actions[frame_idx],
                    "prompt": text_prompt,
                }


def create_dex_direct_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    use_side_camera: bool = False,
    max_len: int = 256,
    min_rating: float | None = None,
) -> DexCanvasDirectDataset:
    """Create a DexCanvas direct dataset for training.
    
    Args:
        data_config: Data configuration
        action_horizon: Number of future actions to return
        model_config: Model configuration
        use_side_camera: If True, loads side camera as side_rgb
        max_len: Maximum sequence length for trajectory loading
        min_rating: Minimum trajectory rating filter
    
    Returns:
        DexCanvasDirectDataset instance
    """
    #dataset_path = data_config.repo_id
    dataset_path = "/home/sifei/openpi/dexcanvas/data_source/mocap/releases/v0.5/trajectories_preprocessed.lance"
    
    # Check if it's a fake dataset
    if dataset_path == "fake":
        from openpi.training.data_loader import FakeDataset
        return FakeDataset(model_config, num_samples=1024)
    
    dataset = DexCanvasDirectDataset(
        dataset_path=dataset_path,
        action_horizon=action_horizon,
        max_len=max_len,
        load_active_only=True,
        use_side_camera=use_side_camera,
        min_rating=min_rating,
    )
    
    return dataset


def create_dex_direct_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
):
    """Create a direct data loader for the DexCanvas VLA dataset.
    
    This function creates a data loader that reads directly from DexCanvas VLA format
    without requiring intermediate conversion. Images are mapped to standard pi0 camera keys:
    - ego camera (camera2) -> base_0_rgb
    - side camera (camera1) -> left_wrist_0_rgb (optional, controlled by use_side_camera in config)
    - right_wrist_0_rgb -> zeros (not available in DexCanvas)
    
    Args:
        config: Training configuration
        sharding: JAX sharding for distributed training
        shuffle: Whether to shuffle the dataset (Note: handled by dataset itself via HF datasets)
        num_batches: Maximum number of batches to return
        skip_norm_stats: Whether to skip normalization
        framework: Framework to use ("jax" or "pytorch")
    
    Returns:
        DataLoader instance
    """
    from openpi.training.data_loader import DataLoaderImpl, TorchDataLoader

    class IterableTransformedDataset(torch.utils.data.IterableDataset):
        def __init__(self, dataset: torch.utils.data.IterableDataset, transforms: list):
            self._dataset = dataset
            self._transform = _transforms.compose(transforms)

        def __iter__(self):
            for sample in self._dataset:
                yield self._transform(sample)

        def __len__(self) -> int:
            return len(self._dataset)

    
    data_config = config.data.create(config.assets_dirs, config.model)
    #logging.info(f"DexCanvas direct data_config: {data_config}")
    
    # Extract parameters from the data config if it's a DexDataConfig
    use_side_camera = False
    max_len = 256
    min_rating = None
    
    if isinstance(config.data, _config.DexDataConfig):
        use_side_camera = config.data.use_side_camera
        # Could add more config options here if needed
    
    # Create the base dataset
    dataset = create_dex_direct_dataset(
        data_config, 
        config.model.action_horizon, 
        config.model, 
        use_side_camera=use_side_camera,
        max_len=max_len,
        min_rating=min_rating,
    )

    if shuffle:
        if dataset.base_dataset.supports_shuffle:
            dataset.base_dataset.shuffle(buffer_size=1000)
        else:
            logging.warning("Dataset does not support shuffling. Training will proceed with sequential data.")
    
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Norm stats must be provided for DexCanvas direct dataset. "
                "Run compute_dex_norm_stats.py first."
            )
        else:
            norm_stats = data_config.norm_stats
    
    dataset = IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )
    
    # Create PyTorch DataLoader for IterableDataset
    # Note: IterableDataset doesn't support DistributedSampler
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            local_batch_size = config.batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = config.batch_size
    else:
        local_batch_size = config.batch_size // jax.process_count()
    
    logging.info(f"DexCanvas direct local_batch_size: {local_batch_size}")
    
    # IterableDataset doesn't support num_workers > 0 without proper worker splitting
    # Force num_workers=0 for direct loader
    num_workers = 0
    logging.info(f"Using num_workers={num_workers} for IterableDataset (DexCanvas direct loader)")
    
    # Use torch.utils.data.DataLoader directly for IterableDataset
    # Use torch.utils.data.DataLoader directly for IterableDataset
    if framework == "jax":
        if sharding is None:
            # Use data parallel sharding by default
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
    
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=False, # Force shuffle=False for IterableDataset to avoid DataLoader error
        num_batches=num_batches,
        num_workers=num_workers,
    )
    
    return DataLoaderImpl(data_config, data_loader)

