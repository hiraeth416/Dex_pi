"""Compute normalization statistics for DexCanvas VLA dataset.

Simplified version that directly uses dataset_vla and DataLoader.
"""

import sys
from pathlib import Path
import numpy as np
import tqdm
import tyro
from torch.utils.data import DataLoader

import openpi.shared.normalize as normalize

# Add dexcanvas to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dexcanvas"))
from dexcanvas import DexCanvasDataset_VLA


def main(
    dataset_path: str = "/home/sifei/openpi/dexcanvas/data_source/mocap/releases/v0.5/trajectories_preprocessed.lance",
    output_dir: str = "/home/sifei/openpi/dataset_dex",
    batch_size: int = 1,
    num_workers: int = 0,
    max_batches: int | None = None,
):
    """Compute normalization statistics directly from DexCanvas VLA dataset.
    
    Args:
        dataset_path: Path to dataset (HF Hub ID or local path)
        output_dir: Where to save norm_stats.json
        batch_size: Batch size for data loading
        num_workers: Number of workers for DataLoader
        max_batches: Maximum number of batches to process (None = all)
    """
    print(f"Loading dataset from {dataset_path}...")
    dataset_vla = DexCanvasDataset_VLA(
        dataset_path=dataset_path,
        load_active_only=True,
    )
    
    print(f"Creating DataLoader with batch_size={batch_size}, num_workers={num_workers}...")
    dataloader = DataLoader(
        dataset_vla,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    
    # Initialize running statistics for state and actions
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}
    
    # Process batches
    total = max_batches if max_batches else len(dataloader)
    print(f"Computing statistics from {total} batches...")
    
    for i, batch in enumerate(tqdm.tqdm(dataloader, total=total, desc="Computing stats")):
        if max_batches and i >= max_batches:
            break
        #import ipdb; ipdb.set_trace()
        # batch["mano_urdf_dof"] has shape [batch_size, seq_len, robot_dof]
        # Flatten to [batch_size * seq_len, robot_dof] for statistics
        urdf_dof = batch["mano_urdf_dof"].numpy()
        batch_size_actual, seq_len, robot_dof = urdf_dof.shape
        urdf_dof_flat = urdf_dof.reshape(-1, robot_dof)
        
        stats["state"].update(urdf_dof_flat)
        stats["actions"].update(urdf_dof_flat)
    
    # Get final statistics
    norm_stats = {key: stat.get_statistics() for key, stat in stats.items()}
    
    
    # Save to output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    print("Done!")


if __name__ == "__main__":
    tyro.cli(main)
