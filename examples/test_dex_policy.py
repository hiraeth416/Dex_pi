"""Test the trained DexCanvas policy on real dataset examples.

This script:
1. Loads a trained policy
2. Loads real examples from the DexCanvas dataset
3. Runs inference to get predicted actions
4. Compares predictions with ground truth actions
5. Outputs both for analysis
"""

import dataclasses
import json
from pathlib import Path

import jax
import numpy as np

from openpi.models import model as _model
from openpi.policies import dex_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader_dex_direct as _data_loader_direct


def print_action_comparison(pred_actions: np.ndarray, gt_actions: np.ndarray, num_steps: int = 5):
    """Print a comparison of predicted vs ground truth actions.
    
    Args:
        pred_actions: Predicted actions [action_horizon, action_dim]
        gt_actions: Ground truth actions [action_horizon, action_dim]
        num_steps: Number of timesteps to display
    """
    print("\n" + "=" * 80)
    print("ACTION COMPARISON (Predicted vs Ground Truth)")
    print("=" * 80)
    
    # Get the actual number of DOF (26 for DexCanvas)
    dof = 26
    
    # Truncate to show only first num_steps
    num_steps = min(num_steps, len(pred_actions))
    
    for t in range(num_steps):
        print(f"\n--- Timestep {t} ---")
        pred = pred_actions[t, :dof]
        gt = gt_actions[t, :dof]
        
        # Compute error metrics
        abs_error = np.abs(pred - gt)
        mean_error = np.mean(abs_error)
        max_error = np.max(abs_error)
        
        print(f"Predicted action:    [{', '.join(f'{x:7.4f}' for x in pred[:5])}...] (first 5 DOF)")
        print(f"Ground truth action: [{', '.join(f'{x:7.4f}' for x in gt[:5])}...] (first 5 DOF)")
        print(f"Mean absolute error: {mean_error:.6f}")
        print(f"Max absolute error:  {max_error:.6f}")
    
    print("\n" + "=" * 80)


def compute_metrics(pred_actions: np.ndarray, gt_actions: np.ndarray):
    """Compute error metrics over the entire action sequence.
    
    Args:
        pred_actions: Predicted actions [action_horizon, action_dim]
        gt_actions: Ground truth actions [action_horizon, action_dim]
    
    Returns:
        Dictionary with various metrics
    """
    # Use only the actual DexCanvas DOF (26)
    dof = 26
    pred = pred_actions[:, :dof]
    gt = gt_actions[:, :dof]
    
    # Compute various error metrics
    abs_error = np.abs(pred - gt)
    squared_error = (pred - gt) ** 2
    
    metrics = {
        "mean_absolute_error": np.mean(abs_error),
        "max_absolute_error": np.max(abs_error),
        "mean_squared_error": np.mean(squared_error),
        "rmse": np.sqrt(np.mean(squared_error)),
        # Per-timestep metrics
        "mae_per_timestep": np.mean(abs_error, axis=1),
        "mse_per_timestep": np.mean(squared_error, axis=1),
    }
    
    return metrics


def test_policy_on_dataset(
    config_name: str = "dex_pi05",
    checkpoint_dir: str = "checkpoints/dex_pi05/dex_test_pi05/30000",
    num_samples: int = 5,
):
    """Test the trained policy on real dataset examples.
    
    Args:
        config_name: Name of the config to use
        checkpoint_dir: Path to the checkpoint directory
        num_samples: Number of samples to test
    """
    print(f"Loading config: {config_name}")
    config = _config.get_config(config_name)
    
    print(f"Loading policy from: {checkpoint_dir}")
    # Create a trained policy
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    
    # Create the raw dataset directly (without transforms)
    print(f"Loading raw dataset...")
    raw_dataset = _data_loader_direct.create_dex_direct_dataset(
        data_config=config.data.create(config.assets_dirs, config.model),
        action_horizon=config.model.action_horizon,
        model_config=config.model,
        use_side_camera=False,
        max_len=256,
        min_rating=None,
    )
    
    print(f"Dataset loaded, iterating through {num_samples} samples...")
    
    # Store all metrics
    all_metrics = []
    
    # Get iterator from raw dataset
    data_iter = iter(raw_dataset)
    
    # Test on num_samples examples
    for i in range(num_samples):
        print(f"\n{'='*80}")
        print(f"TESTING SAMPLE {i+1}/{num_samples}")
        print(f"{'='*80}")
        
        try:
            # Get a raw sample from the dataset (without transforms)
            sample = next(data_iter)
            import ipdb; ipdb.set_trace()
            
            # Get ground truth actions (before any padding or normalization)
            gt_actions = sample["actions"].copy()  # [action_horizon, robot_dof]
            
            print(f"Prompt: {sample['prompt']}")
            print(f"State shape: {sample['state'].shape}")
            print(f"Ground truth actions shape: {gt_actions.shape}")
            print(f"Images: {list(sample['image'].keys())}")
            
            # Run inference (this will apply all transforms internally)
            result = policy.infer(sample)
            
            # Extract predictions
            pred_actions = np.asarray(result["actions"])  # [action_horizon, action_dim]
            
            print(f"\nPredicted actions shape: {pred_actions.shape}")
            print(f"Ground truth actions shape: {gt_actions.shape}")
            
            # Compute and display metrics
            metrics = compute_metrics(pred_actions, gt_actions)
            all_metrics.append(metrics)
            
            print(f"\nMetrics for sample {i+1}:")
            print(f"  Mean Absolute Error (MAE): {metrics['mean_absolute_error']:.6f}")
            print(f"  Max Absolute Error:        {metrics['max_absolute_error']:.6f}")
            print(f"  Mean Squared Error (MSE):  {metrics['mean_squared_error']:.6f}")
            print(f"  Root MSE (RMSE):           {metrics['rmse']:.6f}")
            
            # Print detailed comparison for first few timesteps
            print_action_comparison(pred_actions, gt_actions, num_steps=3)
        
        except StopIteration:
            print(f"Reached end of dataset after {i} samples")
            break
    
    # Compute average metrics across all samples
    print(f"\n{'='*80}")
    print("AVERAGE METRICS ACROSS ALL SAMPLES")
    print(f"{'='*80}")
    
    avg_mae = np.mean([m['mean_absolute_error'] for m in all_metrics])
    avg_max = np.mean([m['max_absolute_error'] for m in all_metrics])
    avg_mse = np.mean([m['mean_squared_error'] for m in all_metrics])
    avg_rmse = np.mean([m['rmse'] for m in all_metrics])
    
    print(f"Average MAE:  {avg_mae:.6f}")
    print(f"Average Max:  {avg_max:.6f}")
    print(f"Average MSE:  {avg_mse:.6f}")
    print(f"Average RMSE: {avg_rmse:.6f}")
    
    # Plot MAE vs timestep (averaged across samples)
    print("\nAverage MAE per timestep:")
    mae_per_timestep = np.mean([m['mae_per_timestep'] for m in all_metrics], axis=0)
    for t, mae in enumerate(mae_per_timestep[:10]):  # Show first 10 timesteps
        print(f"  t={t}: {mae:.6f}")
    
    # Clean up
    del policy
    
    print(f"\n{'='*80}")
    print("Testing complete!")
    print(f"{'='*80}")


def main():
    """Main function to run the test."""
    # Example usage - adjust these parameters as needed
    test_policy_on_dataset(
        config_name="dex_pi05",
        checkpoint_dir="checkpoints/dex_pi05/dex_test_pi05/30000",
        num_samples=5,  # Test on 5 samples
    )


if __name__ == "__main__":
    import os
    from multiprocessing import cpu_count 
    cpu_num = cpu_count()
    cpu_use = 4
    cur_pid = os.getpid()
    os.sched_setaffinity(cur_pid, list(range(cpu_num))[:cpu_use])
    print(f"set the max number of cpu used to {cpu_use}")

    main()
