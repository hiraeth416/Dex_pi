# Isaac Sim Validation for pi0.5 Policy

This directory contains scripts to validate the trained pi0.5 policy in NVIDIA Isaac Sim.

## Prerequisites

1.  **NVIDIA Isaac Sim**: Ensure you have Isaac Sim installed and the environment variables sourced.
2.  **OpenPI**: The `openpi` package must be installed or in your PYTHONPATH.
3.  **Checkpoints**: You need a trained checkpoint (e.g., `dex_pi05`).
4.  **Assets**: The MANO hand URDF must be available at `mano_assets/operators/s02/mano_hand.urdf`.

## Usage

To run the validation script, use the following command:

```bash
# Run in headless mode (default)
python Isaac4pi05/validate_policy.py

# Run with GUI (if available)
python Isaac4pi05/validate_policy.py --headless False

# Specify a different checkpoint
python Isaac4pi05/validate_policy.py --checkpoint_dir /path/to/checkpoint

# Specify a different config
python Isaac4pi05/validate_policy.py --config_name dex_pi05
```

## Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--config_name` | Training config name | `dex_pi05` |
| `--checkpoint_dir` | Path to the checkpoint directory | `.../dex_test_pi05/9999` |
| `--headless` | Run in headless mode (no GUI) | `True` |
| `--robot_urdf_path` | Path to the MANO hand URDF | `.../mano_hand.urdf` |
| `--num_episodes` | Number of episodes to run | `1` |
| `--max_steps_per_episode` | Maximum steps per episode | `200` |
| `--output_dir` | Directory to save videos | `isaac_validation_output` |

## Output

The script will generate video files (e.g., `episode_0.mp4`) in the `output_dir`. These videos show the side-by-side view of the Ego and Side cameras during the simulation.
