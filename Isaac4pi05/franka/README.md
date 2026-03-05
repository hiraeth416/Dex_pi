# Testing π0.5-LIBERO with Franka in Isaac Sim

This directory contains a test script for running the pretrained π0.5-LIBERO policy with a Franka Panda robot in NVIDIA Isaac Sim.

## Model Information

### π0.5-LIBERO Policy
- **Model**: π0.5 (diffusion-based policy)
- **Training dataset**: LIBERO
- **Checkpoint**: `gs://openpi-assets/checkpoints/pi05_libero`
- **Action dimension**: 7 (6 DOF end-effector + 1 gripper)
- **Action type**: Delta actions in end-effector space
- **Action horizon**: 10 steps (action chunking)
- **State dimension**: 8 ([ee_pos (3), ee_quat (4), gripper (1)])

### Why π0.5-LIBERO?

This is the correct model for Franka robot because:

1. **Action Space Match**: LIBERO dataset uses 7-dimensional actions which match Franka's end-effector control (6 DOF pose + gripper)
2. **State Format**: LIBERO uses end-effector pose + gripper state, which is natural for Franka
3. **Task Domain**: Trained on tabletop manipulation tasks similar to what Franka typically performs
4. **Available Checkpoint**: Pre-trained checkpoint is available at `gs://openpi-assets/checkpoints/pi05_libero`

### Action Format (7D Vector)
```python
action = [
    dx,     # End-effector X position delta
    dy,     # End-effector Y position delta  
    dz,     # End-effector Z position delta
    droll,  # End-effector roll delta
    dpitch, # End-effector pitch delta
    dyaw,   # End-effector yaw delta
    gripper # Gripper action (-1=close, +1=open)
]
```

### Observation Format
```python
observation = {
    "observation/image": np.ndarray,        # (224, 224, 3) uint8 - base camera
    "observation/wrist_image": np.ndarray,  # (224, 224, 3) uint8 - wrist camera
    "observation/state": np.ndarray,        # (8,) float32 - [ee_pos, ee_quat, gripper]
    "prompt": str                            # Task description
}
```

## Prerequisites

1. **NVIDIA Isaac Sim 5.1.0+**: Ensure Isaac Sim is installed and environment is properly configured
2. **OpenPI**: Install the OpenPI package
3. **Python dependencies**: JAX, NumPy, etc. (from OpenPI requirements)

```bash
# Make sure Isaac Sim is sourced
source ~/.local/share/ov/pkg/isaac-sim-*/setup_python_env.sh

# Or if using conda environment
conda activate isaac-sim
```

## Installation

```bash
# Install OpenPI if not already installed
cd /home/sifei/openpi
pip install -e .

# Install additional dependencies if needed
pip install tyro imageio
```

## Usage

### Basic Usage

```bash
# Run with GUI (default will download checkpoint automatically)
python Isaac4pi05/franka/test.py

# Run in headless mode
python Isaac4pi05/franka/test.py --headless True

# Run with custom checkpoint
python Isaac4pi05/franka/test.py --checkpoint_dir /path/to/checkpoint

# Change task prompt
python Isaac4pi05/franka/test.py --task_prompt "grasp the red cube"

# Run multiple episodes
python Isaac4pi05/franka/test.py --num_episodes 10
```

### Debug Mode

Test without loading the policy (uses random actions):

```bash
python Isaac4pi05/franka/test.py --debug True
```

### Full Command Line Options

```bash
python Isaac4pi05/franka/test.py --help
```

Available options:
- `--config_name`: Training config name (default: "pi05_libero")
- `--checkpoint_dir`: Checkpoint path or GCS URL (default: "gs://openpi-assets/checkpoints/pi05_libero")
- `--headless`: Run without GUI (default: False)
- `--camera_width`: Camera width (default: 224)
- `--camera_height`: Camera height (default: 224)
- `--num_episodes`: Number of episodes (default: 5)
- `--max_steps_per_episode`: Max steps per episode (default: 500)
- `--task_prompt`: Task description (default: "pick up the cube and place it at the target")
- `--output_dir`: Output directory (default: "isaac_validation_output")
- `--debug`: Use random actions (default: False)

## Implementation Details

### Key Components

1. **Policy Loading**: Uses OpenPI's `create_trained_policy` to load the pre-trained π0.5-LIBERO model
2. **Environment Setup**: Creates Franka robot, cameras, and objects in Isaac Sim
3. **Observation Handling**: Converts Isaac Sim state to LIBERO observation format
4. **Action Execution**: Applies policy actions to Franka end-effector and gripper

### Camera Setup

- **Base Camera**: Third-person view at `(1.0, 0.0, 0.8)` looking at robot
- **Wrist Camera**: Mounted on end-effector for first-person view
- Both cameras output 224x224 RGB images

### Action Execution

The current implementation uses a simplified action execution:
- Position deltas are applied directly to end-effector
- Gripper opens/closes based on action sign
- **Note**: For production use, implement proper inverse kinematics (IK) controller

### Limitations

1. **Simplified Control**: Current implementation doesn't use full IK for orientation control
2. **Action Scaling**: May need to tune action scaling for sim-to-real transfer
3. **Camera Calibration**: Camera poses may need adjustment for optimal policy performance

## Expected Behavior

The policy should:
1. Observe the scene through base and wrist cameras
2. Generate delta actions to move end-effector toward cube
3. Close gripper when near cube
4. Move cube toward green target marker
5. Success when cube is within 5cm of target

## Troubleshooting

### Policy doesn't load
- Check internet connection (downloads from GCS)
- Verify checkpoint path exists
- Try debug mode first: `--debug True`

### Robot doesn't move
- Check action scaling (may be too small)
- Verify observations are correct shape
- Check Isaac Sim physics settings

### Camera images are black
- Ensure simulation is running (`world.step(render=True)`)
- Check camera is initialized after world reset
- Verify camera poses are correct

### Actions are erratic
- Policy may need fine-tuning for Isaac Sim
- Try reducing action magnitude
- Check state normalization

## Comparison with Other Models

| Model | Action Dim | Use Case |
|-------|-----------|----------|
| π0.5-LIBERO | 7 (6+1) | ✅ Franka - tabletop manipulation |
| π0.5-DEX | 26 | ❌ Dexterous hand, not robot arm |
| π0.5-DROID | 7 (6+1) | ✅ Could work, but LIBERO is better match |
| π0-LIBERO | 7 (6+1) | ✅ Similar but π0.5 has better performance |

## Next Steps

1. **Improve IK Control**: Implement proper inverse kinematics for full 6-DOF control
2. **Add More Tasks**: Test on different pick-and-place configurations
3. **Record Videos**: Save camera feeds for analysis
4. **Tune Parameters**: Adjust action scaling and control frequency
5. **Sim-to-Real**: Transfer policy to real Franka robot

## References

- [OpenPI GitHub](https://github.com/Physical-Intelligence/openpi)
- [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/)
- [LIBERO Benchmark](https://libero-project.github.io/)
- [π0 Paper](https://www.physicalintelligence.company/blog/pi0)

## License

This code follows the OpenPI license. See the main repository for details.
