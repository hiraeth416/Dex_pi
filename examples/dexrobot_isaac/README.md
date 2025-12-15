# pi0.5 Dex Policy in Isaac Sim

This example demonstrates how to run the pi0.5 dexterous manipulation policy in NVIDIA Isaac Sim.

## Overview

The pi0.5 policy is trained for dexterous manipulation tasks using the DexCanvas dataset. This integration allows you to:

- Run the trained policy in Isaac Sim simulation
- Control dexterous robots with vision-based manipulation
- Execute action chunks with replanning
- Visualize policy behavior in real-time

## Prerequisites

### 1. Isaac Sim Installation

Install NVIDIA Isaac Sim following the [official guide](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/index.html).

Recommended version: Isaac Sim 2023.1 or later

### 2. Python Environment

The code is designed to work with the OpenPI environment:

```bash
# From the openpi root directory
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 3. Policy Checkpoint

You need a trained pi0.5 checkpoint. Options:

**Option A: Use your local checkpoint**
```bash
# Update the checkpoint path in main.py or pass via command line
CHECKPOINT_DIR="/home/sifei/openpi/checkpoints/debug_dex/dex_test_pi05/9"
```

**Option B: Download pre-trained checkpoint** (if available)
```bash
# Download from GCS bucket (update with actual path when available)
uv run python -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/pi05_dex')"
```

## Isaac Sim Setup

You need to implement the Isaac Sim-specific components in `main.py`. The code includes TODO markers for:

### 1. Scene Setup

Create your robot and environment in Isaac Sim:

```python
# In IsaacSimEnvironment._setup_robot()
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage

# Load your dexterous robot (e.g., Shadow Hand, Allegro Hand, etc.)
robot = Robot(
    prim_path="/World/DexRobot",
    name="dex_robot",
    usd_path="/path/to/your/robot.usd"
)
```

### 2. Camera Setup

Setup cameras to match the policy's expectations:

```python
# In IsaacSimEnvironment._setup_cameras()
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.sensor import Camera

# Ego (main) camera
ego_camera = Camera(
    prim_path="/World/ego_camera",
    resolution=(224, 224),
)

# Optional: Side camera for additional viewpoint
if self.args.use_side_camera:
    side_camera = Camera(
        prim_path="/World/side_camera",
        resolution=(224, 224),
    )
```

### 3. Observation Collection

Implement observation collection:

```python
# In IsaacSimEnvironment.get_observation()
def get_observation(self) -> dict:
    # Get robot state (26 DOF for DexCanvas format)
    joint_positions = self.robot.get_joint_positions()
    
    # Get camera images (RGB, uint8, HWC format)
    ego_image = self.ego_camera.get_rgba()[:, :, :3]  # Remove alpha
    
    observation = {
        "state": joint_positions.astype(np.float32),
        "image": {
            "ego_rgb": ego_image.astype(np.uint8),
        },
        "prompt": self.args.task_prompt,
    }
    
    if self.args.use_side_camera:
        side_image = self.side_camera.get_rgba()[:, :, :3]
        observation["image"]["side_rgb"] = side_image.astype(np.uint8)
    
    return observation
```

### 4. Action Execution

Implement action execution:

```python
# In IsaacSimEnvironment.step()
from omni.isaac.core.utils.types import ArticulationAction

def step(self, action: np.ndarray):
    # Apply joint position commands
    action_obj = ArticulationAction(joint_positions=action)
    self.robot.apply_action(action_obj)
    
    # Step simulation
    self.world.step(render=self.args.render)
    
    # Return new observation and status
    observation = self.get_observation()
    reward = self._compute_reward()  # Implement your reward function
    done = self._check_done()  # Implement termination condition
    info = {}
    
    return observation, reward, done, info
```

## Usage

### Basic Usage

Run with default parameters:

```bash
uv run examples/dexrobot_isaac/main.py
```

### Custom Configuration

Specify config and checkpoint:

```bash
uv run examples/dexrobot_isaac/main.py \
    --config-name debug_dex \
    --checkpoint-dir /path/to/checkpoint \
    --task-prompt "grasp the red cube"
```

### Headless Mode

Run without GUI (faster):

```bash
uv run examples/dexrobot_isaac/main.py --headless
```

### With Side Camera

Enable side camera for additional viewpoint:

```bash
uv run examples/dexrobot_isaac/main.py --use-side-camera
```

### Save Videos

Record episodes:

```bash
uv run examples/dexrobot_isaac/main.py \
    --save-video \
    --video-output-dir data/isaac_videos
```

### Custom Execution Parameters

```bash
uv run examples/dexrobot_isaac/main.py \
    --num-episodes 50 \
    --max-steps-per-episode 500 \
    --action-horizon 50 \
    --replan-frequency 10
```

## Command Line Arguments

### Policy Configuration
- `--config-name`: Training config name (default: `debug_dex`)
- `--checkpoint-dir`: Path to checkpoint directory
- `--robot-dof`: Number of robot DOF (default: 26)

### Isaac Sim Parameters
- `--render`: Enable rendering (default: True)
- `--headless`: Run without GUI (default: False)

### Camera Parameters
- `--use-side-camera`: Use side camera (default: False)
- `--ego-camera-name`: Ego camera name in scene (default: `ego_camera`)
- `--side-camera-name`: Side camera name (default: `side_camera`)
- `--image-width`: Image width (default: 224)
- `--image-height`: Image height (default: 224)

### Execution Parameters
- `--num-episodes`: Number of episodes (default: 10)
- `--max-steps-per-episode`: Max steps per episode (default: 1000)
- `--action-horizon`: Action prediction horizon (default: 50)
- `--replan-frequency`: Steps between replanning (default: 10)

### Task Parameters
- `--task-prompt`: Task description (default: `"grasp the object"`)

### Output
- `--save-video`: Save episode videos (default: False)
- `--video-output-dir`: Video output directory (default: `data/isaac_sim/videos`)

## Expected Data Format

The policy expects observations in the following format:

```python
observation = {
    "state": np.array([...]),  # Shape: [26] for robot DOF
    "image": {
        "ego_rgb": np.array([...]),  # Shape: [224, 224, 3], dtype: uint8
        "side_rgb": np.array([...])  # Optional: [224, 224, 3], dtype: uint8
    },
    "prompt": "task description"
}
```

The policy outputs actions in this format:

```python
output = {
    "actions": np.array([...])  # Shape: [action_horizon, 26]
}
```

## Action Execution Strategy

The code implements an action chunking strategy:

1. **Initial Planning**: Policy predicts `action_horizon` future actions (default: 50)
2. **Execution**: Actions are executed one at a time
3. **Replanning**: New actions are predicted every `replan_frequency` steps (default: 10)

This allows for:
- Smooth execution of planned action sequences
- Adaptive replanning based on environment feedback
- Balance between planning efficiency and reactivity

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'isaacsim'`
- **Solution**: Ensure Isaac Sim is installed and Python path is configured

**Issue**: Camera images are all black
- **Solution**: Check camera positioning and lighting in the scene

**Issue**: Robot doesn't move
- **Solution**: Verify action dimensions match robot DOF and joints are not locked

**Issue**: Policy inference is slow
- **Solution**: 
  - Ensure CUDA is available: `import torch; print(torch.cuda.is_available())`
  - Use headless mode for faster simulation
  - Increase `replan_frequency` to reduce inference calls

### Debugging Tips

1. **Test policy independently**:
```python
from openpi.policies import dex_policy, policy_config
from openpi.training import config

cfg = config.get_config("debug_dex")
policy = policy_config.create_trained_policy(cfg, "/path/to/checkpoint")

# Test with dummy data
example = dex_policy.make_dex_example()
result = policy.infer(example)
print("Actions shape:", result["actions"].shape)
```

2. **Verify observation format**:
```python
obs = env.get_observation()
print("State shape:", obs["state"].shape)
print("Ego image shape:", obs["image"]["ego_rgb"].shape)
print("Ego image dtype:", obs["image"]["ego_rgb"].dtype)
```

3. **Enable verbose logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Example: Full Isaac Sim Integration

Here's a complete example for a simple dexterous manipulation task:

```python
# examples/dexrobot_isaac/example_integration.py

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.sensor import Camera
import numpy as np

# Your policy setup
from openpi.policies import policy_config
from openpi.training import config

# Initialize world
world = World()

# Add robot
robot = Robot(prim_path="/World/Robot", name="dex_robot")

# Add cameras
ego_cam = Camera(prim_path="/World/ego_camera", resolution=(224, 224))

# Load policy
cfg = config.get_config("debug_dex")
policy = policy_config.create_trained_policy(cfg, "/path/to/checkpoint")

# Reset
world.reset()

# Run episode
for step in range(100):
    # Get observation
    obs = {
        "state": robot.get_joint_positions()[:26],
        "image": {
            "ego_rgb": ego_cam.get_rgba()[:, :, :3],
        },
        "prompt": "grasp the object"
    }
    
    # Get action
    result = policy.infer(obs)
    action = result["actions"][0]  # First action from chunk
    
    # Execute
    robot.apply_action(ArticulationAction(joint_positions=action))
    world.step(render=True)

simulation_app.close()
```

## Additional Resources

- [OpenPI Documentation](../../README.md)
- [Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/)
- [DexCanvas Dataset](../../dexcanvas/README.md)
- [Training Documentation](../../docs/)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{openpi2024,
  title={OpenPI: Open Physical Intelligence},
  author={Physical Intelligence Team},
  year={2024}
}
```
