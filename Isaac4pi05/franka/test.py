"""
Test script for running pretrained pi0.5-LIBERO policy with Franka robot in Isaac Sim.

This script demonstrates how to:
1. Load the pretrained pi0.5-LIBERO model
2. Set up a Franka Panda robot in Isaac Sim
3. Run the policy in a pick-and-place task

The pi0.5-LIBERO model:
- Action dimension: 7 (6 DOF arm + 1 gripper)
- Action type: Delta actions in end-effector space
- Action horizon: 10 steps
- Inputs: base_rgb, wrist_rgb, state (8D), prompt
"""

import argparse
import dataclasses
import os
import sys
import time
from datetime import datetime
from typing import Optional

import numpy as np

# Add openpi to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# OpenPI imports
from openpi.policies import policy_config as _policy_config
from openpi.policies import libero_policy
from openpi.training import config as _config
from openpi.shared import download


@dataclasses.dataclass
class Args:
    """Arguments for running pi0.5-LIBERO with Franka in Isaac Sim."""
    
    # Policy configuration
    config_name: str = "pi05_libero"
    """Training config name for LIBERO"""
    
    checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_libero"
    """Path or URL to the checkpoint directory"""
    
    # Isaac Sim parameters
    headless: bool = False
    """Run in headless mode (no GUI)"""
    
    # Camera parameters
    camera_width: int = 224
    """Camera width for policy input"""
    
    camera_height: int = 224
    """Camera height for policy input"""
    
    # Execution parameters
    num_episodes: int = 5
    """Number of episodes to run"""
    
    max_steps_per_episode: int = 500
    """Maximum steps per episode"""
    
    task_prompt: str = "pick up the cube and place it at the target"
    """Task description/prompt for the policy"""
    
    # Output
    output_dir: str = "isaac_validation_output"
    """Directory to save videos and logs"""
    
    # Debug
    debug: bool = False
    """Debug mode - use random actions instead of policy"""


class FrankaIsaacSimEnvironment:
    """Wrapper for Isaac Sim environment with Franka robot for pi0.5-LIBERO policy."""
    
    def __init__(self, args: Args, simulation_app):
        self.args = args
        self.simulation_app = simulation_app
        
        # Import omni modules after SimulationApp is started
        from isaacsim.core.api import World
        from isaacsim.robot.manipulators.examples.franka import Franka
        from isaacsim.core.api.objects import DynamicCuboid
        from isaacsim.sensors.camera import Camera
        import isaacsim.core.utils.numpy.rotations as rot_utils
        
        # Create world
        self.world = World(stage_units_in_meters=1.0)
        
        # Add ground plane
        self.world.scene.add_default_ground_plane()
        
        # Add Franka robot
        print("Adding Franka robot...")
        self.franka = self.world.scene.add(
            Franka(
                prim_path="/World/Franka",
                name="franka",
                position=np.array([0.0, 0.0, 0.0]),
            )
        )
        
        # Add target cube for pick and place
        self.cube = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Cube",
                name="cube",
                position=np.array([0.4, 0.0, 0.05]),
                scale=np.array([0.025, 0.025, 0.025]),
                color=np.array([1.0, 0.0, 0.0]),
                mass=0.05
            )
        )
        
        # Add target position marker (visual only)
        self.target = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Target",
                name="target",
                position=np.array([0.0, 0.4, 0.05]),
                scale=np.array([0.025, 0.025, 0.025]),
                color=np.array([0.0, 1.0, 0.0]),
                mass=0.0  # Static marker
            )
        )
        
        # Add cameras
        # Base/third-person camera
        self.base_camera = self.world.scene.add(
            Camera(
                prim_path="/World/BaseCamera",
                name="base_camera",
                position=np.array([1.0, 0.0, 0.8]),
                resolution=(self.args.camera_width, self.args.camera_height),
            )
        )
        # Look at the robot
        self.base_camera.set_local_pose(
            translation=np.array([1.0, 0.0, 0.8]),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([0, 30, 180]), degrees=True
            )
        )
        
        # Wrist camera (mounted on end-effector)
        self.wrist_camera = self.world.scene.add(
            Camera(
                prim_path="/World/Franka/panda_hand/wrist_camera",
                name="wrist_camera",
                position=np.array([0.0, 0.0, 0.05]),
                resolution=(self.args.camera_width, self.args.camera_height),
            )
        )
        
        print("Scene setup complete!")
        
    def reset(self):
        """Reset the environment for a new episode."""
        print("Resetting environment...")
        
        # Reset world
        self.world.reset()
        
        # Randomize cube position slightly
        cube_pos = np.array([0.4, 0.0, 0.05]) + np.random.uniform(-0.1, 0.1, size=3)
        cube_pos[2] = 0.05  # Keep z fixed
        self.cube.set_world_pose(position=cube_pos)
        
        # Reset gripper to open position
        self.franka.gripper.set_joint_positions(
            self.franka.gripper.joint_opened_positions
        )
        
        # Step a few times to stabilize
        for _ in range(10):
            self.world.step(render=True)
        
        return self.get_observation()
    
    def get_observation(self):
        """Get observation in the format expected by pi0.5-LIBERO policy."""
        # Get images from cameras
        base_image = self.base_camera.get_rgba()[:, :, :3]  # Remove alpha channel
        wrist_image = self.wrist_camera.get_rgba()[:, :, :3]
        
        # Get robot state
        # For LIBERO, state is 8D: [ee_pos (3), ee_quat (4), gripper (1)]
        ee_pos, ee_quat = self.franka.end_effector.get_world_pose()
        gripper_pos = self.franka.gripper.get_joint_positions()
        gripper_state = np.mean(gripper_pos)  # Average gripper joint positions
        
        state = np.concatenate([
            ee_pos,
            ee_quat,
            [gripper_state]
        ])
        
        # Create observation dict matching LIBERO format
        observation = {
            "observation/image": base_image.astype(np.uint8),
            "observation/wrist_image": wrist_image.astype(np.uint8),
            "observation/state": state.astype(np.float32),
            "prompt": self.args.task_prompt,
        }
        
        return observation
    
    def step(self, action):
        """
        Execute action in the environment.
        
        Action format for LIBERO (7D):
        - [0:3]: End-effector position delta (dx, dy, dz)
        - [3:6]: End-effector orientation delta (roll, pitch, yaw)
        - [6]: Gripper action (-1 = close, +1 = open)
        """
        # Get current end-effector pose
        ee_pos, ee_quat = self.franka.end_effector.get_world_pose()
        
        # Apply position delta
        target_pos = ee_pos + action[:3]
        
        # For orientation, we could apply delta but for simplicity
        # we keep it fixed for now (more complex IK needed for full 6DOF control)
        target_quat = ee_quat
        
        # Set gripper target
        gripper_action = action[6]
        if gripper_action < 0:
            # Close gripper
            gripper_target = self.franka.gripper.joint_closed_positions
        else:
            # Open gripper
            gripper_target = self.franka.gripper.joint_opened_positions
        
        # Apply actions
        # Note: For proper control, you'd use an IK controller here
        # For this example, we use a simplified approach
        from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
        
        # This is simplified - in practice you'd need proper IK control
        # For now, just apply joint positions
        self.franka.gripper.set_joint_positions(gripper_target)
        
        # Step simulation
        self.world.step(render=True)
        
        # Get new observation
        obs = self.get_observation()
        
        # Compute reward (distance to target)
        cube_pos, _ = self.cube.get_world_pose()
        target_pos, _ = self.target.get_world_pose()
        distance = np.linalg.norm(cube_pos - target_pos)
        reward = -distance
        
        # Check if done
        done = distance < 0.05
        
        info = {
            "distance_to_target": distance,
            "success": done,
        }
        
        return obs, reward, done, info


def main():
    """Main function to run the test."""
    import tyro
    args = tyro.cli(Args)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Output directory: {run_dir}")
    
    # Initialize Isaac Sim
    from isaacsim.core.simulation_app import SimulationApp
    
    simulation_app = SimulationApp({"headless": args.headless})
    
    try:
        # Create environment
        env = FrankaIsaacSimEnvironment(args, simulation_app)
        
        # Initialize world
        env.world.reset()
        
        # Load policy
        if not args.debug:
            print(f"Loading pi0.5-LIBERO policy from {args.checkpoint_dir}...")
            config = _config.get_config(args.config_name)
            checkpoint_dir = download.maybe_download(args.checkpoint_dir)
            policy = _policy_config.create_trained_policy(
                config, 
                checkpoint_dir,
                default_prompt=args.task_prompt
            )
            print("Policy loaded successfully!")
            
            # Test with dummy example to check shapes
            print("\nTesting policy with dummy example...")
            dummy_example = libero_policy.make_libero_example()
            dummy_result = policy.infer(dummy_example)
            print(f"Actions shape: {dummy_result['actions'].shape}")
            print(f"Expected shape: (1, action_horizon=10, action_dim=7)")
        else:
            print("Debug mode: using random actions")
            policy = None
        
        # Run episodes
        print(f"\nRunning {args.num_episodes} episodes...")
        success_count = 0
        
        for episode in range(args.num_episodes):
            print(f"\n{'='*50}")
            print(f"Episode {episode + 1}/{args.num_episodes}")
            print(f"{'='*50}")
            
            obs = env.reset()
            episode_reward = 0
            
            for step in range(args.max_steps_per_episode):
                # Get action from policy
                if policy is not None:
                    # Run inference
                    result = policy.infer(obs)
                    # Get first action from the action sequence
                    action = result["actions"][0, 0, :]  # Shape: (7,)
                else:
                    # Random action for debugging
                    action = np.random.randn(7) * 0.01
                    action[6] = np.random.choice([-1, 1])  # Gripper
                
                # Execute action
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                # Print progress
                if step % 50 == 0:
                    print(f"  Step {step}: distance={info['distance_to_target']:.4f}")
                
                if done:
                    print(f"  Success! Reached target in {step} steps.")
                    success_count += 1
                    break
            
            print(f"Episode {episode + 1} finished: reward={episode_reward:.2f}")
        
        print(f"\n{'='*50}")
        print(f"Results: {success_count}/{args.num_episodes} episodes successful")
        print(f"Success rate: {100.0 * success_count / args.num_episodes:.1f}%")
        print(f"{'='*50}")
        
    finally:
        # Cleanup
        simulation_app.close()


if __name__ == "__main__":
    main()
