"""
Isaac Sim integration for pi0.5 Dex policy.

This script demonstrates how to run the pi0.5 policy in Isaac Sim
for dexterous manipulation tasks.
"""

import dataclasses
import logging
import pathlib
import time
from typing import Optional

import numpy as np
import tyro

# OpenPI imports
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

# Note: You will need to install Isaac Sim and import the appropriate modules
# This is a template showing the integration pattern
# Uncomment and modify these imports based on your Isaac Sim setup:
# from isaacsim import SimulationApp
# from omni.isaac.core import World
# from omni.isaac.core.robots import Robot
# from omni.isaac.core.utils.types import ArticulationAction


@dataclasses.dataclass
class Args:
    """Arguments for running pi0.5 in Isaac Sim."""
    
    # Policy configuration
    config_name: str = "debug_dex"
    """Training config name (e.g., 'debug_dex', 'pi05_dex')"""
    
    checkpoint_dir: str = "/home/sifei/openpi/checkpoints/debug_dex/dex_test_pi05/9"
    """Path to the checkpoint directory"""
    
    # Isaac Sim parameters
    render: bool = True
    """Whether to render the simulation"""
    
    headless: bool = False
    """Run in headless mode (no GUI)"""
    
    # Robot parameters
    robot_dof: int = 26
    """Number of robot degrees of freedom"""
    
    # Camera parameters
    use_side_camera: bool = False
    """Whether to use side camera in addition to ego camera"""
    
    ego_camera_name: str = "ego_camera"
    """Name of the ego (main) camera in Isaac Sim"""
    
    side_camera_name: str = "side_camera"
    """Name of the side camera in Isaac Sim (if use_side_camera is True)"""
    
    image_width: int = 224
    """Image width for policy input"""
    
    image_height: int = 224
    """Image height for policy input"""
    
    # Execution parameters
    num_episodes: int = 10
    """Number of episodes to run"""
    
    max_steps_per_episode: int = 1000
    """Maximum steps per episode"""
    
    action_horizon: int = 50
    """Action horizon (number of future actions predicted)"""
    
    replan_frequency: int = 10
    """How often to replan (in steps)"""
    
    # Task parameters
    task_prompt: str = "grasp the object"
    """Task description/prompt for the policy"""
    
    # Output
    save_video: bool = False
    """Whether to save video of episodes"""
    
    video_output_dir: str = "data/isaac_sim/videos"
    """Directory to save videos"""


class IsaacSimEnvironment:
    """Wrapper for Isaac Sim environment to work with pi0.5 policy."""
    
    def __init__(self, args: Args):
        self.args = args
        
        # TODO: Initialize Isaac Sim
        # This is a placeholder - you'll need to implement this based on your Isaac Sim setup
        logging.info("Initializing Isaac Sim environment...")
        
        # Example initialization (modify based on your setup):
        # self.simulation_app = SimulationApp({
        #     "headless": args.headless,
        #     "width": 1920,
        #     "height": 1080,
        # })
        # self.world = World()
        # self.robot = self._setup_robot()
        # self.cameras = self._setup_cameras()
        
        self.current_step = 0
        logging.info("Isaac Sim environment initialized")
    
    def _setup_robot(self):
        """Setup the robot in Isaac Sim."""
        # TODO: Implement robot setup
        # Example:
        # robot = Robot(prim_path="/World/Robot", name="dex_robot")
        # return robot
        pass
    
    def _setup_cameras(self):
        """Setup cameras in Isaac Sim."""
        # TODO: Implement camera setup
        # Example:
        # cameras = {
        #     "ego": Camera(prim_path="/World/ego_camera"),
        # }
        # if self.args.use_side_camera:
        #     cameras["side"] = Camera(prim_path="/World/side_camera")
        # return cameras
        pass
    
    def reset(self) -> dict:
        """Reset the environment and return initial observation."""
        logging.info("Resetting environment...")
        
        # TODO: Implement environment reset
        # Example:
        # self.world.reset()
        # self.robot.initialize()
        
        self.current_step = 0
        return self.get_observation()
    
    def get_observation(self) -> dict:
        """
        Get current observation in the format expected by pi0.5 dex policy.
        
        Returns:
            dict with keys:
                - state: [robot_dof] float32 array
                - image: dict with 'ego_rgb' and optionally 'side_rgb'
                - prompt: str
        """
        # TODO: Implement observation collection from Isaac Sim
        # This is a placeholder showing the expected format
        
        # Get robot state (joint positions, velocities, etc.)
        # Example:
        # robot_state = self.robot.get_joint_positions()
        robot_state = np.zeros(self.args.robot_dof, dtype=np.float32)
        
        # Get camera images
        # Example:
        # ego_image = self.cameras["ego"].get_rgba()[:, :, :3]  # Get RGB only
        ego_image = np.random.randint(0, 256, 
                                      (self.args.image_height, self.args.image_width, 3),
                                      dtype=np.uint8)
        
        observation = {
            "state": robot_state,
            "image": {
                "ego_rgb": ego_image,
            },
            "prompt": self.args.task_prompt,
        }
        
        # Add side camera if enabled
        if self.args.use_side_camera:
            # Example:
            # side_image = self.cameras["side"].get_rgba()[:, :, :3]
            side_image = np.random.randint(0, 256,
                                          (self.args.image_height, self.args.image_width, 3),
                                          dtype=np.uint8)
            observation["image"]["side_rgb"] = side_image
        
        return observation
    
    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """
        Execute action in the environment.
        
        Args:
            action: [robot_dof] array of joint commands
            
        Returns:
            observation, reward, done, info
        """
        # TODO: Implement action execution in Isaac Sim
        # Example:
        # action_obj = ArticulationAction(joint_positions=action)
        # self.robot.apply_action(action_obj)
        # self.world.step(render=self.args.render)
        
        self.current_step += 1
        
        # Get new observation
        observation = self.get_observation()
        
        # TODO: Compute reward and check if episode is done
        reward = 0.0
        done = self.current_step >= self.args.max_steps_per_episode
        info = {"step": self.current_step}
        
        return observation, reward, done, info
    
    def close(self):
        """Close the environment and cleanup."""
        logging.info("Closing environment...")
        # TODO: Cleanup Isaac Sim resources
        # Example:
        # self.simulation_app.close()
        pass


class PolicyExecutor:
    """Handles policy execution with action chunking and replanning."""
    
    def __init__(self, policy, args: Args):
        self.policy = policy
        self.args = args
        self.action_queue = []
        self.steps_since_replan = 0
    
    def get_action(self, observation: dict) -> np.ndarray:
        """
        Get next action from policy, replanning if necessary.
        
        Args:
            observation: Current observation dict
            
        Returns:
            Action array [robot_dof]
        """
        # Check if we need to replan
        if (len(self.action_queue) == 0 or 
            self.steps_since_replan >= self.args.replan_frequency):
            self._replan(observation)
        
        # Get next action from queue
        action = self.action_queue.pop(0)
        self.steps_since_replan += 1
        
        return action
    
    def _replan(self, observation: dict):
        """Run policy inference and update action queue."""
        logging.info("Replanning actions...")
        
        # Run policy inference
        start_time = time.time()
        result = self.policy.infer(observation)
        inference_time = time.time() - start_time
        
        # Extract actions
        actions = result["actions"]  # Shape: [action_horizon, robot_dof]
        
        logging.info(f"Policy inference took {inference_time*1000:.1f}ms")
        logging.info(f"Predicted {len(actions)} actions")
        
        # Update action queue
        self.action_queue = list(actions)
        self.steps_since_replan = 0
    
    def reset(self):
        """Reset the policy executor state."""
        self.action_queue = []
        self.steps_since_replan = 0


def run_episode(env: IsaacSimEnvironment, 
                policy_executor: PolicyExecutor,
                episode_num: int) -> dict:
    """
    Run a single episode.
    
    Args:
        env: Isaac Sim environment
        policy_executor: Policy executor
        episode_num: Episode number
        
    Returns:
        Dictionary with episode statistics
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"Starting Episode {episode_num + 1}")
    logging.info(f"{'='*60}")
    
    # Reset environment and policy
    observation = env.reset()
    policy_executor.reset()
    
    # Episode tracking
    episode_reward = 0.0
    episode_steps = 0
    done = False
    
    while not done:
        # Get action from policy
        action = policy_executor.get_action(observation)
        
        # Execute action
        observation, reward, done, info = env.step(action)
        
        episode_reward += reward
        episode_steps += 1
        
        if episode_steps % 100 == 0:
            logging.info(f"Step {episode_steps}, Reward: {episode_reward:.2f}")
    
    # Episode summary
    stats = {
        "episode": episode_num,
        "steps": episode_steps,
        "reward": episode_reward,
    }
    
    logging.info(f"\nEpisode {episode_num + 1} finished:")
    logging.info(f"  Steps: {episode_steps}")
    logging.info(f"  Total Reward: {episode_reward:.2f}")
    
    return stats


def main(args: Args) -> None:
    """Main execution function."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logging.info("="*60)
    logging.info("pi0.5 Dex Policy in Isaac Sim")
    logging.info("="*60)
    logging.info(f"Config: {args.config_name}")
    logging.info(f"Checkpoint: {args.checkpoint_dir}")
    logging.info(f"Task: {args.task_prompt}")
    logging.info("="*60)
    
    # Load policy
    logging.info("\nLoading policy...")
    config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(config, args.checkpoint_dir)
    logging.info("Policy loaded successfully!")
    
    # Initialize environment
    env = IsaacSimEnvironment(args)
    
    # Initialize policy executor
    policy_executor = PolicyExecutor(policy, args)
    
    # Create output directory if saving videos
    if args.save_video:
        pathlib.Path(args.video_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run episodes
    all_stats = []
    try:
        for episode_num in range(args.num_episodes):
            stats = run_episode(env, policy_executor, episode_num)
            all_stats.append(stats)
    
    except KeyboardInterrupt:
        logging.info("\nInterrupted by user")
    
    finally:
        # Cleanup
        env.close()
        del policy
        
        # Print summary statistics
        if all_stats:
            logging.info("\n" + "="*60)
            logging.info("Summary Statistics")
            logging.info("="*60)
            avg_steps = np.mean([s["steps"] for s in all_stats])
            avg_reward = np.mean([s["reward"] for s in all_stats])
            logging.info(f"Episodes completed: {len(all_stats)}")
            logging.info(f"Average steps: {avg_steps:.1f}")
            logging.info(f"Average reward: {avg_reward:.2f}")
            logging.info("="*60)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
