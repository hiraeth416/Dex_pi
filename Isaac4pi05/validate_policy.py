"""
Isaac Sim validation script for pi0.5 Dex policy.
"""

import argparse
import dataclasses
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import jax
import logging
import warnings
import tqdm

# Suppress warnings and set logging level
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)
os.environ['PYTHONWARNINGS'] = 'ignore'

import cv2
import numpy as np
import tyro

# OpenPI imports
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.policies import dex_policy

import isaacsim  # Ensure Isaac Sim is sourced in the environment

# DOF converter for DexCanvas <-> IsaacSim joint order conversion
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
from dof_converter import dexcanvas_to_isaacsim, isaacsim_to_dexcanvas


# Setup error logging
def setup_error_logger(output_dir: str):
    """Setup a logger that writes errors to a file."""
    log_file = os.path.join(output_dir, f"error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.ERROR)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Create logger
    error_logger = logging.getLogger('isaac_validation')
    error_logger.setLevel(logging.DEBUG)
    error_logger.addHandler(file_handler)
    
    # Also add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    error_logger.addHandler(console_handler)
    
    return error_logger, log_file

# Isaac Sim imports will be handled inside the class or main to allow argument parsing first
# because SimulationApp needs to be initialized before other omni imports.

@dataclasses.dataclass
class Args:
    """Arguments for running pi0.5 in Isaac Sim."""
    
    # Policy configuration
    config_name: str = "dex_pi05"
    """Training config name (e.g., 'debug_dex', 'dex_pi05')"""
    
    checkpoint_dir: str = "checkpoints/dex_pi05/dex_test_pi05/9999"
    """Path to the checkpoint directory"""
    
    # Isaac Sim parameters
    headless: bool = True
    """Run in headless mode (no GUI)"""
    
    # Robot parameters
    robot_urdf_path: str = "mano_assets/operators/s02/mano_hand.urdf"
    """Path to the MANO hand URDF"""
    
    robot_dof: int = 26
    """Number of robot degrees of freedom"""
    
    # Camera parameters
    camera_width: int = 1280
    """Camera native width for rendering"""
    
    camera_height: int = 720
    """Camera native height for rendering"""
    
    image_width: int = 224
    """Image width for policy input (after resizing)"""
    
    image_height: int = 224
    """Image height for policy input (after resizing)"""
    
    use_side_camera: bool = False
    """Whether to use side camera (set to False if policy was trained without side camera)"""
    
    # Execution parameters
    num_episodes: int = 1
    """Number of episodes to run"""
    
    max_steps_per_episode: int = 200
    """Maximum steps per episode"""
    
    task_prompt: str = "grasp the object"
    """Task description/prompt for the policy"""
    
    # Output
    output_dir: str = "isaac_validation_output"
    """Directory to save videos"""
    
    # Debug
    debug: bool = False
    """Debug mode - use random actions instead of loading policy"""


class IsaacSimEnvironment:
    """Wrapper for Isaac Sim environment to work with pi0.5 policy."""
    
    def __init__(self, args: Args, simulation_app):
        self.args = args
        self.simulation_app = simulation_app
        
        # Import omni modules here after SimulationApp is started
        from isaacsim.core.api import World
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.api.objects import DynamicCuboid
        from isaacsim.sensors.camera import Camera
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.prims import get_prim_at_path
        import isaacsim.core.utils.numpy.rotations as rot_utils
        
        self.world = World(stage_units_in_meters=1.0)
        
        # Set physics parameters for stable simulation
        # Get physics context and set parameters
        from isaacsim.core.api.physics_context import PhysicsContext
        physics_ctx = self.world.get_physics_context()
        
        # Set simulation timestep (50 Hz = 0.02s per step)
        physics_ctx.set_physics_dt(1.0 / 50.0)
        
        # Set gravity (default is -9.81 in z direction)
        physics_ctx.set_gravity(-9.81)
        
        # Set solver parameters for better stability
        physics_ctx.set_solver_type("TGS")  # Temporal Gauss-Seidel solver
        
        print(f"Physics context configured:")
        print(f"  Timestep: {physics_ctx.get_physics_dt():.4f}s ({1.0/physics_ctx.get_physics_dt():.1f} Hz)")
        print(f"  Gravity: {physics_ctx.get_gravity()}")
        print(f"  Solver: {physics_ctx.get_solver_type()}")
        
        # Add Ground Plane
        self.world.scene.add_default_ground_plane()
        
        # Add MANO Hand
        # We need to make sure the path exists
        if not os.path.exists(self.args.robot_urdf_path):
            raise FileNotFoundError(f"URDF not found at {self.args.robot_urdf_path}")

        #self.robot_prim_path = "/World/mano_hand"
        
        # Import URDF using Isaac Sim's URDF importer
        import omni.kit.commands
        from isaacsim.core.utils.extensions import get_extension_path_from_name
        from isaacsim.asset.importer.urdf import _urdf
        
        # Create import configuration
        import_config = _urdf.ImportConfig()
        import_config.merge_fixed_joints = False
        import_config.fix_base = True
        import_config.import_inertia_tensor = True
        import_config.self_collision = True

        
        # Create a Urdf interface and parse the URDF
        urdf_interface = _urdf.acquire_urdf_interface()
        
        
        # Import the parsed robot into the stage at the desired prim path
        status, robot_model = omni.kit.commands.execute(
            "URDFParseFile",
            urdf_path=self.args.robot_urdf_path,
            import_config=import_config
        )
        result, prim_path = omni.kit.commands.execute(
            "URDFImportRobot",
            urdf_robot=robot_model,
            import_config=import_config,
        )
        
        self.robot = self.world.scene.add(
            Robot(
                prim_path=prim_path,
                name="mano_hand",
                position=np.array([0.2, 0.0, 0.5]), # Lift it up a bit
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        
        # Set contact properties for the robot (for better grasping)
        from pxr import UsdPhysics, PhysxSchema
        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath(prim_path)
        
        # Enable contact reporting and set friction for all child prims
        for prim in robot_prim.GetAllChildren():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                # Add physics material with higher friction
                physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                physx_collision_api.CreateContactOffsetAttr(0.02)
                physx_collision_api.CreateRestOffsetAttr(0.001)
        
        print(f"Robot collision properties configured for better grasping")
        
        # Add Target Object (Cube)
        self.cube = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Cube",
                name="cube",
                position=np.array([0.0, 0.0, 0.1]), # In front of the hand
                scale=np.array([0.05, 0.05, 0.05]),
                color=np.array([1.0, 0.0, 0.0]),  # Bright red
                mass=0.1  # Light mass for easier manipulation
            )
        )

        # Setup Cameras
        # Hand is at (0.4, 0.0, 0.5), cube is at (0.4, 0.0, 0.1)
        # Camera should look at these positions
        ego_position = np.array([0.0, 0.0, 5.0])  # Above and in front, looking at x=0.4
        ego_target = np.array([0.4, 0.0, 0.3])    # Look at midpoint between hand and cube
        self.ego_camera = self._create_camera("/World/Camera_Ego", ego_position, ego_target)
        
        if self.args.use_side_camera:
            # Side camera: from the side
            side_position = np.array([0.4, -0.6, 0.5])
            side_target = np.array([0.4, 0.0, 0.3])
            self.side_camera = self._create_camera("/World/Camera_Side", side_position, side_target)
            print("Side camera enabled")
        else:
            self.side_camera = None
            print("Side camera disabled")
        
        
        self.world.reset()
        
        # Initialize cameras after world reset
        self.ego_camera.initialize()
        if self.side_camera:
            self.side_camera.initialize()
            
        # Ensure rendering is set up
        self.world.step(render=True)
        
        # Test camera capture
        test_frame = self.ego_camera.get_rgba()
        if test_frame is not None:
            print(f"Camera test successful - captured frame shape: {test_frame.shape}, dtype: {test_frame.dtype}")
        else:
            print("WARNING: Camera test failed - no frame captured!")

        print("="*40)
        print("Environment initialized.")
        print("="*40)
        

    def _create_camera(self, prim_path, position, look_at):
        from isaacsim.sensors.camera import Camera
        import isaacsim.core.utils.numpy.rotations as rot_utils
        
        # Use the look_at_to_quaternion function to properly orient camera
        orientation = [0.707, 0.0, 0.0, 0.0]  # [0.0, 0.331, 0.0, 0.944]
        camera = Camera(
            prim_path=prim_path,
            position=position,
            frequency=20,
            resolution=(self.args.camera_width, self.args.camera_height),
            orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 180]), degrees=True),
            #orientation=orientation,
        )
        camera.initialize()
        
        # Add render product annotations for better rendering
        camera.add_distance_to_image_plane_to_frame()
        
        return camera

    def get_observation(self) -> dict:
        """Get observation in the format expected by the policy."""
        
        # Step the world once to ensure camera renders
        self.world.step(render=True)
        
        # Get Joint State
        # Note: We need to ensure the order matches the URDF and policy expectation.
        # The policy expects 26 DOF in DexCanvas format.
        joint_positions_isaac = self.robot.get_joint_positions()
        
        # If joint count doesn't match, we might need to pad or slice
        if len(joint_positions_isaac) != self.args.robot_dof:
            # print(f"Warning: Robot has {len(joint_positions_isaac)} joints, policy expects {self.args.robot_dof}")
            if len(joint_positions_isaac) > self.args.robot_dof:
                joint_positions_isaac = joint_positions_isaac[:self.args.robot_dof]
            else:
                joint_positions_isaac = np.pad(joint_positions_isaac, (0, self.args.robot_dof - len(joint_positions_isaac)))
        
        # Convert from IsaacSim joint order to DexCanvas format (policy expects DexCanvas format)
        joint_positions = isaacsim_to_dexcanvas(joint_positions_isaac)

        # Get Images - need to initialize/update cameras first
        self.ego_camera.initialize()
        
        # Get RGBA data for ego camera
        ego_rgba = self.ego_camera.get_rgba()
        
        # Check if camera data is valid and create fallback
        if ego_rgba is None or not isinstance(ego_rgba, np.ndarray) or ego_rgba.size == 0:
            print(f"Warning: ego_camera returned invalid data (None:{ego_rgba is None} or empty:{ego_rgba is not None and ego_rgba.size == 0})")
            ego_rgb = np.zeros((self.args.image_height, self.args.image_width, 3), dtype=np.uint8)
        elif len(ego_rgba.shape) != 3:
            print(f"Warning: ego_camera returned invalid shape: {ego_rgba.shape}")
            ego_rgb = np.zeros((self.args.image_height, self.args.image_width, 3), dtype=np.uint8)
        else:
            ego_rgb = ego_rgba[:, :, :3]  # Drop Alpha
            # Ensure uint8
            if ego_rgb.dtype != np.uint8:
                ego_rgb = (ego_rgb * 255).astype(np.uint8)
            
            # Resize from camera resolution to policy input size
            if ego_rgb.shape[0] != self.args.image_height or ego_rgb.shape[1] != self.args.image_width:
                ego_rgb = cv2.resize(ego_rgb, (self.args.image_width, self.args.image_height), interpolation=cv2.INTER_LINEAR)
        
        # Build image dict
        image_dict = {"ego_rgb": ego_rgb}
        
        # Only get side camera if enabled
        if self.args.use_side_camera and self.side_camera is not None:
            self.side_camera.initialize()
            side_rgba = self.side_camera.get_rgba()
            
            if side_rgba is None or not isinstance(side_rgba, np.ndarray) or side_rgba.size == 0:
                print(f"Warning: side_camera returned invalid data (None:{side_rgba is None} or empty:{side_rgba is not None and side_rgba.size == 0})")
                side_rgb = np.zeros((self.args.image_height, self.args.image_width, 3), dtype=np.uint8)
            elif len(side_rgba.shape) != 3:
                print(f"Warning: side_camera returned invalid shape: {side_rgba.shape}")
                side_rgb = np.zeros((self.args.image_height, self.args.image_width, 3), dtype=np.uint8)
            else:
                side_rgb = side_rgba[:, :, :3]
                # Ensure uint8
                if side_rgb.dtype != np.uint8:
                    side_rgb = (side_rgb * 255).astype(np.uint8)
                
                # Resize from camera resolution to policy input size
                if side_rgb.shape[0] != self.args.image_height or side_rgb.shape[1] != self.args.image_width:
                    side_rgb = cv2.resize(side_rgb, (self.args.image_width, self.args.image_height), interpolation=cv2.INTER_LINEAR)
                
            image_dict["side_rgb"] = side_rgb
        return {
            "state": joint_positions.astype(np.float32),
            "image": image_dict,
            "prompt": self.args.task_prompt,
        }

    def step(self, action: np.ndarray):
        """Apply action to the robot.
        
        Args:
            action: Joint positions in DexCanvas format (from policy output)
        """
        # Action is expected to be joint positions (26 DOF) in DexCanvas format
        # Need to convert to IsaacSim joint order before applying
        
        # Ensure action is the right size
        if len(action) != self.args.robot_dof:
             print(f"Warning: Action has {len(action)} DOF, expected {self.args.robot_dof}")
             pass
        
        # Convert from DexCanvas format (policy output) to IsaacSim joint order
        action_isaac = dexcanvas_to_isaacsim(action)
             
        from isaacsim.core.utils.types import ArticulationAction
        
        # Create ArticulationAction with converted joint order
        # Assuming position control
        articulation_action = ArticulationAction(joint_positions=action_isaac)
        
        self.robot.apply_action(articulation_action)
        self.world.step(render=True)
    
    def check_grasp_success(self, height_threshold: float = 0.60) -> bool:
        """Check if the object has been successfully grasped.
        
        Args:
            height_threshold: Minimum z-coordinate for successful grasp (in meters)
            
        Returns:
            True if object is lifted above threshold, False otherwise
        """
        # Get current position of the cube
        cube_position, _ = self.cube.get_world_pose()
        cube_z = cube_position[2]
        
        # Check if cube has been lifted above the threshold
        # Initial cube z position is 0.05, so lifting above 0.15 indicates success
        is_grasped = cube_z > height_threshold
        
        return is_grasped
    
    def get_object_height(self) -> float:
        """Get current height (z-coordinate) of the object.
        
        Returns:
            Current z-coordinate of the cube
        """
        cube_position, _ = self.cube.get_world_pose()
        return cube_position[2]

    def close(self):
        self.simulation_app.close()


def main(args: Args):
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Create subdirectories for images
    images_dir = os.path.join(run_output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    logger, log_file = setup_error_logger(run_output_dir)
    logger.info(f"Run output directory: {run_output_dir}")
    logger.info(f"Error log file: {log_file}")
    logger.info(f"Starting validation with config: {args.config_name}")
    
    env = None
    policy = None
    simulation_app = None
    
    try:
        # 1. Initialize Isaac Sim (must be first)
        logger.info("Step 1: Initializing Isaac Sim...")
        try:
            from isaacsim import SimulationApp
        except ImportError as e:
            logger.error(f"Failed to import SimulationApp: {e}")
            print("✗ Could not import SimulationApp. Please ensure Isaac Sim is installed and sourced.")
            return

        try:
            simulation_app = SimulationApp({"headless": args.headless})
            logger.info("✓ SimulationApp initialized")
        except Exception as e:
            logger.error(f"Failed to create SimulationApp: {e}\n{traceback.format_exc()}")
            print(f"✗ Failed to create SimulationApp: {e}")
            return
        
        # Suppress Isaac Sim logging after initialization
        try:
            import carb
            carb.settings.get_settings().set("/log/level", 4)
            carb.settings.get_settings().set("/log/fileLogLevel", 4)
        except Exception as e:
            logger.warning(f"Could not set Isaac Sim log level: {e}")

        # 2. Initialize Environment
        logger.info("Step 2: Initializing Environment...")
        print("\n========== Initializing Environment ==========")
        try:
            env = IsaacSimEnvironment(args, simulation_app)
            logger.info("✓ Environment initialized successfully")
            print("✓ Environment initialized successfully\n")
        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}\n{traceback.format_exc()}")
            print(f"✗ Failed to initialize environment: {e}")
            print(f"Check error log: {log_file}")
            if simulation_app:
                simulation_app.close()
            return
        
        # 3. Load Policy (skip in debug mode)
        if args.debug:
            logger.info("Step 3: Debug mode - skipping policy loading")
            print("Debug mode enabled - using random actions")
            policy = None
        else:
            logger.info("Step 3: Loading policy...")
            print(f"Loading policy from {args.checkpoint_dir}...")
            try:
                config = _config.get_config(args.config_name)
                policy = _policy_config.create_trained_policy(config, args.checkpoint_dir)
                logger.info("✓ Policy loaded successfully")
                print("✓ Policy loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load policy: {e}\n{traceback.format_exc()}")
                print(f"✗ Failed to load policy: {e}")
                print(f"Check error log: {log_file}")
                if env:
                    env.close()
                return

        print("="*40)
        print("Starting Inference Loop...")
        print("="*40)

        # 5. Run Inference Loop
        for episode_idx in tqdm.tqdm(range(args.num_episodes), desc="Episodes"):
            logger.info(f"Starting Episode {episode_idx}/{args.num_episodes}")
            print(f"\nStarting Episode {episode_idx}")
            
            # Reset environment
            try:
                env.world.reset()
                env.ego_camera.initialize()
                if args.use_side_camera and env.side_camera:
                    env.side_camera.initialize()
                logger.info("Environment reset successful")
            except Exception as e:
                logger.error(f"Failed to reset environment at episode {episode_idx}: {e}\n{traceback.format_exc()}")
                print(f"✗ Failed to reset environment: {e}")
                continue
            
            video_frames = []
            episode_errors = 0
            grasp_success = False
            max_cube_height = 0.0
            initial_cube_height = env.get_object_height()
            
            print(f"Initial cube height: {initial_cube_height:.4f}m")
            
            for step_idx in tqdm.tqdm(range(args.max_steps_per_episode), desc="Steps", leave=False):
                # Get Observation
                try:
                    obs = env.get_observation()
                    #import ipdb; ipdb.set_trace()
                    # Save camera images
                    try:
                        ego_img = obs["image"]["ego_rgb"]
                        ego_img_path = os.path.join(images_dir, f"episode_{episode_idx:03d}_step_{step_idx:04d}_ego.png")
                        cv2.imwrite(ego_img_path, cv2.cvtColor(ego_img, cv2.COLOR_RGB2BGR))
                        
                        if args.use_side_camera and "side_rgb" in obs["image"]:
                            side_img = obs["image"]["side_rgb"]
                            side_img_path = os.path.join(images_dir, f"episode_{episode_idx:03d}_step_{step_idx:04d}_side.png")
                            cv2.imwrite(side_img_path, cv2.cvtColor(side_img, cv2.COLOR_RGB2BGR))
                    except Exception as e:
                        logger.warning(f"Failed to save images at episode {episode_idx}, step {step_idx}: {e}")
                        
                except Exception as e:
                    logger.error(f"Failed to get observation at episode {episode_idx}, step {step_idx}: {e}\n{traceback.format_exc()}")
                    print(f"✗ Failed to get observation at step {step_idx}: {e}")
                    episode_errors += 1
                    if episode_errors >= 5:
                        logger.error(f"Too many errors in episode {episode_idx}, skipping...")
                        break
                    continue
                
                # Run Policy or generate random action
                if args.debug:
                    # Debug mode: use random actions
                    try:
                        # Generate random joint positions in valid range (-1, 1)
                        action = np.random.uniform(-0.5, 0.5, size=args.robot_dof).astype(np.float32)
                    except Exception as e:
                        logger.error(f"Failed to generate random action at episode {episode_idx}, step {step_idx}: {e}\n{traceback.format_exc()}")
                        print(f"✗ Failed to generate random action at step {step_idx}: {e}")
                        episode_errors += 1
                        if episode_errors >= 5:
                            logger.error(f"Too many errors in episode {episode_idx}, skipping...")
                            break
                        continue
                else:
                    # Normal mode: run policy
                    try:
                        result = policy.infer(obs)
                    except Exception as e:
                        logger.error(f"Policy inference failed at episode {episode_idx}, step {step_idx}: {e}\n{traceback.format_exc()}")
                        print(f"✗ Policy inference failed at step {step_idx}: {e}")
                        episode_errors += 1
                        if episode_errors >= 5:
                            logger.error(f"Too many errors in episode {episode_idx}, skipping...")
                            break
                        continue
                    
                    # Get Action
                    try:
                        action = result["actions"]
                        if len(action.shape) == 2:
                            action = action[0]
                    except Exception as e:
                        logger.error(f"Failed to extract action at episode {episode_idx}, step {step_idx}: {e}\n{traceback.format_exc()}")
                        print(f"✗ Failed to extract action: {e}")
                        episode_errors += 1
                        if episode_errors >= 5:
                            logger.error(f"Too many errors in episode {episode_idx}, skipping...")
                            break
                        continue
                
                # Apply Action
                try:
                    env.step(action)
                    
                    # Check grasp success after applying action
                    current_height = env.get_object_height()
                    max_cube_height = max(max_cube_height, current_height)
                    
                    if not grasp_success and env.check_grasp_success():
                        grasp_success = True
                        print(f"\n✓ Grasp successful at step {step_idx}! Cube height: {current_height:.4f}m")
                        logger.info(f"Grasp successful at episode {episode_idx}, step {step_idx}, height: {current_height:.4f}m")
                        # End episode early on successful grasp
                        print(f"Episode ending early due to successful grasp")
                        break
                    
                except Exception as e:
                    logger.error(f"Failed to apply action at episode {episode_idx}, step {step_idx}: {e}\n{traceback.format_exc()}")
                    print(f"✗ Failed to apply action at step {step_idx}: {e}")
                    episode_errors += 1
                    if episode_errors >= 5:
                        logger.error(f"Too many errors in episode {episode_idx}, skipping...")
                        break
                    continue
                
                # Visualization / Logging
                try:
                    # Combine ego and side views for video (if side camera is enabled)
                    ego_img = obs["image"]["ego_rgb"]
                    
                    if args.use_side_camera and "side_rgb" in obs["image"]:
                        side_img = obs["image"]["side_rgb"]
                        combined_img = np.hstack([ego_img, side_img])
                    else:
                        combined_img = ego_img
                    
                    # Convert RGB to BGR for OpenCV
                    combined_img_bgr = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
                    video_frames.append(combined_img_bgr)
                except Exception as e:
                    logger.warning(f"Failed to process frame at episode {episode_idx}, step {step_idx}: {e}")
                
                if step_idx % 1 == 0:
                    current_height = env.get_object_height()
                    print(f"Step {step_idx}/{args.max_steps_per_episode} | Cube height: {current_height:.4f}m")

            # Episode summary
            final_height = env.get_object_height()
            height_delta = final_height - initial_cube_height
            
            print(f"\n{'='*40}")
            print(f"Episode {episode_idx} Summary:")
            print(f"  Initial cube height: {initial_cube_height:.4f}m")
            print(f"  Final cube height: {final_height:.4f}m")
            print(f"  Max cube height: {max_cube_height:.4f}m")
            print(f"  Height delta: {height_delta:.4f}m")
            print(f"  Grasp success: {'✓ YES' if grasp_success else '✗ NO'}")
            print(f"{'='*40}\n")
            
            logger.info(f"Episode {episode_idx} - Grasp: {grasp_success}, Max height: {max_cube_height:.4f}m, Delta: {height_delta:.4f}m")

            # Save Video
            logger.info(f"Saving video for episode {episode_idx}...")
            try:
                if len(video_frames) == 0:
                    logger.warning(f"No frames to save for episode {episode_idx}")
                else:
                    video_path = os.path.join(run_output_dir, f"episode_{episode_idx}.mp4")
                    height, width, _ = video_frames[0].shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
                    
                    for frame in video_frames:
                        out.write(frame)
                    out.release()
                    logger.info(f"✓ Saved video to {video_path}")
                    print(f"✓ Saved video to {video_path}")
            except Exception as e:
                logger.error(f"Failed to save video for episode {episode_idx}: {e}\n{traceback.format_exc()}")
                print(f"✗ Failed to save video: {e}")

        print("="*40)
        print("Inference Completed.")
        print("="*40)
        logger.info("Inference loop completed")
        
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}\n{traceback.format_exc()}")
        print(f"✗ Unexpected error: {e}")
        print(f"Check error log: {log_file}")
    finally:
        # 6. Cleanup
        logger.info("Cleaning up...")
        try:
            if env:
                env.close()
            if policy:
                del policy
            logger.info("✓ Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
