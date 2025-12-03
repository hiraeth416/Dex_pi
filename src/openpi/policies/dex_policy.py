import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_dex_example() -> dict:
    """Creates a random input example for the Dex policy."""
    return {
        "state": np.random.rand(26),
        "image": {
            "ego_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "side_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        },
        "prompt": "grasp the object",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DexInputs(transforms.DataTransformFn):
    """
    Transform inputs from DexCanvas dataset to the model's expected format.
    
    The DexCanvas dataset provides:
    - state: [26] robot DOF (urdf_dof)
    - image: dict with ego_rgb, side_rgb
    - prompt: text instruction
    - actions: [action_horizon, 26] next frame's urdf_dof
    
    This transform remaps DexCanvas camera keys to pi0's expected keys:
    - ego_rgb -> base_0_rgb
    - side_rgb -> left_wrist_0_rgb
    - Creates dummy right_wrist_0_rgb (zeros)
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Remap DexCanvas camera keys to pi0's expected keys
        images = data["image"]
        image_masks = data.get("image_mask", {})
        
        # Create remapped image dict with pi0's expected keys
        remapped_images = {
            "base_0_rgb": images.get("ego_rgb"),
            "left_wrist_0_rgb": images.get("side_rgb"),
            # Create dummy right_wrist_0_rgb (not available in DexCanvas)
            "right_wrist_0_rgb": np.zeros_like(images.get("ego_rgb")),
        }
        
        remapped_masks = {
            "base_0_rgb": image_masks.get("ego_rgb", np.True_),
            "left_wrist_0_rgb": image_masks.get("side_rgb", np.False_),
            # We only mask padding images for pi0 model, not pi0-FAST
            "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
        }
        
        # Create inputs dict
        inputs = {
            "state": data["state"],
            "image": remapped_images,
            "image_mask": remapped_masks,
        }

        # Pass actions during training
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (language instruction)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DexOutputs(transforms.DataTransformFn):
    """
    Transform model outputs back to the DexCanvas dataset format.
    
    The model outputs 32-dim actions (padded), but we only need the first 26 dimensions
    which correspond to the robot's actual DOF (urdf_dof).
    
    This is used for inference only.
    """

    def __call__(self, data: dict) -> dict:
        # Return only the first 26 actions (robot DOF)
        # The model is configured with action_dim=32 (padded), but we only use 26 DOF
        return {"actions": np.asarray(data["actions"][:, :26])}
