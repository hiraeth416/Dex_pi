#!/usr/bin/env python3
"""
DOF Converter for DexCanvas and IsaacSim MANO Hand

This module provides bidirectional conversion between DexCanvas URDF DOF format
and IsaacSim joint order for MANO hand control.

DexCanvas URDF DOF Order (26 DOF):
    0-5:   Base joints (wrist position/rotation: x, y, z, rx, ry, rz)
    6-9:   Thumb (abd, flex, mcp, ip)
    10-13: Index (abd, flex, pip, dip)
    14-17: Middle (abd, flex, pip, dip)
    18-21: Ring (abd, flex, pip, dip)
    22-25: Pinky (abd, flex, pip, dip)

IsaacSim Joint Order (26 DOF):
    0-5:   Base joints (same as DexCanvas)
    6-10:  Abduction joints (thumb, index, middle, ring, pinky)
    11-15: Flexion joints (thumb, index, middle, ring, pinky)
    16-20: MCP/PIP joints (thumb_mcp, index_pip, middle_pip, ring_pip, pinky_pip)
    21-25: IP/DIP joints (thumb_ip, index_dip, middle_dip, ring_dip, pinky_dip)
"""

import numpy as np
from typing import Union


# ============================================================================
# Joint Mapping Constants
# ============================================================================

# Mapping from DexCanvas URDF order to IsaacSim order
DEXCANVAS_TO_ISAACSIM = np.array([
    # Base joints (unchanged)
    0, 1, 2, 3, 4, 5,
    
    # Thumb: urdf[6,7,8,9] -> isaac[6,11,16,21]
    6, 11, 16, 21,
    
    # Index: urdf[10,11,12,13] -> isaac[7,12,17,22]
    7, 12, 17, 22,
    
    # Middle: urdf[14,15,16,17] -> isaac[8,13,18,23]
    8, 13, 18, 23,
    
    # Ring: urdf[18,19,20,21] -> isaac[9,14,19,24]
    9, 14, 19, 24,
    
    # Pinky: urdf[22,23,24,25] -> isaac[10,15,20,25]
    10, 15, 20, 25
], dtype=int)

# Mapping from IsaacSim order to DexCanvas URDF order (inverse mapping)
ISAACSIM_TO_DEXCANVAS = np.array([
    # Base joints (unchanged)
    0, 1, 2, 3, 4, 5,
    
    # Abduction joints: isaac[6-10] -> urdf[6,10,14,18,22]
    6, 10, 14, 18, 22,
    
    # Flexion joints: isaac[11-15] -> urdf[7,11,15,19,23]
    7, 11, 15, 19, 23,
    
    # MCP/PIP joints: isaac[16-20] -> urdf[8,12,16,20,24]
    8, 12, 16, 20, 24,
    
    # IP/DIP joints: isaac[21-25] -> urdf[9,13,17,21,25]
    9, 13, 17, 21, 25
], dtype=int)


# ============================================================================
# Conversion Functions
# ============================================================================

def dexcanvas_to_isaacsim(dof_dexcanvas: Union[np.ndarray, list]) -> np.ndarray:
    """
    Convert DOF from DexCanvas URDF order to IsaacSim joint order.
    
    This function remaps joint positions from the DexCanvas dataset format
    (grouped by finger) to the IsaacSim articulation format (grouped by joint type).
    
    Args:
        dof_dexcanvas: Joint positions in DexCanvas URDF order.
                       Shape: (26,) or (N, 26) where N is number of frames.
    
    Returns:
        Joint positions in IsaacSim order with same shape as input.
    
    Example:
        >>> dof_dexcanvas = np.random.randn(26)  # From DexCanvas dataset
        >>> dof_isaac = dexcanvas_to_isaacsim(dof_dexcanvas)
        >>> # Now ready to apply to IsaacSim hand articulation
        >>> action = ArticulationAction(joint_positions=dof_isaac)
        >>> hand_articulation.apply_action(action)
    """
    dof_array = np.asarray(dof_dexcanvas)
    
    if dof_array.ndim == 1:
        # Single frame: (26,) -> (26,)
        if dof_array.shape[0] != 26:
            raise ValueError(f"Expected 26 DOF, got {dof_array.shape[0]}")
        
        dof_isaac = np.zeros(26, dtype=dof_array.dtype)
        for urdf_idx, isaac_idx in enumerate(DEXCANVAS_TO_ISAACSIM):
            dof_isaac[isaac_idx] = dof_array[urdf_idx]
        return dof_isaac
    
    elif dof_array.ndim == 2:
        # Multiple frames: (N, 26) -> (N, 26)
        if dof_array.shape[1] != 26:
            raise ValueError(f"Expected 26 DOF per frame, got {dof_array.shape[1]}")
        
        num_frames = dof_array.shape[0]
        dof_isaac = np.zeros((num_frames, 26), dtype=dof_array.dtype)
        for urdf_idx, isaac_idx in enumerate(DEXCANVAS_TO_ISAACSIM):
            dof_isaac[:, isaac_idx] = dof_array[:, urdf_idx]
        return dof_isaac
    
    else:
        raise ValueError(f"Input must be 1D or 2D array, got shape {dof_array.shape}")


def isaacsim_to_dexcanvas(dof_isaac: Union[np.ndarray, list]) -> np.ndarray:
    """
    Convert DOF from IsaacSim joint order to DexCanvas URDF order.
    
    This function remaps joint positions from the IsaacSim articulation format
    (grouped by joint type) to the DexCanvas dataset format (grouped by finger).
    Useful for saving IsaacSim simulation results in DexCanvas format.
    
    Args:
        dof_isaac: Joint positions in IsaacSim order.
                   Shape: (26,) or (N, 26) where N is number of frames.
    
    Returns:
        Joint positions in DexCanvas URDF order with same shape as input.
    
    Example:
        >>> # Get current joint positions from IsaacSim
        >>> dof_isaac = hand_articulation.get_joint_positions()
        >>> # Convert to DexCanvas format for saving
        >>> dof_dexcanvas = isaacsim_to_dexcanvas(dof_isaac)
        >>> np.save("trajectory_dexcanvas_format.npy", dof_dexcanvas)
    """
    dof_array = np.asarray(dof_isaac)
    
    if dof_array.ndim == 1:
        # Single frame: (26,) -> (26,)
        if dof_array.shape[0] != 26:
            raise ValueError(f"Expected 26 DOF, got {dof_array.shape[0]}")
        
        dof_dexcanvas = np.zeros(26, dtype=dof_array.dtype)
        for isaac_idx, urdf_idx in enumerate(ISAACSIM_TO_DEXCANVAS):
            dof_dexcanvas[urdf_idx] = dof_array[isaac_idx]
        return dof_dexcanvas
    
    elif dof_array.ndim == 2:
        # Multiple frames: (N, 26) -> (N, 26)
        if dof_array.shape[1] != 26:
            raise ValueError(f"Expected 26 DOF per frame, got {dof_array.shape[1]}")
        
        num_frames = dof_array.shape[0]
        dof_dexcanvas = np.zeros((num_frames, 26), dtype=dof_array.dtype)
        for isaac_idx, urdf_idx in enumerate(ISAACSIM_TO_DEXCANVAS):
            dof_dexcanvas[:, urdf_idx] = dof_array[:, isaac_idx]
        return dof_dexcanvas
    
    else:
        raise ValueError(f"Input must be 1D or 2D array, got shape {dof_array.shape}")


# ============================================================================
# Verification Functions
# ============================================================================

def verify_mapping():
    """
    Verify that the mapping is bijective (one-to-one).
    
    This ensures that conversions are lossless and invertible.
    """
    # Check DexCanvas -> IsaacSim mapping
    dexcanvas_indices = np.arange(26)
    isaac_indices = DEXCANVAS_TO_ISAACSIM[dexcanvas_indices]
    
    if len(set(isaac_indices)) != 26:
        raise ValueError("DexCanvas->IsaacSim mapping is not bijective!")
    
    # Check IsaacSim -> DexCanvas mapping
    isaac_indices = np.arange(26)
    dexcanvas_indices = ISAACSIM_TO_DEXCANVAS[isaac_indices]
    
    if len(set(dexcanvas_indices)) != 26:
        raise ValueError("IsaacSim->DexCanvas mapping is not bijective!")
    
    # Check round-trip conversion
    test_dof = np.arange(26, dtype=float)
    
    # DexCanvas -> IsaacSim -> DexCanvas
    round_trip_1 = isaacsim_to_dexcanvas(dexcanvas_to_isaacsim(test_dof))
    if not np.allclose(test_dof, round_trip_1):
        raise ValueError("Round-trip DexCanvas->Isaac->DexCanvas failed!")
    
    # IsaacSim -> DexCanvas -> IsaacSim
    round_trip_2 = dexcanvas_to_isaacsim(isaacsim_to_dexcanvas(test_dof))
    if not np.allclose(test_dof, round_trip_2):
        raise ValueError("Round-trip Isaac->DexCanvas->Isaac failed!")
    
    print("✓ Mapping verification passed!")
    print("  - DexCanvas->IsaacSim mapping is bijective")
    print("  - IsaacSim->DexCanvas mapping is bijective")
    print("  - Round-trip conversions are lossless")


def print_mapping_table():
    """Print a human-readable mapping table for debugging."""
    joint_names_dexcanvas = [
        "base_x", "base_y", "base_z", "base_rx", "base_ry", "base_rz",
        "thumb_abd", "thumb_flex", "thumb_mcp", "thumb_ip",
        "index_abd", "index_flex", "index_pip", "index_dip",
        "middle_abd", "middle_flex", "middle_pip", "middle_dip",
        "ring_abd", "ring_flex", "ring_pip", "ring_dip",
        "pinky_abd", "pinky_flex", "pinky_pip", "pinky_dip",
    ]
    
    joint_names_isaac = [
        "base_x", "base_y", "base_z", "base_rx", "base_ry", "base_rz",
        "thumb_abd", "index_abd", "middle_abd", "ring_abd", "pinky_abd",
        "thumb_flex", "index_flex", "middle_flex", "ring_flex", "pinky_flex",
        "thumb_mcp", "index_pip", "middle_pip", "ring_pip", "pinky_pip",
        "thumb_ip", "index_dip", "middle_dip", "ring_dip", "pinky_dip",
    ]
    
    print("\n" + "=" * 70)
    print("DexCanvas URDF -> IsaacSim Joint Mapping")
    print("=" * 70)
    print(f"{'DexCanvas Idx':<15} {'Joint Name':<20} {'IsaacSim Idx':<15} {'Joint Name':<20}")
    print("-" * 70)
    
    for dex_idx in range(26):
        isaac_idx = DEXCANVAS_TO_ISAACSIM[dex_idx]
        print(f"{dex_idx:<15} {joint_names_dexcanvas[dex_idx]:<20} "
              f"{isaac_idx:<15} {joint_names_isaac[isaac_idx]:<20}")
    
    print("=" * 70 + "\n")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Verify mappings
    verify_mapping()
    
    # Print mapping table
    print_mapping_table()
    
    # Example 1: Single frame conversion
    print("\n=== Example 1: Single Frame Conversion ===")
    dof_dexcanvas = np.random.randn(26)
    print(f"DexCanvas DOF (first 6): {dof_dexcanvas[:6]}")
    
    dof_isaac = dexcanvas_to_isaacsim(dof_dexcanvas)
    print(f"IsaacSim DOF (first 6):  {dof_isaac[:6]}")
    
    dof_back = isaacsim_to_dexcanvas(dof_isaac)
    print(f"Back to DexCanvas (first 6): {dof_back[:6]}")
    print(f"Conversion error: {np.max(np.abs(dof_dexcanvas - dof_back)):.2e}")
    
    # Example 2: Multiple frames conversion
    print("\n=== Example 2: Multiple Frames Conversion ===")
    num_frames = 100
    trajectory_dexcanvas = np.random.randn(num_frames, 26)
    print(f"Input shape: {trajectory_dexcanvas.shape}")
    
    trajectory_isaac = dexcanvas_to_isaacsim(trajectory_dexcanvas)
    print(f"IsaacSim shape: {trajectory_isaac.shape}")
    
    trajectory_back = isaacsim_to_dexcanvas(trajectory_isaac)
    print(f"Output shape: {trajectory_back.shape}")
    print(f"Max conversion error: {np.max(np.abs(trajectory_dexcanvas - trajectory_back)):.2e}")
    
    print("\n✓ All examples completed successfully!")
