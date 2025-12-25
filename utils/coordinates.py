"""
Coordinate System Transformations

Supports transformations between:
- COLMAP (Y-down, Z-forward) 
- Blender (Z-up, Y-forward)
- OpenGL (Y-up, -Z-forward)
- OpenCV (Y-down, Z-forward)
- Unreal (Z-up, X-forward)
- Custom (user-defined matrix)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from enum import Enum


class CoordinateSystem(Enum):
    COLMAP = "colmap"       # Y-down, Z-forward (right-handed)
    BLENDER = "blender"     # Z-up, Y-forward (right-handed)
    OPENGL = "opengl"       # Y-up, -Z-forward (right-handed)
    OPENCV = "opencv"       # Y-down, Z-forward (right-handed) - same as COLMAP
    UNREAL = "unreal"       # Z-up, X-forward (left-handed)
    UNITY = "unity"         # Y-up, Z-forward (left-handed)
    HOUDINI = "houdini"     # Y-up, -Z-forward (right-handed) - same as OpenGL
    MAYA = "maya"           # Y-up, Z-forward (right-handed)
    USD = "usd"             # Y-up, -Z-forward (right-handed) - same as OpenGL
    CUSTOM = "custom"


# Transformation matrices FROM COLMAP TO target coordinate system
# COLMAP uses: X-right, Y-down, Z-forward (camera looking down +Z)
COORD_TRANSFORMS: Dict[CoordinateSystem, np.ndarray] = {
    # Identity - no transform needed
    CoordinateSystem.COLMAP: np.eye(4),
    CoordinateSystem.OPENCV: np.eye(4),
    
    # COLMAP to Blender: swap Y and Z, negate new Y
    # COLMAP: X-right, Y-down, Z-forward
    # Blender: X-right, Y-forward, Z-up
    CoordinateSystem.BLENDER: np.array([
        [1,  0,  0,  0],
        [0,  0,  1,  0],
        [0, -1,  0,  0],
        [0,  0,  0,  1]
    ], dtype=np.float64),
    
    # COLMAP to OpenGL: negate Y and Z
    # COLMAP: X-right, Y-down, Z-forward
    # OpenGL: X-right, Y-up, Z-backward
    CoordinateSystem.OPENGL: np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ], dtype=np.float64),
    
    # COLMAP to Unreal: rotate 90° around X, swap handedness
    # COLMAP: X-right, Y-down, Z-forward (RH)
    # Unreal: X-forward, Y-right, Z-up (LH)
    CoordinateSystem.UNREAL: np.array([
        [0,  0,  1,  0],
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0,  0,  1]
    ], dtype=np.float64),
    
    # COLMAP to Unity: negate Z (flip handedness)
    # COLMAP: X-right, Y-down, Z-forward (RH)
    # Unity: X-right, Y-up, Z-forward (LH)
    CoordinateSystem.UNITY: np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0,  1,  0],
        [0,  0,  0,  1]
    ], dtype=np.float64),
    
    # Same as OpenGL
    CoordinateSystem.HOUDINI: np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ], dtype=np.float64),
    
    # Maya: X-right, Y-up, Z-forward (slight adjustment)
    CoordinateSystem.MAYA: np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0,  1,  0],
        [0,  0,  0,  1]
    ], dtype=np.float64),
    
    # USD uses OpenGL conventions
    CoordinateSystem.USD: np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ], dtype=np.float64),
}


def get_coordinate_transform(
    source: CoordinateSystem,
    target: CoordinateSystem,
    custom_matrix: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Get transformation matrix from source to target coordinate system.
    
    Args:
        source: Source coordinate system
        target: Target coordinate system
        custom_matrix: Optional 4x4 transformation matrix for CUSTOM target
        
    Returns:
        4x4 transformation matrix
    """
    if target == CoordinateSystem.CUSTOM:
        if custom_matrix is None:
            raise ValueError("Custom matrix required for CUSTOM coordinate system")
        return custom_matrix
    
    # Get transforms relative to COLMAP
    if source == CoordinateSystem.COLMAP:
        return COORD_TRANSFORMS[target]
    
    if target == CoordinateSystem.COLMAP:
        # Inverse of source-to-COLMAP
        return np.linalg.inv(COORD_TRANSFORMS[source])
    
    # Chain: source -> COLMAP -> target
    source_to_colmap = np.linalg.inv(COORD_TRANSFORMS[source])
    colmap_to_target = COORD_TRANSFORMS[target]
    return colmap_to_target @ source_to_colmap


def transform_position(
    position: np.ndarray,
    transform: np.ndarray
) -> np.ndarray:
    """Transform a 3D position using a 4x4 transformation matrix."""
    pos_homogeneous = np.append(position, 1.0)
    transformed = transform @ pos_homogeneous
    return transformed[:3]


def transform_rotation_matrix(
    rotation: np.ndarray,
    transform: np.ndarray
) -> np.ndarray:
    """Transform a 3x3 rotation matrix to a new coordinate system."""
    # Extract rotation part of transform (upper-left 3x3)
    rot_transform = transform[:3, :3]
    return rot_transform @ rotation @ rot_transform.T


def rotation_matrix_to_euler(
    R: np.ndarray,
    order: str = 'xyz'
) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles in degrees.
    
    Args:
        R: 3x3 rotation matrix
        order: Rotation order ('xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx')
        
    Returns:
        Tuple of (rx, ry, rz) in degrees
    """
    # Clamp values to avoid numerical issues
    def clamp(x, min_val=-1.0, max_val=1.0):
        return max(min_val, min(max_val, x))
    
    if order == 'xyz':
        # Tait-Bryan angles: Rx * Ry * Rz
        sy = clamp(R[0, 2])
        ry = np.arcsin(sy)
        
        if np.abs(sy) < 0.9999:
            rx = np.arctan2(-R[1, 2], R[2, 2])
            rz = np.arctan2(-R[0, 1], R[0, 0])
        else:
            rx = np.arctan2(R[2, 1], R[1, 1])
            rz = 0
            
    elif order == 'zyx':
        # Common for aerospace
        sy = clamp(-R[2, 0])
        ry = np.arcsin(sy)
        
        if np.abs(sy) < 0.9999:
            rx = np.arctan2(R[2, 1], R[2, 2])
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = 0
            rz = np.arctan2(-R[0, 1], R[1, 1])
    else:
        # Default to XYZ
        return rotation_matrix_to_euler(R, 'xyz')
    
    return (
        np.degrees(rx),
        np.degrees(ry),
        np.degrees(rz)
    )


def euler_to_rotation_matrix(
    rx: float, ry: float, rz: float,
    order: str = 'xyz',
    degrees: bool = True
) -> np.ndarray:
    """
    Convert Euler angles to rotation matrix.
    
    Args:
        rx, ry, rz: Rotation angles around X, Y, Z axes
        order: Rotation order
        degrees: If True, angles are in degrees; otherwise radians
        
    Returns:
        3x3 rotation matrix
    """
    if degrees:
        rx = np.radians(rx)
        ry = np.radians(ry)
        rz = np.radians(rz)
    
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    if order == 'xyz':
        return Rz @ Ry @ Rx
    elif order == 'zyx':
        return Rx @ Ry @ Rz
    else:
        return Rz @ Ry @ Rx


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Normalize
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n > 0:
        w, x, y, z = w/n, x/n, y/n, z/n
    
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion [w, x, y, z].
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])


def compute_scale_factor(
    source_system: CoordinateSystem,
    target_system: CoordinateSystem,
    source_unit: str = "meters",
    target_unit: str = "meters"
) -> float:
    """
    Compute scale factor for unit conversion.
    
    Args:
        source_system: Source coordinate system
        target_system: Target coordinate system
        source_unit: Source unit ('meters', 'centimeters', 'millimeters', 'inches', 'feet')
        target_unit: Target unit
        
    Returns:
        Scale factor to multiply positions by
    """
    unit_to_meters = {
        "meters": 1.0,
        "centimeters": 0.01,
        "millimeters": 0.001,
        "inches": 0.0254,
        "feet": 0.3048,
    }
    
    source_scale = unit_to_meters.get(source_unit, 1.0)
    target_scale = unit_to_meters.get(target_unit, 1.0)
    
    return source_scale / target_scale
