"""
ComfyUI-COLMAP Utilities
"""

from .coordinates import (
    CoordinateSystem,
    get_coordinate_transform,
    transform_position,
    transform_rotation_matrix,
    rotation_matrix_to_euler,
    euler_to_rotation_matrix,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    compute_scale_factor,
)

from .camera_types import (
    CameraModel,
    MotionType,
    CameraIntrinsics,
    CameraExtrinsics,
    CameraMotion,
    SparsePoint,
    CameraData,
)

from .colmap_wrapper import (
    GPUMode,
    MatcherType,
    FeatureType,
    COLMAPWrapper,
    COLMAPError,
    check_colmap_installation,
    PYCOLMAP_AVAILABLE,
)

__all__ = [
    # Coordinates
    "CoordinateSystem",
    "get_coordinate_transform",
    "transform_position",
    "transform_rotation_matrix",
    "rotation_matrix_to_euler",
    "euler_to_rotation_matrix",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "compute_scale_factor",
    # Camera types
    "CameraModel",
    "MotionType",
    "CameraIntrinsics",
    "CameraExtrinsics",
    "CameraMotion",
    "SparsePoint",
    "CameraData",
    # COLMAP wrapper
    "GPUMode",
    "MatcherType", 
    "FeatureType",
    "COLMAPWrapper",
    "COLMAPError",
    "check_colmap_installation",
    "PYCOLMAP_AVAILABLE",
]
