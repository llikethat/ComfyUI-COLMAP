"""
Camera Data Types for ComfyUI-COLMAP

Defines the CAMERA_DATA type that can be passed between nodes
and potentially shared with SAM3DBody for combined workflows.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json


class CameraModel(Enum):
    """Camera distortion models supported by COLMAP"""
    SIMPLE_PINHOLE = "SIMPLE_PINHOLE"       # f, cx, cy
    PINHOLE = "PINHOLE"                     # fx, fy, cx, cy
    SIMPLE_RADIAL = "SIMPLE_RADIAL"         # f, cx, cy, k1
    RADIAL = "RADIAL"                       # f, cx, cy, k1, k2
    OPENCV = "OPENCV"                       # fx, fy, cx, cy, k1, k2, p1, p2
    OPENCV_FISHEYE = "OPENCV_FISHEYE"       # fx, fy, cx, cy, k1, k2, k3, k4
    FULL_OPENCV = "FULL_OPENCV"             # fx, fy, cx, cy, k1-k6, p1, p2


class MotionType(Enum):
    """Detected camera motion types"""
    STATIC = "static"           # Tripod, locked off
    HANDHELD = "handheld"       # Subtle shake, organic movement
    PAN = "pan"                 # Horizontal rotation
    TILT = "tilt"               # Vertical rotation
    ROLL = "roll"               # Rotation around view axis
    DOLLY = "dolly"             # Forward/backward translation
    TRUCK = "truck"             # Left/right translation
    CRANE = "crane"             # Up/down translation
    ORBIT = "orbit"             # Circular movement around subject
    DRONE = "drone"             # 3D flight path
    TRACKING = "tracking"       # Following a subject
    ZOOM = "zoom"               # Focal length change (detected via FOV)
    COMPLEX = "complex"         # Multiple combined motions


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    model: CameraModel = CameraModel.PINHOLE
    width: int = 1920
    height: int = 1080
    
    # Core parameters
    focal_length_x: float = 1000.0      # fx in pixels
    focal_length_y: float = 1000.0      # fy in pixels
    principal_point_x: float = 960.0    # cx in pixels
    principal_point_y: float = 540.0    # cy in pixels
    
    # Distortion parameters (optional)
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    p1: float = 0.0     # Tangential distortion
    p2: float = 0.0
    
    @property
    def focal_length_mm(self) -> float:
        """Estimate focal length in mm assuming 35mm full-frame equivalent"""
        sensor_width_mm = 36.0  # Full-frame sensor width
        return (self.focal_length_x / self.width) * sensor_width_mm
    
    @property
    def fov_horizontal(self) -> float:
        """Horizontal field of view in degrees"""
        return 2 * np.degrees(np.arctan(self.width / (2 * self.focal_length_x)))
    
    @property
    def fov_vertical(self) -> float:
        """Vertical field of view in degrees"""
        return 2 * np.degrees(np.arctan(self.height / (2 * self.focal_length_y)))
    
    def to_matrix(self) -> np.ndarray:
        """Get 3x3 intrinsic matrix K"""
        return np.array([
            [self.focal_length_x, 0, self.principal_point_x],
            [0, self.focal_length_y, self.principal_point_y],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.value,
            "width": self.width,
            "height": self.height,
            "fx": self.focal_length_x,
            "fy": self.focal_length_y,
            "cx": self.principal_point_x,
            "cy": self.principal_point_y,
            "focal_length_mm": self.focal_length_mm,
            "fov_h": self.fov_horizontal,
            "fov_v": self.fov_vertical,
            "distortion": {
                "k1": self.k1, "k2": self.k2, "k3": self.k3,
                "k4": self.k4, "k5": self.k5, "k6": self.k6,
                "p1": self.p1, "p2": self.p2
            }
        }


@dataclass
class CameraExtrinsics:
    """Camera extrinsic parameters for a single frame"""
    frame_index: int = 0
    image_name: str = ""
    
    # Camera pose in world coordinates
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    
    # Optional: direct quaternion storage
    quaternion: Optional[np.ndarray] = None  # [w, x, y, z]
    
    # Confidence/quality metrics
    num_observations: int = 0       # Number of 3D points observed
    reprojection_error: float = 0.0 # Mean reprojection error in pixels
    
    @property
    def rotation_euler(self) -> Tuple[float, float, float]:
        """Get rotation as Euler angles (rx, ry, rz) in degrees"""
        from .coordinates import rotation_matrix_to_euler
        return rotation_matrix_to_euler(self.rotation_matrix)
    
    @property
    def cam_from_world(self) -> np.ndarray:
        """Get 4x4 transformation matrix (world to camera)"""
        T = np.eye(4)
        T[:3, :3] = self.rotation_matrix
        T[:3, 3] = -self.rotation_matrix @ self.position
        return T
    
    @property
    def world_from_cam(self) -> np.ndarray:
        """Get 4x4 transformation matrix (camera to world)"""
        T = np.eye(4)
        T[:3, :3] = self.rotation_matrix.T
        T[:3, 3] = self.position
        return T
    
    def to_dict(self) -> Dict[str, Any]:
        rx, ry, rz = self.rotation_euler
        return {
            "frame_index": self.frame_index,
            "image_name": self.image_name,
            "position": self.position.tolist(),
            "rotation_euler": {"rx": rx, "ry": ry, "rz": rz},
            "rotation_matrix": self.rotation_matrix.tolist(),
            "quaternion": self.quaternion.tolist() if self.quaternion is not None else None,
            "num_observations": self.num_observations,
            "reprojection_error": self.reprojection_error
        }


@dataclass
class CameraMotion:
    """Camera motion analysis for a single frame"""
    frame_index: int = 0
    
    # Rotation velocities (degrees per frame)
    pan: float = 0.0        # Horizontal rotation velocity
    tilt: float = 0.0       # Vertical rotation velocity  
    roll: float = 0.0       # Roll velocity
    
    # Translation velocities (units per frame)
    dolly: float = 0.0      # Forward/backward velocity
    truck: float = 0.0      # Left/right velocity
    crane: float = 0.0      # Up/down velocity
    
    # Derived metrics
    total_rotation: float = 0.0     # Total rotation magnitude
    total_translation: float = 0.0  # Total translation magnitude
    speed: float = 0.0              # 3D velocity magnitude
    
    # Classification
    motion_type: MotionType = MotionType.STATIC
    motion_confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "rotation": {
                "pan": self.pan,
                "tilt": self.tilt,
                "roll": self.roll,
                "total": self.total_rotation
            },
            "translation": {
                "dolly": self.dolly,
                "truck": self.truck,
                "crane": self.crane,
                "total": self.total_translation
            },
            "speed": self.speed,
            "motion_type": self.motion_type.value,
            "motion_confidence": self.motion_confidence
        }


@dataclass  
class SparsePoint:
    """A 3D point from sparse reconstruction"""
    point_id: int
    xyz: np.ndarray
    rgb: Optional[np.ndarray] = None
    error: float = 0.0
    num_observations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.point_id,
            "xyz": self.xyz.tolist(),
            "rgb": self.rgb.tolist() if self.rgb is not None else None,
            "error": self.error,
            "num_observations": self.num_observations
        }


@dataclass
class CameraData:
    """
    Complete camera data from COLMAP reconstruction.
    
    This is the main data type passed between COLMAP nodes
    and can be used for export or integration with SAM3DBody.
    """
    # Reconstruction metadata
    reconstruction_id: int = 0
    num_frames: int = 0
    fps: float = 24.0
    
    # Camera parameters
    intrinsics: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    
    # Per-frame data
    extrinsics: List[CameraExtrinsics] = field(default_factory=list)
    motion: List[CameraMotion] = field(default_factory=list)
    
    # Sparse point cloud (optional)
    sparse_points: List[SparsePoint] = field(default_factory=list)
    
    # Quality metrics
    mean_reprojection_error: float = 0.0
    registered_frames: int = 0
    total_frames: int = 0
    
    # Coordinate system info
    coordinate_system: str = "colmap"
    scale: float = 1.0  # Scale factor (e.g., if scene was normalized)
    
    @property
    def registration_ratio(self) -> float:
        """Fraction of frames successfully registered"""
        if self.total_frames == 0:
            return 0.0
        return self.registered_frames / self.total_frames
    
    def get_frame(self, frame_index: int) -> Optional[CameraExtrinsics]:
        """Get extrinsics for a specific frame"""
        for ext in self.extrinsics:
            if ext.frame_index == frame_index:
                return ext
        return None
    
    def get_motion(self, frame_index: int) -> Optional[CameraMotion]:
        """Get motion data for a specific frame"""
        for m in self.motion:
            if m.frame_index == frame_index:
                return m
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export"""
        return {
            "metadata": {
                "reconstruction_id": self.reconstruction_id,
                "num_frames": self.num_frames,
                "fps": self.fps,
                "coordinate_system": self.coordinate_system,
                "scale": self.scale,
                "registered_frames": self.registered_frames,
                "total_frames": self.total_frames,
                "mean_reprojection_error": self.mean_reprojection_error
            },
            "intrinsics": self.intrinsics.to_dict(),
            "frames": [ext.to_dict() for ext in self.extrinsics],
            "motion": [m.to_dict() for m in self.motion],
            "sparse_points_count": len(self.sparse_points)
        }
    
    def to_json(self, include_points: bool = False) -> str:
        """Export to JSON string"""
        data = self.to_dict()
        if include_points:
            data["sparse_points"] = [p.to_dict() for p in self.sparse_points]
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraData":
        """Create CameraData from dictionary"""
        meta = data.get("metadata", {})
        intr_data = data.get("intrinsics", {})
        
        intrinsics = CameraIntrinsics(
            model=CameraModel(intr_data.get("model", "PINHOLE")),
            width=intr_data.get("width", 1920),
            height=intr_data.get("height", 1080),
            focal_length_x=intr_data.get("fx", 1000),
            focal_length_y=intr_data.get("fy", 1000),
            principal_point_x=intr_data.get("cx", 960),
            principal_point_y=intr_data.get("cy", 540),
        )
        
        extrinsics = []
        for frame_data in data.get("frames", []):
            ext = CameraExtrinsics(
                frame_index=frame_data.get("frame_index", 0),
                image_name=frame_data.get("image_name", ""),
                position=np.array(frame_data.get("position", [0, 0, 0])),
                rotation_matrix=np.array(frame_data.get("rotation_matrix", np.eye(3).tolist())),
            )
            extrinsics.append(ext)
        
        motion = []
        for m_data in data.get("motion", []):
            m = CameraMotion(
                frame_index=m_data.get("frame_index", 0),
                pan=m_data.get("rotation", {}).get("pan", 0),
                tilt=m_data.get("rotation", {}).get("tilt", 0),
                roll=m_data.get("rotation", {}).get("roll", 0),
                dolly=m_data.get("translation", {}).get("dolly", 0),
                truck=m_data.get("translation", {}).get("truck", 0),
                crane=m_data.get("translation", {}).get("crane", 0),
                motion_type=MotionType(m_data.get("motion_type", "static")),
            )
            motion.append(m)
        
        return cls(
            reconstruction_id=meta.get("reconstruction_id", 0),
            num_frames=meta.get("num_frames", len(extrinsics)),
            fps=meta.get("fps", 24.0),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            motion=motion,
            coordinate_system=meta.get("coordinate_system", "colmap"),
            scale=meta.get("scale", 1.0),
            registered_frames=meta.get("registered_frames", len(extrinsics)),
            total_frames=meta.get("total_frames", len(extrinsics)),
            mean_reprojection_error=meta.get("mean_reprojection_error", 0.0),
        )
