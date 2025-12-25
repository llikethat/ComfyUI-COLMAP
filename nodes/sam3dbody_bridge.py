"""
COLMAP to SAM3DBody Bridge Node

Converts COLMAP camera data to SAM3DBody2ABC format for integrated
camera solve + body tracking workflows.

This enables:
1. Using COLMAP's robust camera tracking with SAM3DBody's pose estimation
2. Proper camera-to-body alignment in exports
3. Frame-accurate synchronization between camera and body data
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional, List
import math


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles (pan, tilt, roll).
    
    Uses YXZ order which matches typical camera conventions:
    - Pan (Y rotation): horizontal rotation
    - Tilt (X rotation): vertical rotation  
    - Roll (Z rotation): rotation around view axis
    
    Returns angles in radians.
    """
    # Extract Euler angles assuming YXZ order
    # This is the typical convention for camera rotations
    
    # Clamp to avoid numerical issues
    sy = np.clip(-R[2, 0], -1.0, 1.0)
    
    if abs(sy) < 0.99999:
        # Normal case
        pan = math.atan2(R[2, 0], R[2, 2])    # Y rotation (horizontal)
        tilt = math.asin(-R[2, 1]) if abs(R[2, 1]) < 1 else 0  # X rotation (vertical)
        roll = math.atan2(R[0, 1], R[1, 1])    # Z rotation
    else:
        # Gimbal lock
        pan = math.atan2(-R[0, 2], R[0, 0])
        tilt = math.pi / 2 * np.sign(sy)
        roll = 0
    
    return pan, tilt, roll


def rotation_matrix_to_euler_xyz(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to XYZ Euler angles.
    Returns (rx, ry, rz) in radians.
    """
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    singular = sy < 1e-6
    
    if not singular:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0
    
    return rx, ry, rz


class COLMAPToSAM3DBodyCamera:
    """
    Convert COLMAP camera data to SAM3DBody2ABC camera rotation format.
    
    This enables using COLMAP's background-based camera tracking to
    inform SAM3DBody's body pose estimation, resulting in more accurate
    world-space body positioning.
    
    The output CAMERA_ROTATION_DATA can be connected to:
    - ExportAnimatedFBX node's camera_rotations input
    - Other SAM3DBody nodes that accept camera data
    
    Frame Numbering:
    - COLMAP frame indices come from image filenames (e.g., frame_000003.jpg → frame 3)
    - Use frame_offset to remap to video frames (e.g., offset=-2 maps frame 3 → frame 1)
    - Auto mode tries to detect the minimum frame and offset automatically
    """
    
    CATEGORY = "COLMAP"
    FUNCTION = "convert"
    RETURN_TYPES = ("CAMERA_ROTATION_DATA", "STRING")
    RETURN_NAMES = ("camera_rotations", "status")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_data": ("CAMERA_DATA",),
            },
            "optional": {
                "frame_offset": ("INT", {
                    "default": 0,
                    "min": -1000,
                    "max": 1000,
                    "tooltip": "Offset to apply to frame numbers. Use 'auto' logic: if min frame > 1, auto-offset to start at 1."
                }),
                "auto_frame_offset": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically offset frames to start from frame 1"
                }),
                "interpolate_missing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Interpolate camera data for frames that COLMAP didn't register"
                }),
                "total_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "Total frames in video (0 = auto from camera_data)"
                }),
                "euler_order": (["YXZ (Camera)", "XYZ", "ZYX"], {
                    "default": "YXZ (Camera)",
                    "tooltip": "Euler angle extraction order. YXZ is standard for camera pan/tilt/roll."
                }),
            }
        }
    
    def convert(
        self,
        camera_data: Any,
        frame_offset: int = 0,
        auto_frame_offset: bool = True,
        interpolate_missing: bool = True,
        total_frames: int = 0,
        euler_order: str = "YXZ (Camera)",
    ) -> Tuple[Dict, str]:
        """Convert COLMAP camera data to SAM3DBody format."""
        
        if not camera_data or not camera_data.extrinsics:
            return (
                self._empty_rotation_data(),
                "ERROR: No camera data available"
            )
        
        print(f"[COLMAP→SAM3D] Converting {len(camera_data.extrinsics)} camera frames...")
        
        # Get frame indices from COLMAP data
        colmap_frames = sorted([ext.frame_index for ext in camera_data.extrinsics])
        min_frame = min(colmap_frames)
        max_frame = max(colmap_frames)
        
        print(f"[COLMAP→SAM3D] COLMAP frame range: {min_frame} - {max_frame}")
        
        # Calculate frame offset
        if auto_frame_offset and min_frame > 1:
            # Auto-offset to start from frame 1
            calculated_offset = -(min_frame - 1)
            print(f"[COLMAP→SAM3D] Auto frame offset: {calculated_offset} (frames will start at 1)")
        else:
            calculated_offset = frame_offset
            print(f"[COLMAP→SAM3D] Using manual frame offset: {calculated_offset}")
        
        # Determine total frames
        if total_frames > 0:
            num_frames = total_frames
        else:
            num_frames = camera_data.total_frames if camera_data.total_frames > 0 else (max_frame + calculated_offset)
        
        # Extract rotations from COLMAP data
        colmap_rotations = {}
        for ext in camera_data.extrinsics:
            R = ext.rotation_matrix
            
            # Extract Euler angles based on order
            if "YXZ" in euler_order:
                pan, tilt, roll = rotation_matrix_to_euler(R)
            else:
                rx, ry, rz = rotation_matrix_to_euler_xyz(R)
                if "XYZ" in euler_order:
                    pan, tilt, roll = ry, rx, rz
                else:  # ZYX
                    pan, tilt, roll = ry, rx, rz
            
            # Apply frame offset
            output_frame = ext.frame_index + calculated_offset
            
            colmap_rotations[output_frame] = {
                "frame": output_frame,
                "pan": pan,
                "tilt": tilt,
                "roll": roll,
                "pan_deg": np.degrees(pan),
                "tilt_deg": np.degrees(tilt),
                "roll_deg": np.degrees(roll),
                "position": ext.position.copy(),
                "has_data": True,
            }
        
        # Build full rotation list with interpolation
        rotations = []
        
        for frame_idx in range(1, num_frames + 1):
            if frame_idx in colmap_rotations:
                rotations.append(colmap_rotations[frame_idx])
            elif interpolate_missing:
                # Interpolate from nearest known frames
                interpolated = self._interpolate_frame(frame_idx, colmap_rotations)
                rotations.append(interpolated)
            else:
                # No data, use identity
                rotations.append({
                    "frame": frame_idx,
                    "pan": 0.0,
                    "tilt": 0.0,
                    "roll": 0.0,
                    "pan_deg": 0.0,
                    "tilt_deg": 0.0,
                    "roll_deg": 0.0,
                    "has_data": False,
                })
        
        # Get intrinsics
        intr = camera_data.intrinsics
        focal_length_px = intr.focal_length_x if intr else 1000.0
        image_width = intr.width if intr else 1920
        image_height = intr.height if intr else 1080
        
        # Build output data in SAM3DBody format
        camera_rotation_data = {
            "num_frames": len(rotations),
            "image_width": image_width,
            "image_height": image_height,
            "focal_length_px": focal_length_px,
            "tracking_method": "COLMAP Structure from Motion",
            "frame_offset_applied": calculated_offset,
            "colmap_registered_frames": len(camera_data.extrinsics),
            "interpolated_frames": len(rotations) - len(colmap_rotations),
            "rotations": rotations,
            # Additional COLMAP-specific data
            "colmap_data": {
                "mean_reprojection_error": camera_data.mean_reprojection_error,
                "sparse_points_count": len(camera_data.sparse_points),
                "coordinate_system": camera_data.coordinate_system,
            }
        }
        
        # Status summary
        registered = len(camera_data.extrinsics)
        interpolated = len(rotations) - len(colmap_rotations)
        status = f"Converted {registered} COLMAP frames, {interpolated} interpolated, {len(rotations)} total"
        
        if rotations:
            final = rotations[-1]
            print(f"[COLMAP→SAM3D] Final frame {final['frame']}: pan={final['pan_deg']:.2f}°, tilt={final['tilt_deg']:.2f}°")
        
        print(f"[COLMAP→SAM3D] {status}")
        
        return (camera_rotation_data, status)
    
    def _interpolate_frame(
        self,
        frame_idx: int,
        known_rotations: Dict[int, Dict]
    ) -> Dict:
        """Interpolate rotation for a missing frame."""
        
        known_frames = sorted(known_rotations.keys())
        
        if not known_frames:
            return {
                "frame": frame_idx,
                "pan": 0.0, "tilt": 0.0, "roll": 0.0,
                "pan_deg": 0.0, "tilt_deg": 0.0, "roll_deg": 0.0,
                "has_data": False,
            }
        
        # Find surrounding frames
        prev_frame = None
        next_frame = None
        
        for f in known_frames:
            if f <= frame_idx:
                prev_frame = f
            if f >= frame_idx and next_frame is None:
                next_frame = f
        
        # Edge cases
        if prev_frame is None:
            prev_frame = known_frames[0]
        if next_frame is None:
            next_frame = known_frames[-1]
        
        if prev_frame == next_frame:
            # Use single known frame
            src = known_rotations[prev_frame]
            return {
                "frame": frame_idx,
                "pan": src["pan"],
                "tilt": src["tilt"],
                "roll": src["roll"],
                "pan_deg": src["pan_deg"],
                "tilt_deg": src["tilt_deg"],
                "roll_deg": src["roll_deg"],
                "has_data": False,  # Mark as interpolated
            }
        
        # Linear interpolation
        t = (frame_idx - prev_frame) / (next_frame - prev_frame)
        
        prev_rot = known_rotations[prev_frame]
        next_rot = known_rotations[next_frame]
        
        pan = prev_rot["pan"] + t * (next_rot["pan"] - prev_rot["pan"])
        tilt = prev_rot["tilt"] + t * (next_rot["tilt"] - prev_rot["tilt"])
        roll = prev_rot["roll"] + t * (next_rot["roll"] - prev_rot["roll"])
        
        return {
            "frame": frame_idx,
            "pan": pan,
            "tilt": tilt,
            "roll": roll,
            "pan_deg": np.degrees(pan),
            "tilt_deg": np.degrees(tilt),
            "roll_deg": np.degrees(roll),
            "has_data": False,  # Mark as interpolated
        }
    
    def _empty_rotation_data(self) -> Dict:
        """Return empty rotation data structure."""
        return {
            "num_frames": 0,
            "image_width": 1920,
            "image_height": 1080,
            "focal_length_px": 1000.0,
            "tracking_method": "COLMAP (No Data)",
            "rotations": []
        }


class COLMAPCameraToFBXExport:
    """
    Direct export of COLMAP camera to FBX for Maya/Blender import.
    
    This is a simplified exporter specifically for camera data,
    useful when you only need the camera (not body tracking).
    
    Features:
    - Frame offset control to align with video timeline
    - Coordinate system conversion for DCC apps
    - Option to include sparse point cloud as locators
    """
    
    CATEGORY = "COLMAP"
    FUNCTION = "export"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_path", "status")
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_data": ("CAMERA_DATA",),
                "filename": ("STRING", {"default": "colmap_camera"}),
            },
            "optional": {
                "format": (["fbx", "abc", "json"], {"default": "fbx"}),
                "frame_offset": ("INT", {
                    "default": 0,
                    "min": -1000,
                    "max": 1000,
                    "tooltip": "Frame offset (e.g., -2 to shift frame 3 → frame 1)"
                }),
                "auto_frame_offset": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-offset to start from frame 1"
                }),
                "coordinate_system": (["maya", "blender", "unreal", "unity", "colmap"], {
                    "default": "maya"
                }),
                "include_sparse_points": ("BOOLEAN", {"default": False}),
                "output_dir": ("STRING", {"default": ""}),
            }
        }
    
    def export(
        self,
        camera_data: Any,
        filename: str = "colmap_camera",
        format: str = "fbx",
        frame_offset: int = 0,
        auto_frame_offset: bool = True,
        coordinate_system: str = "maya",
        include_sparse_points: bool = False,
        output_dir: str = "",
    ) -> Tuple[str, str]:
        """Export COLMAP camera with frame offset correction."""
        
        # This is a wrapper that applies frame offset before calling the main exporter
        # For now, return a message indicating the main exporter should be used
        
        # The actual implementation would modify camera_data.extrinsics frame indices
        # and then call COLMAPCameraExporter
        
        # Calculate offset if auto
        if auto_frame_offset and camera_data and camera_data.extrinsics:
            min_frame = min(ext.frame_index for ext in camera_data.extrinsics)
            if min_frame > 1:
                frame_offset = -(min_frame - 1)
        
        status = f"Use COLMAPCameraExporter with frame_offset={frame_offset}"
        return ("", status)


# Node registration
NODE_CLASS_MAPPINGS = {
    "COLMAPToSAM3DBodyCamera": COLMAPToSAM3DBodyCamera,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "COLMAPToSAM3DBodyCamera": "🔄 COLMAP → SAM3DBody Camera",
}
