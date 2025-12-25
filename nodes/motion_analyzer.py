"""
COLMAP Motion Analyzer Node

Analyzes camera motion to detect:
- Pan (horizontal rotation)
- Tilt (vertical rotation)  
- Roll (rotation around view axis)
- Dolly (forward/backward)
- Truck (left/right)
- Crane (up/down)
- Complex motion classification (handheld, tripod, drone, etc.)
"""

from typing import Tuple, Any, List, Optional
import numpy as np


class COLMAPMotionAnalyzer:
    """
    Analyze camera motion from camera data.
    
    Computes per-frame motion metrics and classifies the overall
    camera motion type (tripod, handheld, dolly, crane, drone, etc.)
    """
    
    CATEGORY = "COLMAP"
    FUNCTION = "analyze"
    RETURN_TYPES = ("CAMERA_DATA", "STRING", "STRING")
    RETURN_NAMES = ("camera_data", "motion_summary", "motion_json")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_data": ("CAMERA_DATA",),
            },
            "optional": {
                "smoothing_window": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 31,
                    "step": 2,
                    "tooltip": "Window size for motion smoothing (odd number)"
                }),
                "static_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Threshold for detecting static camera"
                }),
                "pan_tilt_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Degrees per frame threshold for pan/tilt detection"
                }),
                "translation_threshold": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "Units per frame threshold for translation detection"
                }),
            }
        }
    
    def analyze(
        self,
        camera_data: Any,
        smoothing_window: int = 5,
        static_threshold: float = 0.1,
        pan_tilt_threshold: float = 0.5,
        translation_threshold: float = 0.01
    ) -> Tuple[Any, str, str]:
        """Analyze camera motion."""
        from ..utils import CameraMotion, MotionType, rotation_matrix_to_euler
        import json
        
        if not camera_data or not camera_data.extrinsics:
            return (
                camera_data,
                "No camera data to analyze",
                "{}"
            )
        
        # Ensure odd window size
        if smoothing_window % 2 == 0:
            smoothing_window += 1
        
        motion_list = []
        extrinsics = sorted(camera_data.extrinsics, key=lambda x: x.frame_index)
        
        # Compute per-frame motion
        for i in range(len(extrinsics)):
            if i == 0:
                # First frame has no motion
                motion = CameraMotion(
                    frame_index=extrinsics[i].frame_index,
                    motion_type=MotionType.STATIC
                )
            else:
                motion = self._compute_frame_motion(
                    extrinsics[i-1],
                    extrinsics[i],
                    camera_data.fps
                )
            motion_list.append(motion)
        
        # Apply smoothing
        motion_list = self._smooth_motion(motion_list, smoothing_window)
        
        # Classify motion for each frame
        motion_list = self._classify_motion(
            motion_list,
            static_threshold,
            pan_tilt_threshold,
            translation_threshold
        )
        
        # Update camera_data with motion
        camera_data.motion = motion_list
        
        # Generate summary
        motion_summary = self._generate_summary(motion_list)
        
        # Generate detailed JSON
        motion_json = self._generate_json(motion_list, camera_data.fps)
        
        return (camera_data, motion_summary, motion_json)
    
    def _compute_frame_motion(
        self,
        prev_ext: Any,
        curr_ext: Any,
        fps: float
    ) -> "CameraMotion":
        """Compute motion between two consecutive frames."""
        from ..utils import CameraMotion, MotionType, rotation_matrix_to_euler
        
        # Translation difference
        delta_pos = curr_ext.position - prev_ext.position
        
        # Break down translation into dolly/truck/crane
        # Assuming camera forward is -Z, right is +X, up is -Y (COLMAP convention)
        # But this depends on the coordinate system, so we use rotation to define local axes
        R = curr_ext.rotation_matrix
        
        # Camera local axes in world space
        cam_forward = -R[:, 2]  # Camera looks down -Z
        cam_right = R[:, 0]     # Camera right is +X
        cam_up = -R[:, 1]       # Camera up is -Y
        
        # Project translation onto camera axes
        dolly = np.dot(delta_pos, cam_forward)   # Forward/backward
        truck = np.dot(delta_pos, cam_right)     # Left/right
        crane = np.dot(delta_pos, cam_up)        # Up/down
        
        # Rotation difference
        # Compute relative rotation: R_rel = R_curr * R_prev^T
        R_rel = curr_ext.rotation_matrix @ prev_ext.rotation_matrix.T
        
        # Extract Euler angles from relative rotation
        rx, ry, rz = rotation_matrix_to_euler(R_rel)
        
        # Map to camera motion terminology:
        # - Pan: rotation around world up (or camera up projected to world)
        # - Tilt: rotation around camera right axis
        # - Roll: rotation around camera forward axis
        # This is approximate since Euler angles are tricky
        pan = ry   # Roughly horizontal rotation
        tilt = rx  # Roughly vertical rotation
        roll = rz  # Roll
        
        # Compute magnitudes
        total_rotation = np.sqrt(pan**2 + tilt**2 + roll**2)
        total_translation = np.linalg.norm(delta_pos)
        speed = total_translation * fps  # Units per second
        
        return CameraMotion(
            frame_index=curr_ext.frame_index,
            pan=pan,
            tilt=tilt,
            roll=roll,
            dolly=dolly,
            truck=truck,
            crane=crane,
            total_rotation=total_rotation,
            total_translation=total_translation,
            speed=speed,
            motion_type=MotionType.COMPLEX,  # Will be classified later
        )
    
    def _smooth_motion(
        self,
        motion_list: List["CameraMotion"],
        window_size: int
    ) -> List["CameraMotion"]:
        """Apply moving average smoothing to motion data."""
        if len(motion_list) <= window_size:
            return motion_list
        
        half_window = window_size // 2
        
        # Extract motion arrays
        pans = np.array([m.pan for m in motion_list])
        tilts = np.array([m.tilt for m in motion_list])
        rolls = np.array([m.roll for m in motion_list])
        dollys = np.array([m.dolly for m in motion_list])
        trucks = np.array([m.truck for m in motion_list])
        cranes = np.array([m.crane for m in motion_list])
        
        # Apply uniform filter (moving average)
        def smooth(arr):
            kernel = np.ones(window_size) / window_size
            # Pad to handle edges
            padded = np.pad(arr, half_window, mode='edge')
            smoothed = np.convolve(padded, kernel, mode='valid')
            return smoothed
        
        pans_smooth = smooth(pans)
        tilts_smooth = smooth(tilts)
        rolls_smooth = smooth(rolls)
        dollys_smooth = smooth(dollys)
        trucks_smooth = smooth(trucks)
        cranes_smooth = smooth(cranes)
        
        # Update motion list
        for i, m in enumerate(motion_list):
            m.pan = pans_smooth[i]
            m.tilt = tilts_smooth[i]
            m.roll = rolls_smooth[i]
            m.dolly = dollys_smooth[i]
            m.truck = trucks_smooth[i]
            m.crane = cranes_smooth[i]
            m.total_rotation = np.sqrt(m.pan**2 + m.tilt**2 + m.roll**2)
            m.total_translation = np.sqrt(m.dolly**2 + m.truck**2 + m.crane**2)
        
        return motion_list
    
    def _classify_motion(
        self,
        motion_list: List["CameraMotion"],
        static_threshold: float,
        pan_tilt_threshold: float,
        translation_threshold: float
    ) -> List["CameraMotion"]:
        """Classify motion type for each frame."""
        from ..utils import MotionType
        
        for m in motion_list:
            # Check if static
            if (m.total_rotation < static_threshold and 
                m.total_translation < translation_threshold * 0.1):
                m.motion_type = MotionType.STATIC
                m.motion_confidence = 1.0
                continue
            
            # Determine primary motion type
            rotation_dominant = m.total_rotation > pan_tilt_threshold
            translation_dominant = m.total_translation > translation_threshold
            
            if not rotation_dominant and not translation_dominant:
                m.motion_type = MotionType.HANDHELD
                m.motion_confidence = 0.7
                continue
            
            # Check dominant motion
            pan_dominant = abs(m.pan) > abs(m.tilt) and abs(m.pan) > abs(m.roll)
            tilt_dominant = abs(m.tilt) > abs(m.pan) and abs(m.tilt) > abs(m.roll)
            roll_dominant = abs(m.roll) > abs(m.pan) and abs(m.roll) > abs(m.tilt)
            
            dolly_dominant = abs(m.dolly) > abs(m.truck) and abs(m.dolly) > abs(m.crane)
            truck_dominant = abs(m.truck) > abs(m.dolly) and abs(m.truck) > abs(m.crane)
            crane_dominant = abs(m.crane) > abs(m.dolly) and abs(m.crane) > abs(m.truck)
            
            # Classify based on dominant motion
            if rotation_dominant and not translation_dominant:
                if pan_dominant:
                    m.motion_type = MotionType.PAN
                elif tilt_dominant:
                    m.motion_type = MotionType.TILT
                elif roll_dominant:
                    m.motion_type = MotionType.ROLL
                else:
                    m.motion_type = MotionType.COMPLEX
            elif translation_dominant and not rotation_dominant:
                if dolly_dominant:
                    m.motion_type = MotionType.DOLLY
                elif truck_dominant:
                    m.motion_type = MotionType.TRUCK
                elif crane_dominant:
                    m.motion_type = MotionType.CRANE
                else:
                    m.motion_type = MotionType.DRONE
            else:
                # Both rotation and translation significant
                # Could be orbit, tracking shot, or drone
                if crane_dominant and (pan_dominant or tilt_dominant):
                    m.motion_type = MotionType.DRONE
                elif truck_dominant and pan_dominant:
                    m.motion_type = MotionType.ORBIT
                else:
                    m.motion_type = MotionType.TRACKING
            
            m.motion_confidence = 0.8
        
        return motion_list
    
    def _generate_summary(self, motion_list: List["CameraMotion"]) -> str:
        """Generate human-readable motion summary."""
        from ..utils import MotionType
        
        if not motion_list:
            return "No motion data"
        
        # Count motion types
        type_counts = {}
        for m in motion_list:
            t = m.motion_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        
        # Find dominant motion type
        dominant_type = max(type_counts.items(), key=lambda x: x[1])
        
        # Compute statistics
        total_pan = sum(abs(m.pan) for m in motion_list)
        total_tilt = sum(abs(m.tilt) for m in motion_list)
        total_dolly = sum(abs(m.dolly) for m in motion_list)
        total_truck = sum(abs(m.truck) for m in motion_list)
        total_crane = sum(abs(m.crane) for m in motion_list)
        avg_speed = np.mean([m.speed for m in motion_list])
        
        summary_parts = [
            f"Dominant motion: {dominant_type[0]} ({dominant_type[1]}/{len(motion_list)} frames)",
            f"Total pan: {total_pan:.1f}°, tilt: {total_tilt:.1f}°",
            f"Total dolly: {total_dolly:.3f}, truck: {total_truck:.3f}, crane: {total_crane:.3f}",
            f"Average speed: {avg_speed:.4f} units/sec",
            f"Motion breakdown: {type_counts}"
        ]
        
        return "\n".join(summary_parts)
    
    def _generate_json(
        self,
        motion_list: List["CameraMotion"],
        fps: float
    ) -> str:
        """Generate detailed JSON output."""
        import json
        
        data = {
            "fps": fps,
            "total_frames": len(motion_list),
            "frames": [m.to_dict() for m in motion_list],
            "statistics": {
                "total_pan": sum(abs(m.pan) for m in motion_list),
                "total_tilt": sum(abs(m.tilt) for m in motion_list),
                "total_dolly": sum(abs(m.dolly) for m in motion_list),
                "total_truck": sum(abs(m.truck) for m in motion_list),
                "total_crane": sum(abs(m.crane) for m in motion_list),
                "max_speed": max(m.speed for m in motion_list) if motion_list else 0,
                "avg_speed": np.mean([m.speed for m in motion_list]) if motion_list else 0,
            }
        }
        
        return json.dumps(data, indent=2)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
