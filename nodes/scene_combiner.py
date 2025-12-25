"""
COLMAP Scene Combiner Node

Combines camera data from COLMAP with mesh data from SAM3DBody
to create a complete scene with tracked body and tracked camera.
"""

from pathlib import Path
from typing import Tuple, Any, Optional, Dict
import numpy as np


class COLMAPSceneCombiner:
    """
    Combine COLMAP camera data with SAM3DBody mesh sequence.
    
    Creates a unified scene with:
    - Tracked camera from COLMAP
    - Body mesh from SAM3DBody
    - Aligned coordinate systems
    """
    
    CATEGORY = "COLMAP"
    FUNCTION = "combine"
    RETURN_TYPES = ("COMBINED_SCENE", "STRING")
    RETURN_NAMES = ("scene", "status")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_data": ("CAMERA_DATA",),
            },
            "optional": {
                "mesh_sequence": ("MESH_SEQUENCE", {
                    "tooltip": "Mesh sequence from SAM3DBody2abc"
                }),
                "target_coordinate_system": ([
                    "blender", "opengl", "unreal", "unity", "maya", "usd"
                ], {
                    "default": "blender"
                }),
                "scale_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.001,
                    "max": 1000.0,
                    "step": 0.001,
                    "tooltip": "Scale factor for scene"
                }),
                "align_to_ground": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Align scene so Y=0 is the ground plane"
                }),
                "center_origin": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Center the scene at origin"
                }),
            }
        }
    
    def combine(
        self,
        camera_data: Any,
        mesh_sequence: Optional[Any] = None,
        target_coordinate_system: str = "blender",
        scale_factor: float = 1.0,
        align_to_ground: bool = True,
        center_origin: bool = True
    ) -> Tuple[Dict[str, Any], str]:
        """Combine camera and mesh data into a unified scene."""
        from ..utils import (
            CoordinateSystem, get_coordinate_transform,
            transform_position, transform_rotation_matrix,
            rotation_matrix_to_quaternion
        )
        
        if not camera_data or not camera_data.extrinsics:
            return ({}, "ERROR: No camera data provided")
        
        # Get coordinate transform
        source_system = CoordinateSystem(camera_data.coordinate_system)
        target_system = CoordinateSystem(target_coordinate_system)
        coord_transform = get_coordinate_transform(source_system, target_system)
        
        # Transform camera data
        transformed_cameras = []
        all_positions = []
        
        for ext in camera_data.extrinsics:
            pos = transform_position(ext.position, coord_transform)
            rot = transform_rotation_matrix(ext.rotation_matrix, coord_transform)
            
            transformed_cameras.append({
                "frame_index": ext.frame_index,
                "image_name": ext.image_name,
                "position": pos.copy(),
                "rotation_matrix": rot.copy(),
                "quaternion": rotation_matrix_to_quaternion(rot),
            })
            all_positions.append(pos)
        
        all_positions = np.array(all_positions)
        
        # Transform sparse points
        transformed_points = []
        if camera_data.sparse_points:
            for p in camera_data.sparse_points:
                xyz = transform_position(p.xyz, coord_transform)
                transformed_points.append({
                    "xyz": xyz,
                    "rgb": p.rgb.tolist() if p.rgb is not None else None,
                })
                all_positions = np.vstack([all_positions, xyz])
        
        # Compute scene bounds
        scene_min = all_positions.min(axis=0)
        scene_max = all_positions.max(axis=0)
        scene_center = (scene_min + scene_max) / 2
        
        # Apply centering if requested
        offset = np.zeros(3)
        if center_origin:
            offset = -scene_center.copy()
            offset[1] = 0 if align_to_ground else offset[1]  # Don't center Y if grounding
        
        if align_to_ground:
            # Set ground to the lowest point
            ground_y = scene_min[1] if target_system in [
                CoordinateSystem.BLENDER, CoordinateSystem.UNREAL
            ] else scene_min[2]
            if target_system in [CoordinateSystem.BLENDER, CoordinateSystem.UNREAL]:
                offset[2] = offset[2] - ground_y
            else:
                offset[1] = offset[1] - ground_y
        
        # Apply offset and scale to cameras
        for cam in transformed_cameras:
            cam["position"] = (cam["position"] + offset) * scale_factor
        
        # Apply offset and scale to points
        for p in transformed_points:
            p["xyz"] = (p["xyz"] + offset) * scale_factor
        
        # Process mesh sequence if provided
        transformed_meshes = None
        if mesh_sequence is not None:
            transformed_meshes = self._process_mesh_sequence(
                mesh_sequence,
                coord_transform,
                offset,
                scale_factor
            )
        
        # Build combined scene
        scene = {
            "type": "COMBINED_SCENE",
            "coordinate_system": target_coordinate_system,
            "scale": scale_factor,
            "fps": camera_data.fps,
            "num_frames": len(transformed_cameras),
            
            # Camera data
            "intrinsics": camera_data.intrinsics.to_dict(),
            "cameras": transformed_cameras,
            "motion": [m.to_dict() for m in camera_data.motion] if camera_data.motion else [],
            
            # Sparse points
            "sparse_points": transformed_points,
            
            # Mesh sequence (if provided)
            "meshes": transformed_meshes,
            
            # Scene bounds (after transform)
            "bounds": {
                "min": ((scene_min + offset) * scale_factor).tolist(),
                "max": ((scene_max + offset) * scale_factor).tolist(),
            },
        }
        
        # Generate status
        mesh_status = ""
        if transformed_meshes:
            mesh_status = f", {len(transformed_meshes)} meshes"
        
        status = (
            f"Combined scene: {len(transformed_cameras)} cameras, "
            f"{len(transformed_points)} points{mesh_status} "
            f"in {target_coordinate_system} coordinates"
        )
        
        return (scene, status)
    
    def _process_mesh_sequence(
        self,
        mesh_sequence: Any,
        coord_transform: np.ndarray,
        offset: np.ndarray,
        scale_factor: float
    ) -> list:
        """Process mesh sequence from SAM3DBody."""
        from ..utils import transform_position
        
        transformed_meshes = []
        
        # Handle different mesh sequence formats
        if isinstance(mesh_sequence, dict):
            # Dictionary format from SAM3DBody2abc
            frames = mesh_sequence.get("frames", [])
            for frame_data in frames:
                frame_idx = frame_data.get("frame_index", 0)
                vertices = frame_data.get("vertices", [])
                faces = frame_data.get("faces", [])
                joints = frame_data.get("joints", [])
                
                # Transform vertices
                transformed_verts = []
                for v in vertices:
                    v_transformed = transform_position(np.array(v), coord_transform)
                    v_transformed = (v_transformed + offset) * scale_factor
                    transformed_verts.append(v_transformed.tolist())
                
                # Transform joints
                transformed_joints = []
                for j in joints:
                    j_transformed = transform_position(np.array(j), coord_transform)
                    j_transformed = (j_transformed + offset) * scale_factor
                    transformed_joints.append(j_transformed.tolist())
                
                transformed_meshes.append({
                    "frame_index": frame_idx,
                    "vertices": transformed_verts,
                    "faces": faces,
                    "joints": transformed_joints,
                })
        
        elif isinstance(mesh_sequence, list):
            # List format
            for i, mesh in enumerate(mesh_sequence):
                if hasattr(mesh, 'vertices'):
                    vertices = mesh.vertices
                    faces = mesh.faces if hasattr(mesh, 'faces') else []
                else:
                    continue
                
                transformed_verts = []
                for v in vertices:
                    v_transformed = transform_position(np.array(v), coord_transform)
                    v_transformed = (v_transformed + offset) * scale_factor
                    transformed_verts.append(v_transformed.tolist())
                
                transformed_meshes.append({
                    "frame_index": i,
                    "vertices": transformed_verts,
                    "faces": faces if isinstance(faces, list) else faces.tolist(),
                })
        
        return transformed_meshes
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
