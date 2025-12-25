"""
COLMAP Camera Extractor Node

Extracts camera intrinsics and extrinsics from a COLMAP reconstruction.
Converts to the universal CAMERA_DATA format.
"""

from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import numpy as np


class COLMAPCameraExtractor:
    """
    Extract camera data from COLMAP reconstruction.
    
    Converts COLMAP's native format to a universal CAMERA_DATA type
    that can be used for export or integration with other nodes.
    """
    
    CATEGORY = "COLMAP"
    FUNCTION = "extract"
    RETURN_TYPES = ("CAMERA_DATA", "STRING")
    RETURN_NAMES = ("camera_data", "status")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reconstruction": ("COLMAP_RECONSTRUCTION",),
            },
            "optional": {
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1
                }),
                "coordinate_system": ([
                    "colmap", "blender", "opengl", "opencv", 
                    "unreal", "unity", "maya", "houdini", "usd"
                ], {
                    "default": "colmap"
                }),
                "include_sparse_points": ("BOOLEAN", {
                    "default": True
                }),
                "max_sparse_points": ("INT", {
                    "default": 100000,
                    "min": 1000,
                    "max": 1000000,
                    "step": 1000,
                    "tooltip": "Maximum sparse points to include (for memory)"
                }),
            }
        }
    
    def extract(
        self,
        reconstruction: Dict[str, Any],
        fps: float = 24.0,
        coordinate_system: str = "colmap",
        include_sparse_points: bool = True,
        max_sparse_points: int = 100000
    ) -> Tuple[Any, str]:
        """Extract camera data from reconstruction."""
        from ..utils import (
            CameraData, CameraIntrinsics, CameraExtrinsics,
            CameraModel, SparsePoint, CoordinateSystem,
            get_coordinate_transform, transform_position,
            transform_rotation_matrix, rotation_matrix_to_quaternion,
            PYCOLMAP_AVAILABLE
        )
        
        if not reconstruction:
            return (CameraData(), "ERROR: No reconstruction provided")
        
        rec = reconstruction.get("reconstruction")
        if rec is None:
            return (CameraData(), "ERROR: Reconstruction object not found")
        
        if not PYCOLMAP_AVAILABLE:
            return (CameraData(), "ERROR: pycolmap required for extraction")
        
        try:
            # Get coordinate transform if needed
            target_system = CoordinateSystem(coordinate_system)
            if target_system != CoordinateSystem.COLMAP:
                coord_transform = get_coordinate_transform(
                    CoordinateSystem.COLMAP,
                    target_system
                )
            else:
                coord_transform = np.eye(4)
            
            # Extract intrinsics
            intrinsics = self._extract_intrinsics(rec)
            
            # Extract extrinsics with coordinate transform
            extrinsics_list = self._extract_extrinsics(rec, coord_transform)
            
            # Extract sparse points if requested
            sparse_points = []
            if include_sparse_points:
                sparse_points = self._extract_sparse_points(
                    rec, coord_transform, max_sparse_points
                )
            
            # Compute mean reprojection error
            mean_error = reconstruction.get("summary", {}).get(
                "mean_reprojection_error", 0.0
            )
            
            # Build CameraData
            camera_data = CameraData(
                reconstruction_id=0,
                num_frames=len(extrinsics_list),
                fps=fps,
                intrinsics=intrinsics,
                extrinsics=extrinsics_list,
                sparse_points=sparse_points,
                registered_frames=len(extrinsics_list),
                total_frames=reconstruction.get("num_images", len(extrinsics_list)),
                mean_reprojection_error=mean_error,
                coordinate_system=coordinate_system,
                scale=1.0,
            )
            
            status = (
                f"Extracted {len(extrinsics_list)} cameras, "
                f"{len(sparse_points)} points in {coordinate_system} coordinates"
            )
            
            return (camera_data, status)
            
        except Exception as e:
            import traceback
            return (CameraData(), f"ERROR: {str(e)}\n{traceback.format_exc()}")
    
    def _extract_intrinsics(self, rec) -> "CameraIntrinsics":
        """Extract camera intrinsics from first camera."""
        from ..utils import CameraIntrinsics, CameraModel
        
        intrinsics = CameraIntrinsics()
        
        if not rec.cameras:
            return intrinsics
        
        cam_id = list(rec.cameras.keys())[0]
        cam = rec.cameras[cam_id]
        
        model_map = {
            "SIMPLE_PINHOLE": CameraModel.SIMPLE_PINHOLE,
            "PINHOLE": CameraModel.PINHOLE,
            "SIMPLE_RADIAL": CameraModel.SIMPLE_RADIAL,
            "RADIAL": CameraModel.RADIAL,
            "OPENCV": CameraModel.OPENCV,
            "OPENCV_FISHEYE": CameraModel.OPENCV_FISHEYE,
            "FULL_OPENCV": CameraModel.FULL_OPENCV,
        }
        
        intrinsics.width = cam.width
        intrinsics.height = cam.height
        intrinsics.model = model_map.get(cam.model_name, CameraModel.PINHOLE)
        
        params = cam.params
        if cam.model_name == "SIMPLE_PINHOLE":
            intrinsics.focal_length_x = params[0]
            intrinsics.focal_length_y = params[0]
            intrinsics.principal_point_x = params[1]
            intrinsics.principal_point_y = params[2]
        elif cam.model_name == "PINHOLE":
            intrinsics.focal_length_x = params[0]
            intrinsics.focal_length_y = params[1]
            intrinsics.principal_point_x = params[2]
            intrinsics.principal_point_y = params[3]
        elif cam.model_name in ["SIMPLE_RADIAL", "RADIAL"]:
            intrinsics.focal_length_x = params[0]
            intrinsics.focal_length_y = params[0]
            intrinsics.principal_point_x = params[1]
            intrinsics.principal_point_y = params[2]
            if len(params) > 3:
                intrinsics.k1 = params[3]
            if len(params) > 4:
                intrinsics.k2 = params[4]
        elif cam.model_name in ["OPENCV", "FULL_OPENCV"]:
            intrinsics.focal_length_x = params[0]
            intrinsics.focal_length_y = params[1]
            intrinsics.principal_point_x = params[2]
            intrinsics.principal_point_y = params[3]
            if len(params) > 4:
                intrinsics.k1 = params[4]
            if len(params) > 5:
                intrinsics.k2 = params[5]
            if len(params) > 6:
                intrinsics.p1 = params[6]
            if len(params) > 7:
                intrinsics.p2 = params[7]
        
        return intrinsics
    
    def _extract_extrinsics(
        self,
        rec,
        coord_transform: np.ndarray
    ) -> list:
        """Extract extrinsics for all registered images."""
        from ..utils import (
            CameraExtrinsics, transform_position, 
            transform_rotation_matrix, rotation_matrix_to_quaternion
        )
        
        extrinsics_list = []
        
        for image_id, image in rec.images.items():
            name = image.name
            
            # Parse frame index from name
            try:
                frame_idx = int(name.split('_')[-1].split('.')[0])
            except (ValueError, IndexError):
                frame_idx = image_id
            
            # Get pose
            cam_from_world = image.cam_from_world
            R = cam_from_world.rotation.matrix()
            t = cam_from_world.translation
            
            # Camera position in world: -R^T * t
            position = -R.T @ t
            
            # Apply coordinate transform
            position = transform_position(position, coord_transform)
            R = transform_rotation_matrix(R, coord_transform)
            
            # Quaternion
            quat = rotation_matrix_to_quaternion(R)
            
            ext = CameraExtrinsics(
                frame_index=frame_idx,
                image_name=name,
                position=position,
                rotation_matrix=R,
                quaternion=quat,
                num_observations=len(image.points2D),
            )
            extrinsics_list.append(ext)
        
        # Sort by frame index
        extrinsics_list.sort(key=lambda x: x.frame_index)
        
        return extrinsics_list
    
    def _extract_sparse_points(
        self,
        rec,
        coord_transform: np.ndarray,
        max_points: int
    ) -> list:
        """Extract sparse 3D points."""
        from ..utils import SparsePoint, transform_position
        
        sparse_points = []
        count = 0
        
        for point_id, point in rec.points3D.items():
            if count >= max_points:
                break
            
            # Transform position
            xyz = transform_position(point.xyz, coord_transform)
            
            # Get color if available
            rgb = None
            if hasattr(point, 'color'):
                rgb = np.array(point.color, dtype=np.uint8)
            
            sp = SparsePoint(
                point_id=point_id,
                xyz=xyz,
                rgb=rgb,
                error=point.error,
                num_observations=len(point.track.elements),
            )
            sparse_points.append(sp)
            count += 1
        
        return sparse_points
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
