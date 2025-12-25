"""
COLMAP Auto Reconstruct Node

All-in-one node that performs:
1. Feature extraction
2. Feature matching  
3. Sparse reconstruction
4. Camera data extraction

This is the recommended node for most users.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# ComfyUI imports
try:
    import folder_paths
except ImportError:
    folder_paths = None


class COLMAPAutoReconstruct:
    """
    Automatic COLMAP reconstruction from video frames.
    
    Takes a batch of images and outputs complete camera data
    including intrinsics, extrinsics, and sparse point cloud.
    """
    
    CATEGORY = "COLMAP"
    FUNCTION = "process"
    RETURN_TYPES = ("CAMERA_DATA", "STRING", "IMAGE")
    RETURN_NAMES = ("camera_data", "status", "sparse_points_preview")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "matcher_type": (["exhaustive", "sequential", "vocab_tree"], {
                    "default": "sequential"
                }),
                "feature_type": (["sift", "sift_gpu"], {
                    "default": "sift"
                }),
                "max_features": ("INT", {
                    "default": 8192,
                    "min": 1024,
                    "max": 32768,
                    "step": 1024
                }),
                "min_matches": ("INT", {
                    "default": 15,
                    "min": 5,
                    "max": 100,
                    "step": 5
                }),
                "gpu_mode": (["auto", "cpu_only", "force_gpu", "force_offload"], {
                    "default": "auto"
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Mask for dynamic objects (white=exclude from tracking). Use to mask moving subjects for cleaner background-only camera tracking."
                }),
                "mask_mode": (["exclude_white", "include_white"], {
                    "default": "exclude_white",
                    "tooltip": "exclude_white: mask out white areas (moving objects). include_white: only track white areas."
                }),
                "mask_dilation": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 100,
                    "step": 5,
                    "tooltip": "Dilate mask by pixels to ensure clean edges around masked objects"
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1
                }),
                "image_names": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Optional: one filename per line"
                }),
            }
        }
    
    def process(
        self,
        images,
        matcher_type: str,
        feature_type: str,
        max_features: int,
        min_matches: int,
        gpu_mode: str,
        mask = None,
        mask_mode: str = "exclude_white",
        mask_dilation: int = 10,
        fps: float = 24.0,
        image_names: str = ""
    ) -> Tuple[Any, str, Any]:
        """
        Run complete COLMAP reconstruction pipeline.
        
        If mask is provided, features on masked regions will be excluded
        to ensure only static background is used for camera tracking.
        """
        import torch
        
        from ..utils import (
            COLMAPWrapper, GPUMode, MatcherType, FeatureType,
            CameraData, CameraIntrinsics, CameraExtrinsics, 
            CameraModel, SparsePoint, check_colmap_installation,
            PYCOLMAP_AVAILABLE
        )
        
        # Convert tensor to numpy if needed
        if isinstance(images, torch.Tensor):
            images_np = images.cpu().numpy()
        else:
            images_np = np.array(images)
        
        # Convert mask tensor to numpy if provided
        mask_np = None
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)
        
        # Check COLMAP availability
        is_available, status_msg = check_colmap_installation()
        if not is_available:
            empty_preview = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (
                CameraData(),
                f"ERROR: COLMAP not available. {status_msg}",
                empty_preview
            )
        
        # Parse image names if provided
        name_list = None
        if image_names.strip():
            name_list = [n.strip() for n in image_names.strip().split('\n') if n.strip()]
        
        # Process masks if provided
        processed_images = images_np
        if mask_np is not None:
            processed_images = self._apply_mask_to_images(
                images_np, mask_np, mask_mode, mask_dilation
            )
            print(f"[COLMAP] Applied mask to {len(images_np)} images (mode: {mask_mode}, dilation: {mask_dilation}px)")
        
        # Convert enums
        gpu_mode_enum = GPUMode(gpu_mode)
        matcher_enum = MatcherType(matcher_type)
        feature_enum = FeatureType(feature_type)
        
        # Create workspace
        workspace = Path(tempfile.mkdtemp(prefix="comfyui_colmap_"))
        
        try:
            # Initialize wrapper
            wrapper = COLMAPWrapper(
                workspace=workspace,
                gpu_mode=gpu_mode_enum,
                verbose=True
            )
            
            print(f"[COLMAP] Processing {len(processed_images)} images...")
            
            # 1. Prepare images (with mask applied if provided)
            saved_paths = wrapper.prepare_images(processed_images, name_list)
            print(f"[COLMAP] Saved {len(saved_paths)} images to workspace")
            
            # 2. Extract features
            print(f"[COLMAP] Extracting features ({feature_type})...")
            wrapper.extract_features(
                feature_type=feature_enum,
                max_num_features=max_features
            )
            
            # 3. Match features
            print(f"[COLMAP] Matching features ({matcher_type})...")
            wrapper.match_features(matcher_type=matcher_enum)
            
            # 4. Run reconstruction
            print("[COLMAP] Running sparse reconstruction...")
            reconstructions = wrapper.sparse_reconstruction(
                min_num_matches=min_matches,
                multiple_models=False
            )
            
            if not reconstructions:
                empty_preview = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                return (
                    CameraData(),
                    "ERROR: Reconstruction failed - no valid models produced",
                    empty_preview
                )
            
            # 5. Extract camera data
            rec = wrapper.reconstruction
            if rec is None:
                empty_preview = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                return (
                    CameraData(),
                    "ERROR: No reconstruction available",
                    empty_preview
                )
            
            # Build CameraData
            camera_data = self._extract_camera_data(rec, images_np, fps)
            
            # Generate preview of sparse points
            preview_np = self._generate_sparse_preview(camera_data, images_np.shape[1:3])
            # Convert to tensor format (B, H, W, C)
            preview = torch.from_numpy(preview_np).unsqueeze(0).float()
            
            # Get summary
            summary = wrapper.get_reconstruction_summary()
            status = (
                f"SUCCESS: Registered {summary['num_registered_images']}/{len(images_np)} images, "
                f"{summary['num_points3D']} 3D points, "
                f"mean reproj error: {summary['mean_reprojection_error']:.2f}px"
            )
            
            return (camera_data, status, preview)
            
        except Exception as e:
            import traceback
            error_msg = f"ERROR: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            empty_preview = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (
                CameraData(),
                error_msg,
                empty_preview
            )
        
        finally:
            # Cleanup
            if workspace.exists():
                shutil.rmtree(workspace)
    
    def _extract_camera_data(
        self,
        reconstruction,
        images: np.ndarray,
        fps: float
    ) -> "CameraData":
        """Extract CameraData from pycolmap reconstruction"""
        from ..utils import (
            CameraData, CameraIntrinsics, CameraExtrinsics,
            CameraModel, SparsePoint, PYCOLMAP_AVAILABLE,
            rotation_matrix_to_quaternion
        )
        
        if not PYCOLMAP_AVAILABLE:
            return CameraData()
        
        import pycolmap
        
        def get_camera_model_name(cam):
            """Get camera model name from pycolmap Camera object (API varies by version)"""
            # Try different attribute names
            if hasattr(cam, 'model_name'):
                return cam.model_name
            elif hasattr(cam, 'model'):
                model = cam.model
                # model might be an enum or string
                if hasattr(model, 'name'):
                    return model.name
                return str(model)
            elif hasattr(cam, 'model_id'):
                # Map model ID to name
                model_id_map = {
                    0: "SIMPLE_PINHOLE",
                    1: "PINHOLE", 
                    2: "SIMPLE_RADIAL",
                    3: "RADIAL",
                    4: "OPENCV",
                    5: "OPENCV_FISHEYE",
                    6: "FULL_OPENCV",
                }
                return model_id_map.get(cam.model_id, "PINHOLE")
            else:
                # Inspect what attributes are available
                print(f"[COLMAP] Camera attributes: {[a for a in dir(cam) if not a.startswith('_')]}")
                return "SIMPLE_RADIAL"  # Default
        
        # Get first camera for intrinsics (assuming single camera)
        intrinsics = CameraIntrinsics()
        if reconstruction.cameras:
            cam_id = list(reconstruction.cameras.keys())[0]
            cam = reconstruction.cameras[cam_id]
            
            # Get model name
            model_name = get_camera_model_name(cam)
            print(f"[COLMAP] Camera model: {model_name}")
            
            # Map COLMAP model to our enum
            model_map = {
                "SIMPLE_PINHOLE": CameraModel.SIMPLE_PINHOLE,
                "PINHOLE": CameraModel.PINHOLE,
                "SIMPLE_RADIAL": CameraModel.SIMPLE_RADIAL,
                "RADIAL": CameraModel.RADIAL,
                "OPENCV": CameraModel.OPENCV,
                "OPENCV_FISHEYE": CameraModel.OPENCV_FISHEYE,
                "FULL_OPENCV": CameraModel.FULL_OPENCV,
                # Handle enum-style names
                "CameraModelId.SIMPLE_PINHOLE": CameraModel.SIMPLE_PINHOLE,
                "CameraModelId.PINHOLE": CameraModel.PINHOLE,
                "CameraModelId.SIMPLE_RADIAL": CameraModel.SIMPLE_RADIAL,
                "CameraModelId.RADIAL": CameraModel.RADIAL,
                "CameraModelId.OPENCV": CameraModel.OPENCV,
            }
            
            intrinsics.width = cam.width
            intrinsics.height = cam.height
            intrinsics.model = model_map.get(model_name, CameraModel.PINHOLE)
            
            # Extract parameters based on model
            params = cam.params
            if "SIMPLE_PINHOLE" in model_name:
                intrinsics.focal_length_x = params[0]
                intrinsics.focal_length_y = params[0]
                intrinsics.principal_point_x = params[1]
                intrinsics.principal_point_y = params[2]
            elif "PINHOLE" in model_name and "SIMPLE" not in model_name:
                intrinsics.focal_length_x = params[0]
                intrinsics.focal_length_y = params[1]
                intrinsics.principal_point_x = params[2]
                intrinsics.principal_point_y = params[3]
            elif "RADIAL" in model_name:
                intrinsics.focal_length_x = params[0]
                intrinsics.focal_length_y = params[0]
                intrinsics.principal_point_x = params[1]
                intrinsics.principal_point_y = params[2]
                intrinsics.k1 = params[3] if len(params) > 3 else 0
                intrinsics.k2 = params[4] if len(params) > 4 else 0
            elif "OPENCV" in model_name:
                intrinsics.focal_length_x = params[0]
                intrinsics.focal_length_y = params[1]
                intrinsics.principal_point_x = params[2]
                intrinsics.principal_point_y = params[3]
                if len(params) > 4:
                    intrinsics.k1 = params[4]
                    intrinsics.k2 = params[5] if len(params) > 5 else 0
                    intrinsics.p1 = params[6] if len(params) > 6 else 0
                    intrinsics.p2 = params[7] if len(params) > 7 else 0
            else:
                # Default: assume SIMPLE_RADIAL-like structure
                intrinsics.focal_length_x = params[0] if len(params) > 0 else 1000
                intrinsics.focal_length_y = params[0] if len(params) > 0 else 1000
                intrinsics.principal_point_x = params[1] if len(params) > 1 else cam.width / 2
                intrinsics.principal_point_y = params[2] if len(params) > 2 else cam.height / 2
        
        # Extract extrinsics for each registered image
        extrinsics_list = []
        
        # Create mapping from image name to frame index
        for image_id, image in reconstruction.images.items():
            # Get frame index from image name
            name = image.name
            try:
                # Try to extract frame number from name like "frame_000001.jpg"
                frame_idx = int(name.split('_')[-1].split('.')[0])
            except (ValueError, IndexError):
                frame_idx = image_id
            
            # Get camera pose - API varies by pycolmap version
            try:
                # Try new API: cam_from_world is a method or property returning Rigid3d
                if callable(getattr(image, 'cam_from_world', None)):
                    cam_from_world = image.cam_from_world()
                else:
                    cam_from_world = image.cam_from_world
                
                # Extract rotation and translation from Rigid3d
                if hasattr(cam_from_world, 'rotation'):
                    rot = cam_from_world.rotation
                    if callable(getattr(rot, 'matrix', None)):
                        R = rot.matrix()
                    else:
                        R = np.array(rot)
                    t = np.array(cam_from_world.translation)
                else:
                    # Fallback: try direct matrix access
                    R = np.array(cam_from_world.rotmat()) if hasattr(cam_from_world, 'rotmat') else np.eye(3)
                    t = np.array(cam_from_world.tvec) if hasattr(cam_from_world, 'tvec') else np.zeros(3)
                    
            except Exception as e:
                # Try older API with separate rotation/translation
                try:
                    # Very old API: image.qvec and image.tvec
                    if hasattr(image, 'qvec') and hasattr(image, 'tvec'):
                        qvec = np.array(image.qvec)  # COLMAP uses w, x, y, z
                        # Convert quaternion to rotation matrix manually
                        w, x, y, z = qvec[0], qvec[1], qvec[2], qvec[3]
                        R = np.array([
                            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
                        ])
                        t = np.array(image.tvec)
                    elif hasattr(image, 'rotmat') and hasattr(image, 'tvec'):
                        R = np.array(image.rotmat())
                        t = np.array(image.tvec)
                    else:
                        print(f"[COLMAP] Image attributes: {[a for a in dir(image) if not a.startswith('_')]}")
                        print(f"[COLMAP] Warning: Could not extract pose for image {name}, using identity")
                        R = np.eye(3)
                        t = np.zeros(3)
                except Exception as e2:
                    print(f"[COLMAP] Pose extraction failed: {e}, {e2}")
                    R = np.eye(3)
                    t = np.zeros(3)
            
            # Camera position in world coordinates
            # position = -R^T * t
            position = -R.T @ t
            
            # Create quaternion
            quat = rotation_matrix_to_quaternion(R)
            
            # Get number of observations
            num_obs = 0
            if hasattr(image, 'points2D'):
                num_obs = len(image.points2D)
            elif hasattr(image, 'num_points2D'):
                num_obs = image.num_points2D
            
            ext = CameraExtrinsics(
                frame_index=frame_idx,
                image_name=name,
                position=position,
                rotation_matrix=R,
                quaternion=quat,
                num_observations=num_obs,
            )
            extrinsics_list.append(ext)
        
        # Sort by frame index
        extrinsics_list.sort(key=lambda x: x.frame_index)
        
        # Debug: Print camera positions
        if extrinsics_list:
            print(f"[COLMAP] Extracted {len(extrinsics_list)} camera poses")
            cam_positions = np.array([ext.position for ext in extrinsics_list])
            print(f"[COLMAP] Camera positions bounds:")
            print(f"[COLMAP]   X: {cam_positions[:, 0].min():.3f} to {cam_positions[:, 0].max():.3f}")
            print(f"[COLMAP]   Y: {cam_positions[:, 1].min():.3f} to {cam_positions[:, 1].max():.3f}")
            print(f"[COLMAP]   Z: {cam_positions[:, 2].min():.3f} to {cam_positions[:, 2].max():.3f}")
            print(f"[COLMAP]   Center: ({cam_positions[:, 0].mean():.3f}, {cam_positions[:, 1].mean():.3f}, {cam_positions[:, 2].mean():.3f})")
        
        # Extract sparse points
        sparse_points = []
        try:
            # Debug: Inspect reconstruction structure
            print(f"[COLMAP] Reconstruction object type: {type(reconstruction)}")
            print(f"[COLMAP] Reconstruction attributes: {[a for a in dir(reconstruction) if not a.startswith('_')]}")
            
            # Check points3D access
            points3D = reconstruction.points3D
            print(f"[COLMAP] points3D type: {type(points3D)}")
            print(f"[COLMAP] points3D length: {len(points3D) if hasattr(points3D, '__len__') else 'N/A'}")
            
            if len(points3D) > 0:
                # Inspect first point structure
                first_key = list(points3D.keys())[0]
                first_point = points3D[first_key]
                print(f"[COLMAP] First point ID: {first_key}")
                print(f"[COLMAP] First point type: {type(first_point)}")
                print(f"[COLMAP] First point attributes: {[a for a in dir(first_point) if not a.startswith('_')]}")
                if hasattr(first_point, 'xyz'):
                    print(f"[COLMAP] First point xyz: {first_point.xyz}")
                if hasattr(first_point, 'color'):
                    print(f"[COLMAP] First point color: {first_point.color}")
            
            print(f"[COLMAP] Extracting sparse points from {len(points3D)} 3D points")
            for point_id, point in points3D.items():
                # Get xyz
                xyz = np.array(point.xyz) if hasattr(point, 'xyz') else np.zeros(3)
                
                # Get color
                rgb = None
                if hasattr(point, 'color'):
                    rgb = np.array(point.color)
                elif hasattr(point, 'rgb'):
                    rgb = np.array(point.rgb)
                
                # Get error
                error = point.error if hasattr(point, 'error') else 0.0
                
                # Get number of observations
                num_obs = 0
                if hasattr(point, 'track') and hasattr(point.track, 'elements'):
                    num_obs = len(point.track.elements)
                elif hasattr(point, 'track'):
                    num_obs = len(point.track) if hasattr(point.track, '__len__') else 0
                
                sp = SparsePoint(
                    point_id=point_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    num_observations=num_obs,
                )
                sparse_points.append(sp)
            
            print(f"[COLMAP] Successfully extracted {len(sparse_points)} sparse points")
            
            # Print point cloud bounds
            if sparse_points:
                all_xyz = np.array([p.xyz for p in sparse_points])
                print(f"[COLMAP] Point cloud bounds:")
                print(f"[COLMAP]   X: {all_xyz[:, 0].min():.3f} to {all_xyz[:, 0].max():.3f}")
                print(f"[COLMAP]   Y: {all_xyz[:, 1].min():.3f} to {all_xyz[:, 1].max():.3f}")
                print(f"[COLMAP]   Z: {all_xyz[:, 2].min():.3f} to {all_xyz[:, 2].max():.3f}")
                print(f"[COLMAP]   Center: ({all_xyz[:, 0].mean():.3f}, {all_xyz[:, 1].mean():.3f}, {all_xyz[:, 2].mean():.3f})")
                
        except Exception as e:
            print(f"[COLMAP] Warning: Could not extract sparse points: {e}")
            import traceback
            traceback.print_exc()
        
        # Build CameraData
        camera_data = CameraData(
            reconstruction_id=0,
            num_frames=len(extrinsics_list),
            fps=fps,
            intrinsics=intrinsics,
            extrinsics=extrinsics_list,
            sparse_points=sparse_points,
            registered_frames=len(extrinsics_list),
            total_frames=len(images),
            mean_reprojection_error=sum(p.error for p in sparse_points) / max(len(sparse_points), 1),
            coordinate_system="colmap",
            scale=1.0,
        )
        
        return camera_data
    
    def _generate_sparse_preview(
        self,
        camera_data: "CameraData",
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """Generate a simple visualization of sparse points"""
        h, w = image_size
        preview = np.zeros((h, w, 3), dtype=np.float32)
        
        if not camera_data.sparse_points:
            return preview
        
        # Get point cloud bounds
        points = np.array([p.xyz for p in camera_data.sparse_points])
        if len(points) == 0:
            return preview
        
        # Simple orthographic projection (top-down view)
        min_xyz = points.min(axis=0)
        max_xyz = points.max(axis=0)
        range_xyz = max_xyz - min_xyz
        range_xyz[range_xyz == 0] = 1.0
        
        # Normalize to image coordinates
        normalized = (points - min_xyz) / range_xyz
        
        # Project X-Z plane (top-down view)
        px = (normalized[:, 0] * (w - 1)).astype(int)
        pz = (normalized[:, 2] * (h - 1)).astype(int)
        
        # Clamp to valid range
        px = np.clip(px, 0, w - 1)
        pz = np.clip(pz, 0, h - 1)
        
        # Draw points with colors if available
        for i, p in enumerate(camera_data.sparse_points):
            x, z = px[i], pz[i]
            if p.rgb is not None:
                color = p.rgb / 255.0
            else:
                # Color by height (Y)
                height_normalized = normalized[i, 1]
                color = np.array([height_normalized, 0.5, 1.0 - height_normalized])
            
            # Draw point with small radius
            for dx in range(-1, 2):
                for dz in range(-1, 2):
                    xi = min(max(x + dx, 0), w - 1)
                    zi = min(max(z + dz, 0), h - 1)
                    preview[zi, xi] = color
        
        # Draw camera positions
        for ext in camera_data.extrinsics:
            pos = ext.position
            pos_norm = (pos - min_xyz) / range_xyz
            cx = int(np.clip(pos_norm[0] * (w - 1), 0, w - 1))
            cz = int(np.clip(pos_norm[2] * (h - 1), 0, h - 1))
            
            # Draw camera as red cross
            for d in range(-3, 4):
                if 0 <= cx + d < w:
                    preview[cz, cx + d] = [1.0, 0.0, 0.0]
                if 0 <= cz + d < h:
                    preview[cz + d, cx] = [1.0, 0.0, 0.0]
        
        return preview
    
    def _apply_mask_to_images(
        self,
        images: np.ndarray,
        mask: np.ndarray,
        mask_mode: str,
        dilation: int
    ) -> np.ndarray:
        """
        Apply mask to images to exclude dynamic objects from feature detection.
        
        Masked regions are filled with a neutral gray to prevent feature detection
        while maintaining image structure for COLMAP.
        
        Args:
            images: (N, H, W, 3) image batch
            mask: (N, H, W) or (H, W) mask batch
            mask_mode: "exclude_white" or "include_white"
            dilation: pixels to dilate mask by
            
        Returns:
            Masked images with dynamic regions filled
        """
        import cv2
        
        num_images = len(images)
        h, w = images.shape[1:3]
        
        # Handle single mask vs batch of masks
        if mask.ndim == 2:
            # Single mask - replicate for all frames
            masks = np.stack([mask] * num_images)
        elif mask.ndim == 3 and mask.shape[0] == num_images:
            masks = mask
        else:
            # Try to broadcast
            if mask.shape[-2:] == (h, w):
                masks = np.broadcast_to(mask, (num_images, h, w)).copy()
            else:
                print(f"[COLMAP] Warning: Mask shape {mask.shape} doesn't match images {images.shape}, skipping mask")
                return images
        
        # Ensure masks are in 0-1 range
        if masks.max() > 1:
            masks = masks / 255.0
        
        # Apply dilation to masks
        if dilation > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1))
            dilated_masks = []
            for m in masks:
                m_uint8 = (m * 255).astype(np.uint8)
                m_dilated = cv2.dilate(m_uint8, kernel, iterations=1)
                dilated_masks.append(m_dilated / 255.0)
            masks = np.array(dilated_masks)
        
        # Create output images
        masked_images = images.copy()
        
        # Fill color for masked regions (neutral gray that won't create strong features)
        fill_color = np.array([0.5, 0.5, 0.5])
        
        for i in range(num_images):
            img = masked_images[i]
            m = masks[i]
            
            # Determine which pixels to mask
            if mask_mode == "exclude_white":
                # White in mask = exclude from tracking (fill with gray)
                mask_binary = m > 0.5
            else:
                # White in mask = include, so mask out the inverse
                mask_binary = m <= 0.5
            
            # Apply mask - fill masked regions with neutral gray
            for c in range(3):
                img[:, :, c] = np.where(mask_binary, fill_color[c], img[:, :, c])
        
        return masked_images
