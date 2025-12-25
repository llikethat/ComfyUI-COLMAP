"""
COLMAP Feature Extractor Node

Extracts SIFT features from images and stores them in a COLMAP database.
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


class COLMAPFeatureExtractor:
    """
    Extract SIFT features from images.
    
    This is the first step in the COLMAP pipeline.
    Output can be fed into COLMAPFeatureMatcher.
    """
    
    CATEGORY = "COLMAP"
    FUNCTION = "extract"
    RETURN_TYPES = ("COLMAP_DATABASE", "STRING")
    RETURN_NAMES = ("database", "status")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "feature_type": (["sift", "sift_gpu"], {
                    "default": "sift"
                }),
                "max_features": ("INT", {
                    "default": 8192,
                    "min": 512,
                    "max": 32768,
                    "step": 512
                }),
                "first_octave": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 4,
                    "step": 1,
                    "tooltip": "-1 for upsampled image (more features)"
                }),
                "num_octaves": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 8,
                    "step": 1
                }),
                "gpu_mode": (["auto", "cpu_only", "force_gpu"], {
                    "default": "auto"
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Mask for dynamic objects (white=exclude). Use to mask moving subjects."
                }),
                "mask_mode": (["exclude_white", "include_white"], {
                    "default": "exclude_white"
                }),
                "mask_dilation": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 100,
                    "step": 5
                }),
                "image_names": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
            }
        }
    
    def extract(
        self,
        images,
        feature_type: str,
        max_features: int,
        first_octave: int,
        num_octaves: int,
        gpu_mode: str,
        mask = None,
        mask_mode: str = "exclude_white",
        mask_dilation: int = 10,
        image_names: str = ""
    ) -> Tuple[Dict[str, Any], str]:
        """Extract features from images."""
        import torch
        
        from ..utils import (
            COLMAPWrapper, GPUMode, FeatureType,
            check_colmap_installation
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
        
        # Check availability
        is_available, status_msg = check_colmap_installation()
        if not is_available:
            return ({}, f"ERROR: COLMAP not available. {status_msg}")
        
        # Parse image names
        name_list = None
        if image_names.strip():
            name_list = [n.strip() for n in image_names.strip().split('\n') if n.strip()]
        
        # Apply mask if provided
        processed_images = images_np
        if mask_np is not None:
            processed_images = self._apply_mask(images_np, mask_np, mask_mode, mask_dilation)
            print(f"[COLMAP] Applied mask to {len(images_np)} images")
        
        # Create workspace
        workspace = Path(tempfile.mkdtemp(prefix="colmap_feat_"))
        
        try:
            wrapper = COLMAPWrapper(
                workspace=workspace,
                gpu_mode=GPUMode(gpu_mode),
                verbose=True
            )
            
            # Save images (with mask applied)
            wrapper.prepare_images(processed_images, name_list)
            
            # Extract features
            wrapper.extract_features(
                feature_type=FeatureType(feature_type),
                max_num_features=max_features,
                first_octave=first_octave,
                num_octaves=num_octaves
            )
            
            # Return database info for next stage
            database_info = {
                "workspace": str(workspace),
                "database_path": str(wrapper.database_path),
                "image_path": str(wrapper.image_path),
                "num_images": len(images_np),
            }
            
            status = f"Extracted features from {len(images)} images"
            return (database_info, status)
            
        except Exception as e:
            return ({}, f"ERROR: {str(e)}")
    
    def _apply_mask(
        self,
        images: np.ndarray,
        mask: np.ndarray,
        mask_mode: str,
        dilation: int
    ) -> np.ndarray:
        """Apply mask to images for background-only feature extraction."""
        import cv2
        
        num_images = len(images)
        h, w = images.shape[1:3]
        
        # Handle single mask vs batch
        if mask.ndim == 2:
            masks = np.stack([mask] * num_images)
        elif mask.ndim == 3 and mask.shape[0] == num_images:
            masks = mask
        else:
            print(f"[COLMAP] Warning: Mask shape mismatch, skipping")
            return images
        
        if masks.max() > 1:
            masks = masks / 255.0
        
        # Dilate masks
        if dilation > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1))
            dilated = []
            for m in masks:
                m_uint8 = (m * 255).astype(np.uint8)
                m_dilated = cv2.dilate(m_uint8, kernel, iterations=1)
                dilated.append(m_dilated / 255.0)
            masks = np.array(dilated)
        
        # Apply mask
        masked_images = images.copy()
        fill_color = np.array([0.5, 0.5, 0.5])
        
        for i in range(num_images):
            mask_binary = masks[i] > 0.5 if mask_mode == "exclude_white" else masks[i] <= 0.5
            for c in range(3):
                masked_images[i, :, :, c] = np.where(mask_binary, fill_color[c], masked_images[i, :, :, c])
        
        return masked_images
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # Always re-execute


class COLMAPDatabaseInfo:
    """Type hint for COLMAP database info dictionary"""
    workspace: str
    database_path: str
    image_path: str
    num_images: int
