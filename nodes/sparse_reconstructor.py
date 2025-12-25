"""
COLMAP Sparse Reconstructor Node

Runs incremental Structure-from-Motion to create a sparse 3D reconstruction.
"""

from pathlib import Path
from typing import Dict, Tuple, Any


class COLMAPSparseReconstructor:
    """
    Run sparse reconstruction (incremental SfM).
    
    Takes a matched COLMAP database and produces a 3D reconstruction
    with camera poses and sparse point cloud.
    """
    
    CATEGORY = "COLMAP"
    FUNCTION = "reconstruct"
    RETURN_TYPES = ("COLMAP_RECONSTRUCTION", "STRING")
    RETURN_NAMES = ("reconstruction", "status")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "database": ("COLMAP_DATABASE",),
            },
            "optional": {
                "min_num_matches": ("INT", {
                    "default": 15,
                    "min": 5,
                    "max": 100,
                    "step": 5,
                    "tooltip": "Minimum number of matches for an image pair"
                }),
                "multiple_models": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Allow multiple separate reconstructions"
                }),
                "min_model_size": ("INT", {
                    "default": 3,
                    "min": 2,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Minimum number of images in a model"
                }),
                "init_num_trials": ("INT", {
                    "default": 200,
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "tooltip": "Number of trials for initial pair selection"
                }),
            }
        }
    
    def reconstruct(
        self,
        database: Dict[str, Any],
        min_num_matches: int = 15,
        multiple_models: bool = False,
        min_model_size: int = 3,
        init_num_trials: int = 200
    ) -> Tuple[Dict[str, Any], str]:
        """Run sparse reconstruction."""
        from ..utils import COLMAPWrapper, GPUMode
        
        if not database:
            return ({}, "ERROR: No database provided")
        
        workspace = Path(database.get("workspace", ""))
        if not workspace.exists():
            return ({}, "ERROR: Workspace not found")
        
        try:
            wrapper = COLMAPWrapper(
                workspace=workspace,
                gpu_mode=GPUMode.CPU_ONLY,  # Reconstruction is CPU-only anyway
                verbose=True
            )
            
            # Run reconstruction
            reconstructions = wrapper.sparse_reconstruction(
                min_num_matches=min_num_matches,
                multiple_models=multiple_models
            )
            
            if not reconstructions:
                return ({}, "ERROR: Reconstruction failed - no valid models")
            
            # Get summary
            summary = wrapper.get_reconstruction_summary()
            
            # Build reconstruction info
            reconstruction_info = {
                "workspace": str(workspace),
                "database_path": str(wrapper.database_path),
                "image_path": str(wrapper.image_path),
                "sparse_path": str(wrapper.sparse_path),
                "reconstruction": wrapper.reconstruction,
                "summary": summary,
                "num_models": len(reconstructions),
            }
            
            status = (
                f"Reconstruction complete: "
                f"{summary['num_registered_images']}/{database.get('num_images', '?')} images, "
                f"{summary['num_points3D']} points, "
                f"error: {summary['mean_reprojection_error']:.2f}px"
            )
            
            return (reconstruction_info, status)
            
        except Exception as e:
            import traceback
            return ({}, f"ERROR: {str(e)}\n{traceback.format_exc()}")
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
