"""
COLMAP Feature Matcher Node

Matches features between images in the COLMAP database.
"""

from pathlib import Path
from typing import Dict, Tuple, Any


class COLMAPFeatureMatcher:
    """
    Match features between image pairs.
    
    Takes output from COLMAPFeatureExtractor and produces
    matched feature pairs ready for reconstruction.
    """
    
    CATEGORY = "COLMAP"
    FUNCTION = "match"
    RETURN_TYPES = ("COLMAP_DATABASE", "STRING")
    RETURN_NAMES = ("database", "status")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "database": ("COLMAP_DATABASE",),
                "matcher_type": (["exhaustive", "sequential", "vocab_tree"], {
                    "default": "sequential"
                }),
                "gpu_mode": (["auto", "cpu_only", "force_gpu"], {
                    "default": "auto"
                }),
            },
            "optional": {
                # Sequential matcher options
                "sequential_overlap": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of neighboring images to match (sequential only)"
                }),
                "sequential_quadratic": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use quadratic overlap for sequential matcher"
                }),
                # General options
                "max_num_matches": ("INT", {
                    "default": 32768,
                    "min": 1024,
                    "max": 65536,
                    "step": 1024
                }),
            }
        }
    
    def match(
        self,
        database: Dict[str, Any],
        matcher_type: str,
        gpu_mode: str,
        sequential_overlap: int = 10,
        sequential_quadratic: bool = True,
        max_num_matches: int = 32768
    ) -> Tuple[Dict[str, Any], str]:
        """Match features between images."""
        from ..utils import COLMAPWrapper, GPUMode, MatcherType
        
        if not database:
            return ({}, "ERROR: No database provided")
        
        workspace = Path(database.get("workspace", ""))
        if not workspace.exists():
            return ({}, "ERROR: Workspace not found")
        
        try:
            wrapper = COLMAPWrapper(
                workspace=workspace,
                gpu_mode=GPUMode(gpu_mode),
                verbose=True
            )
            
            # Run matching
            wrapper.match_features(matcher_type=MatcherType(matcher_type))
            
            # Update database info
            database["matched"] = True
            
            status = f"Matched features using {matcher_type} matcher"
            return (database, status)
            
        except Exception as e:
            return (database, f"ERROR: {str(e)}")
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
