"""
ComfyUI-COLMAP
Structure from Motion camera tracking for ComfyUI

Features:
- COLMAP-based SfM for robust camera tracking
- Camera motion analysis (pan/tilt/dolly/crane/drone)
- Multiple export formats (JSON, Alembic, FBX, Nuke .chan)
- Flexible coordinate system support
- Integration with SAM3DBody for combined body+camera workflows
"""

from .nodes.feature_extractor import COLMAPFeatureExtractor
from .nodes.feature_matcher import COLMAPFeatureMatcher
from .nodes.sparse_reconstructor import COLMAPSparseReconstructor
from .nodes.auto_reconstruct import COLMAPAutoReconstruct
from .nodes.camera_extractor import COLMAPCameraExtractor
from .nodes.motion_analyzer import COLMAPMotionAnalyzer
from .nodes.camera_exporter import COLMAPCameraExporter
from .nodes.scene_combiner import COLMAPSceneCombiner
from .nodes.camera_visualizer import COLMAPCameraVisualizer
from .nodes.sam3dbody_bridge import COLMAPToSAM3DBodyCamera

__version__ = "1.0.2"

NODE_CLASS_MAPPINGS = {
    # Core COLMAP Pipeline
    "COLMAPFeatureExtractor": COLMAPFeatureExtractor,
    "COLMAPFeatureMatcher": COLMAPFeatureMatcher,
    "COLMAPSparseReconstructor": COLMAPSparseReconstructor,
    
    # All-in-one
    "COLMAPAutoReconstruct": COLMAPAutoReconstruct,
    
    # Camera Processing
    "COLMAPCameraExtractor": COLMAPCameraExtractor,
    "COLMAPMotionAnalyzer": COLMAPMotionAnalyzer,
    "COLMAPCameraExporter": COLMAPCameraExporter,
    
    # Visualization
    "COLMAPCameraVisualizer": COLMAPCameraVisualizer,
    
    # Integration
    "COLMAPSceneCombiner": COLMAPSceneCombiner,
    "COLMAPToSAM3DBodyCamera": COLMAPToSAM3DBodyCamera,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "COLMAPFeatureExtractor": "🎯 COLMAP Feature Extractor",
    "COLMAPFeatureMatcher": "🔗 COLMAP Feature Matcher",
    "COLMAPSparseReconstructor": "🏗️ COLMAP Sparse Reconstructor",
    "COLMAPAutoReconstruct": "🚀 COLMAP Auto Reconstruct",
    "COLMAPCameraExtractor": "📷 COLMAP Camera Extractor",
    "COLMAPMotionAnalyzer": "📊 COLMAP Motion Analyzer",
    "COLMAPCameraExporter": "💾 COLMAP Camera Exporter",
    "COLMAPCameraVisualizer": "👁️ COLMAP Camera Visualizer",
    "COLMAPSceneCombiner": "🎬 Scene Combiner (Camera + Body)",
    "COLMAPToSAM3DBodyCamera": "🔄 COLMAP → SAM3DBody Camera",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
