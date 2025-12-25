"""
COLMAP Wrapper

Handles COLMAP operations via pycolmap bindings with fallback to CLI.
Manages GPU memory and provides progress callbacks.
"""

import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple, Any
from enum import Enum
import numpy as np

# Try to import pycolmap
try:
    import pycolmap
    PYCOLMAP_AVAILABLE = True
except ImportError:
    PYCOLMAP_AVAILABLE = False
    print("[ComfyUI-COLMAP] Warning: pycolmap not available, will use CLI fallback")


class GPUMode(Enum):
    AUTO = "auto"
    CPU_ONLY = "cpu_only"
    FORCE_GPU = "force_gpu"
    FORCE_OFFLOAD = "force_offload"


class MatcherType(Enum):
    EXHAUSTIVE = "exhaustive"
    SEQUENTIAL = "sequential"
    VOCAB_TREE = "vocab_tree"
    SPATIAL = "spatial"
    TRANSITIVE = "transitive"


class FeatureType(Enum):
    SIFT = "sift"
    SIFT_GPU = "sift_gpu"


class COLMAPError(Exception):
    """Custom exception for COLMAP-related errors"""
    pass


class COLMAPWrapper:
    """
    Wrapper for COLMAP operations with pycolmap or CLI fallback.
    """
    
    def __init__(
        self,
        workspace: Optional[Path] = None,
        gpu_mode: GPUMode = GPUMode.AUTO,
        verbose: bool = True
    ):
        self.workspace = workspace or Path(tempfile.mkdtemp(prefix="colmap_"))
        self.gpu_mode = gpu_mode
        self.verbose = verbose
        
        # Create workspace directories
        self.image_path = self.workspace / "images"
        self.database_path = self.workspace / "database.db"
        self.sparse_path = self.workspace / "sparse"
        self.dense_path = self.workspace / "dense"
        
        for path in [self.image_path, self.sparse_path, self.dense_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # State
        self.reconstruction = None
        self._progress_callback: Optional[Callable[[str, float], None]] = None
    
    def set_progress_callback(self, callback: Callable[[str, float], None]):
        """Set callback for progress updates: callback(stage_name, progress_0_to_1)"""
        self._progress_callback = callback
    
    def _report_progress(self, stage: str, progress: float):
        if self._progress_callback:
            self._progress_callback(stage, progress)
        if self.verbose:
            print(f"[COLMAP] {stage}: {progress*100:.1f}%")
    
    def _use_gpu(self) -> bool:
        """Determine whether to use GPU based on gpu_mode and availability"""
        if self.gpu_mode == GPUMode.CPU_ONLY:
            return False
        if self.gpu_mode == GPUMode.FORCE_GPU:
            return True
        if self.gpu_mode == GPUMode.FORCE_OFFLOAD:
            self._offload_comfyui_models()
            return True
        # AUTO: check if GPU is available
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _offload_comfyui_models(self):
        """Attempt to offload ComfyUI models from GPU memory"""
        try:
            # ComfyUI's model management
            import comfy.model_management as model_management
            model_management.soft_empty_cache()
            model_management.unload_all_models()
            if self.verbose:
                print("[COLMAP] Offloaded ComfyUI models from GPU")
        except ImportError:
            pass
        except Exception as e:
            if self.verbose:
                print(f"[COLMAP] Warning: Could not offload models: {e}")
    
    def prepare_images(
        self,
        images,
        image_names: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Save images to workspace for COLMAP processing.
        
        Args:
            images: Tensor or Numpy array of shape (N, H, W, 3) in [0, 1] range
            image_names: Optional list of image names
            
        Returns:
            List of saved image paths
        """
        from PIL import Image
        
        # Convert tensor to numpy if needed
        try:
            import torch
            if isinstance(images, torch.Tensor):
                images = images.cpu().numpy()
        except ImportError:
            pass
        
        # Ensure we have a numpy array
        if not isinstance(images, np.ndarray):
            images = np.array(images)
        
        # Clear existing images
        for f in self.image_path.glob("*"):
            f.unlink()
        
        saved_paths = []
        num_images = len(images)
        
        for i in range(num_images):
            img_array = images[i]
            
            # Ensure this frame is also numpy
            if not isinstance(img_array, np.ndarray):
                try:
                    import torch
                    if isinstance(img_array, torch.Tensor):
                        img_array = img_array.cpu().numpy()
                    else:
                        img_array = np.array(img_array)
                except ImportError:
                    img_array = np.array(img_array)
            
            # Convert to uint8
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
            
            # Generate filename
            if image_names and i < len(image_names):
                name = image_names[i]
            else:
                name = f"frame_{i:06d}.jpg"
            
            if not name.lower().endswith(('.jpg', '.jpeg', '.png')):
                name = f"{name}.jpg"
            
            # Save image
            img_path = self.image_path / name
            Image.fromarray(img_array).save(img_path, quality=95)
            saved_paths.append(img_path)
            
            self._report_progress("Saving images", (i + 1) / num_images)
        
        return saved_paths
    
    def extract_features(
        self,
        feature_type: FeatureType = FeatureType.SIFT,
        max_num_features: int = 8192,
        first_octave: int = -1,
        num_octaves: int = 4,
        **kwargs
    ) -> bool:
        """
        Extract features from images.
        
        Args:
            feature_type: Type of features to extract
            max_num_features: Maximum number of features per image
            **kwargs: Additional SIFT options
            
        Returns:
            True if successful
        """
        self._report_progress("Feature extraction", 0.0)
        
        if PYCOLMAP_AVAILABLE:
            try:
                use_gpu = self._use_gpu() and feature_type == FeatureType.SIFT_GPU
                
                # Set device as separate parameter (new pycolmap API)
                if use_gpu:
                    device = pycolmap.Device.cuda
                else:
                    device = pycolmap.Device.cpu
                
                # Create extraction options object
                extraction_options = pycolmap.SiftExtractionOptions()
                extraction_options.max_num_features = max_num_features
                extraction_options.first_octave = first_octave
                extraction_options.num_octaves = num_octaves
                
                pycolmap.extract_features(
                    database_path=str(self.database_path),
                    image_path=str(self.image_path),
                    sift_options=extraction_options,
                    device=device
                )
                
                self._report_progress("Feature extraction", 1.0)
                return True
                
            except Exception as e:
                print(f"[COLMAP] pycolmap feature extraction failed: {e}")
                # Try alternative API (older pycolmap versions)
                try:
                    return self._extract_features_pycolmap_legacy(
                        feature_type, max_num_features, first_octave, num_octaves
                    )
                except Exception as e2:
                    print(f"[COLMAP] Legacy pycolmap also failed: {e2}")
                    print("[COLMAP] Falling back to CLI...")
        
        # CLI fallback
        return self._extract_features_cli(feature_type, max_num_features)
    
    def _extract_features_pycolmap_legacy(
        self,
        feature_type: FeatureType,
        max_num_features: int,
        first_octave: int,
        num_octaves: int
    ) -> bool:
        """Try older pycolmap API versions"""
        use_gpu = self._use_gpu() and feature_type == FeatureType.SIFT_GPU
        
        # Try different API patterns
        try:
            # Pattern 1: FeatureExtractionOptions object
            opts = pycolmap.FeatureExtractionOptions()
            opts.sift.max_num_features = max_num_features
            opts.sift.first_octave = first_octave
            opts.sift.num_octaves = num_octaves
            
            device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu
            
            pycolmap.extract_features(
                database_path=str(self.database_path),
                image_path=str(self.image_path),
                extraction_options=opts,
                device=device
            )
            self._report_progress("Feature extraction", 1.0)
            return True
        except:
            pass
        
        # Pattern 2: Simple dict (very old versions)
        try:
            pycolmap.extract_features(
                str(self.database_path),
                str(self.image_path)
            )
            self._report_progress("Feature extraction", 1.0)
            return True
        except:
            pass
        
        raise Exception("No compatible pycolmap API found")
    
    def _extract_features_cli(
        self,
        feature_type: FeatureType,
        max_num_features: int
    ) -> bool:
        """Feature extraction via COLMAP CLI"""
        cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(self.database_path),
            "--image_path", str(self.image_path),
            "--SiftExtraction.max_num_features", str(max_num_features),
        ]
        
        if not self._use_gpu():
            cmd.extend(["--SiftExtraction.use_gpu", "0"])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise COLMAPError(f"Feature extraction failed: {result.stderr}")
            self._report_progress("Feature extraction", 1.0)
            return True
        except FileNotFoundError:
            raise COLMAPError("COLMAP CLI not found. Please install COLMAP.")
    
    def match_features(
        self,
        matcher_type: MatcherType = MatcherType.EXHAUSTIVE,
        **kwargs
    ) -> bool:
        """
        Match features across images.
        
        Args:
            matcher_type: Type of matcher to use
            **kwargs: Additional matcher options
            
        Returns:
            True if successful
        """
        self._report_progress("Feature matching", 0.0)
        
        if PYCOLMAP_AVAILABLE:
            try:
                # Create matching options - FeatureMatchingOptions has use_gpu
                matching_options = pycolmap.FeatureMatchingOptions()
                matching_options.use_gpu = False  # Force CPU to avoid CUDA issues
                
                print(f"[COLMAP] Matching with use_gpu={matching_options.use_gpu}")
                
                if matcher_type == MatcherType.EXHAUSTIVE:
                    pycolmap.match_exhaustive(
                        database_path=str(self.database_path),
                        matching_options=matching_options,
                        device=pycolmap.Device.cpu
                    )
                elif matcher_type == MatcherType.SEQUENTIAL:
                    pycolmap.match_sequential(
                        database_path=str(self.database_path),
                        matching_options=matching_options,
                        device=pycolmap.Device.cpu
                    )
                else:
                    pycolmap.match_exhaustive(
                        database_path=str(self.database_path),
                        matching_options=matching_options,
                        device=pycolmap.Device.cpu
                    )
                
                self._report_progress("Feature matching", 1.0)
                return True
                
            except Exception as e:
                print(f"[COLMAP] pycolmap matching failed: {e}")
                import traceback
                traceback.print_exc()
                print("[COLMAP] Falling back to CLI...")
        
        return self._match_features_cli(matcher_type)
    
    def _match_features_cli(self, matcher_type: MatcherType) -> bool:
        """Feature matching via COLMAP CLI"""
        matcher_cmd = {
            MatcherType.EXHAUSTIVE: "exhaustive_matcher",
            MatcherType.SEQUENTIAL: "sequential_matcher",
            MatcherType.VOCAB_TREE: "vocab_tree_matcher",
            MatcherType.SPATIAL: "spatial_matcher",
            MatcherType.TRANSITIVE: "transitive_matcher",
        }
        
        cmd = [
            "colmap", matcher_cmd.get(matcher_type, "exhaustive_matcher"),
            "--database_path", str(self.database_path),
        ]
        
        if not self._use_gpu():
            cmd.extend(["--SiftMatching.use_gpu", "0"])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise COLMAPError(f"Feature matching failed: {result.stderr}")
            self._report_progress("Feature matching", 1.0)
            return True
        except FileNotFoundError:
            raise COLMAPError("COLMAP CLI not found. Please install COLMAP.")
    
    def sparse_reconstruction(
        self,
        min_num_matches: int = 15,
        multiple_models: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[int, Any]:
        """
        Run sparse reconstruction (incremental SfM).
        
        Args:
            min_num_matches: Minimum matches for image pair
            multiple_models: Allow multiple sub-reconstructions
            progress_callback: Called with (num_registered, total_images)
            
        Returns:
            Dictionary of reconstruction objects
        """
        self._report_progress("Sparse reconstruction", 0.0)
        
        if PYCOLMAP_AVAILABLE:
            try:
                # Clear existing sparse output
                if self.sparse_path.exists():
                    shutil.rmtree(self.sparse_path)
                self.sparse_path.mkdir(parents=True)
                
                # Get total image count for progress
                try:
                    with pycolmap.Database.open(self.database_path) as db:
                        num_images = db.num_images()
                except:
                    # Older API
                    db = pycolmap.Database(str(self.database_path))
                    num_images = len(list(db.read_all_images()))
                    db.close()
                
                registered = [0]  # Mutable for closure
                
                def on_initial_pair():
                    registered[0] = 2
                    self._report_progress("Sparse reconstruction", 2 / max(num_images, 1))
                    if progress_callback:
                        progress_callback(2, num_images)
                
                def on_next_image():
                    registered[0] += 1
                    self._report_progress("Sparse reconstruction", registered[0] / max(num_images, 1))
                    if progress_callback:
                        progress_callback(registered[0], num_images)
                
                # Try different API patterns
                reconstructions = None
                
                # Pattern 1: Newest API with options dict
                try:
                    reconstructions = pycolmap.incremental_mapping(
                        database_path=str(self.database_path),
                        image_path=str(self.image_path),
                        output_path=str(self.sparse_path),
                        options={
                            "min_num_matches": min_num_matches,
                            "multiple_models": multiple_models,
                        }
                    )
                except TypeError:
                    pass
                
                # Pattern 2: With callbacks
                if reconstructions is None:
                    try:
                        reconstructions = pycolmap.incremental_mapping(
                            self.database_path,
                            self.image_path,
                            self.sparse_path,
                            initial_image_pair_callback=on_initial_pair,
                            next_image_callback=on_next_image,
                            options={
                                "min_num_matches": min_num_matches,
                                "multiple_models": multiple_models,
                            }
                        )
                    except TypeError:
                        pass
                
                # Pattern 3: Minimal API
                if reconstructions is None:
                    try:
                        reconstructions = pycolmap.incremental_mapping(
                            str(self.database_path),
                            str(self.image_path),
                            str(self.sparse_path)
                        )
                    except:
                        pass
                
                # Store the best reconstruction
                if reconstructions:
                    # Get reconstruction with most registered images
                    if isinstance(reconstructions, dict):
                        best_idx = max(
                            reconstructions.keys(),
                            key=lambda k: reconstructions[k].num_reg_images() if hasattr(reconstructions[k], 'num_reg_images') else 0
                        )
                        self.reconstruction = reconstructions[best_idx]
                    elif hasattr(reconstructions, '__iter__'):
                        # List of reconstructions
                        self.reconstruction = reconstructions[0] if reconstructions else None
                    else:
                        self.reconstruction = reconstructions
                
                self._report_progress("Sparse reconstruction", 1.0)
                return reconstructions if isinstance(reconstructions, dict) else {0: reconstructions} if reconstructions else {}
                
            except Exception as e:
                print(f"[COLMAP] pycolmap reconstruction failed: {e}")
                import traceback
                traceback.print_exc()
                print("[COLMAP] Falling back to CLI...")
        
        return self._sparse_reconstruction_cli(min_num_matches, multiple_models)
    
    def _sparse_reconstruction_cli(
        self,
        min_num_matches: int,
        multiple_models: bool
    ) -> Dict[int, Any]:
        """Sparse reconstruction via COLMAP CLI"""
        if self.sparse_path.exists():
            shutil.rmtree(self.sparse_path)
        self.sparse_path.mkdir(parents=True)
        
        cmd = [
            "colmap", "mapper",
            "--database_path", str(self.database_path),
            "--image_path", str(self.image_path),
            "--output_path", str(self.sparse_path),
            "--Mapper.min_num_matches", str(min_num_matches),
        ]
        
        if multiple_models:
            cmd.extend(["--Mapper.multiple_models", "1"])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise COLMAPError(f"Sparse reconstruction failed: {result.stderr}")
            
            # Load reconstructions from output
            reconstructions = {}
            for i, model_dir in enumerate(sorted(self.sparse_path.iterdir())):
                if model_dir.is_dir():
                    if PYCOLMAP_AVAILABLE:
                        rec = pycolmap.Reconstruction(str(model_dir))
                        reconstructions[i] = rec
                        if i == 0:
                            self.reconstruction = rec
            
            self._report_progress("Sparse reconstruction", 1.0)
            return reconstructions
            
        except FileNotFoundError:
            raise COLMAPError("COLMAP CLI not found. Please install COLMAP.")
    
    def get_reconstruction_summary(self) -> Dict[str, Any]:
        """Get summary of the current reconstruction"""
        if self.reconstruction is None:
            return {"error": "No reconstruction available"}
        
        rec = self.reconstruction
        return {
            "num_cameras": len(rec.cameras) if hasattr(rec, 'cameras') else 0,
            "num_images": rec.num_images() if hasattr(rec, 'num_images') else 0,
            "num_registered_images": rec.num_reg_images() if hasattr(rec, 'num_reg_images') else 0,
            "num_points3D": rec.num_points3D() if hasattr(rec, 'num_points3D') else 0,
            "mean_reprojection_error": self._compute_mean_reprojection_error(),
        }
    
    def _compute_mean_reprojection_error(self) -> float:
        """Compute mean reprojection error across all observations"""
        if self.reconstruction is None or not PYCOLMAP_AVAILABLE:
            return 0.0
        
        try:
            total_error = 0.0
            num_obs = 0
            for point in self.reconstruction.points3D.values():
                total_error += point.error * len(point.track.elements)
                num_obs += len(point.track.elements)
            
            if num_obs > 0:
                return total_error / num_obs
            return 0.0
        except Exception:
            return 0.0
    
    def cleanup(self):
        """Clean up workspace"""
        if self.workspace.exists():
            shutil.rmtree(self.workspace)


def check_colmap_installation() -> Tuple[bool, str]:
    """
    Check if COLMAP is properly installed.
    
    Returns:
        Tuple of (is_available, status_message)
    """
    messages = []
    
    # Check pycolmap
    if PYCOLMAP_AVAILABLE:
        try:
            version = pycolmap.__version__
            messages.append(f"pycolmap {version} available")
        except Exception:
            messages.append("pycolmap available (version unknown)")
    else:
        messages.append("pycolmap not installed")
    
    # Check CLI
    try:
        result = subprocess.run(
            ["colmap", "help"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            messages.append("COLMAP CLI available")
        else:
            messages.append("COLMAP CLI found but returned error")
    except FileNotFoundError:
        messages.append("COLMAP CLI not found")
    
    is_available = PYCOLMAP_AVAILABLE or "CLI available" in str(messages)
    return is_available, " | ".join(messages)
