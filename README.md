# ComfyUI-COLMAP

**Structure-from-Motion camera tracking for ComfyUI**

Extract camera intrinsics, extrinsics, and motion data from video sequences using COLMAP's robust SfM pipeline.

![COLMAP Pipeline](https://colmap.github.io//_images/incremental-sfm.png)

## Features

- 🎯 **Robust Camera Tracking** — COLMAP's industry-standard SfM
- 📊 **Motion Analysis** — Detect pan, tilt, roll, dolly, truck, crane, drone movements
- 🔄 **Multiple Coordinate Systems** — Blender, Unreal, Unity, Maya, OpenGL, USD, etc.
- 💾 **Multiple Export Formats** — JSON, CSV, Alembic, FBX, Nuke .chan, OpenCV YAML
- 🎬 **SAM3DBody Integration** — Optional scene combining with body mesh tracking
- ⚡ **GPU Acceleration** — Optional CUDA support for faster processing

## Installation

### 1. Install the Custom Node

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/llikethat/ComfyUI-COLMAP.git
cd ComfyUI-COLMAP
pip install -r requirements.txt
```

### 2. Install COLMAP

**Option A: pycolmap (Recommended)**
```bash
# CPU version
pip install pycolmap

# GPU/CUDA version (Linux only)
pip install pycolmap-cuda12
```

**Option B: COLMAP CLI**
- Ubuntu: `sudo apt install colmap`
- macOS: `brew install colmap`
- Windows: [Download from GitHub](https://github.com/colmap/colmap/releases)

### 3. Optional: Blender (for Alembic/FBX export)
Install Blender and ensure it's in your PATH:
```bash
# Ubuntu
sudo apt install blender

# macOS
brew install --cask blender
```

## Nodes

### 🚀 COLMAP Auto Reconstruct
All-in-one node for simple workflows. Takes images, outputs camera data.

**Inputs:**
- `images` — Batch of images (from VHS Load Video, etc.)
- `matcher_type` — exhaustive, sequential, or vocab_tree
- `feature_type` — sift or sift_gpu
- `max_features` — Maximum SIFT features per image (default: 8192)
- `min_matches` — Minimum matches for image pairs (default: 15)
- `gpu_mode` — auto, cpu_only, force_gpu, or force_offload

**Outputs:**
- `camera_data` — Complete camera tracking data (CAMERA_DATA type)
- `status` — Processing status message
- `sparse_points_preview` — Visualization of sparse point cloud

---

### 🎯 COLMAP Feature Extractor
Extract SIFT features from images.

### 🔗 COLMAP Feature Matcher
Match features between image pairs.

### 🏗️ COLMAP Sparse Reconstructor
Run incremental SfM reconstruction.

### 📷 COLMAP Camera Extractor
Extract camera data from reconstruction with coordinate system conversion.

### 📊 COLMAP Motion Analyzer
Analyze camera motion to detect:
- **Pan** — Horizontal rotation (left/right)
- **Tilt** — Vertical rotation (up/down)
- **Roll** — Rotation around view axis
- **Dolly** — Forward/backward movement
- **Truck** — Left/right movement
- **Crane** — Up/down movement
- **Motion Classification** — Static, handheld, tripod, drone, tracking, orbit

### 💾 COLMAP Camera Exporter
Export camera data to various formats:
- **JSON** — Universal format with all data
- **CSV** — Spreadsheet-compatible
- **Alembic (.abc)** — Camera animation for 3D software
- **FBX** — 3D scene with animated camera
- **Nuke .chan** — VFX compositing format
- **OpenCV YAML** — Computer vision applications
- **COLMAP Text** — Native COLMAP format

## Workflows

### Basic Camera Tracking
```
[VHS Load Video] → [COLMAP Auto Reconstruct] → [COLMAP Camera Exporter]
                                             ↘
                                              [COLMAP Motion Analyzer]
```

### Advanced Pipeline
```
[VHS Load Video] → [COLMAP Feature Extractor] → [COLMAP Feature Matcher] 
                                                          ↓
[COLMAP Camera Exporter] ← [COLMAP Camera Extractor] ← [COLMAP Sparse Reconstructor]
```


## Coordinate Systems

| System | Up | Forward | Handedness | Use Case |
|--------|-----|---------|------------|----------|
| `colmap` | -Y | +Z | Right | COLMAP native |
| `blender` | +Z | +Y | Right | Blender, 3D modeling |
| `opengl` | +Y | -Z | Right | OpenGL, WebGL |
| `opencv` | -Y | +Z | Right | Computer vision |
| `unreal` | +Z | +X | Left | Unreal Engine |
| `unity` | +Y | +Z | Left | Unity |
| `maya` | +Y | +Z | Right | Autodesk Maya |
| `houdini` | +Y | -Z | Right | Houdini |
| `usd` | +Y | -Z | Right | Universal Scene Description |

## Camera Motion Output

The motion analyzer provides per-frame motion data:

```json
{
  "frame_001": {
    "rotation": {
      "pan": 2.3,      // degrees/frame
      "tilt": -0.5,
      "roll": 0.1
    },
    "translation": {
      "dolly": 0.02,   // units/frame
      "truck": 0.01,
      "crane": 0.005
    },
    "motion_type": "handheld",
    "speed": 0.15
  }
}
```

## GPU Memory Management

When using ComfyUI with Stable Diffusion models loaded, GPU memory can be a concern. Use the `gpu_mode` option:

| Mode | Description |
|------|-------------|
| `auto` | Automatically detect and use GPU if available |
| `cpu_only` | Force CPU processing (slower but no VRAM conflict) |
| `force_gpu` | Always use GPU (may fail if VRAM is full) |
| `force_offload` | Unload SD models from VRAM before COLMAP processing |

## Tips for Best Results

1. **Image Quality** — Use sharp, well-lit images with minimal motion blur
2. **Overlap** — Ensure 60-80% overlap between consecutive frames
3. **Avoid** — Pure rotation (no parallax), textureless surfaces, moving objects
4. **Sequential Matcher** — Best for video sequences with ordered frames
5. **Exhaustive Matcher** — Best for unordered photo collections (slower)

## Masking Dynamic Objects (Important!)

**Problem:** COLMAP assumes a static scene. Moving subjects (people, cars) create features that confuse the solver.

**Solution:** Use the `mask` input to exclude dynamic objects:

```
[Video] ──┬──► [SAM/Segmentation] ──► [Person Mask]
          │                                 │
          │                                 ▼
          └──────────────────────► [COLMAP Auto Reconstruct] ──► Clean camera
                                   (mask input)                    tracking
```

**Mask Options:**
| Parameter | Description |
|-----------|-------------|
| `mask` | MASK input - white areas are excluded |
| `mask_mode` | `exclude_white` (default) or `include_white` |
| `mask_dilation` | Expand mask by N pixels (default: 10) |

**Workflow with SAM3DBody:**
1. Run SAM segmentation to get person mask
2. Feed mask to COLMAP (inverted, so person = white = excluded)
3. COLMAP tracks camera using only background features
4. Apply tracked camera to SAM3DBody mesh sequence

## Troubleshooting

### "No valid models produced"
- Check that images have sufficient texture and overlap
- Try lowering `min_matches` threshold
- Try `exhaustive` matcher instead of `sequential`

### "COLMAP not available"
- Install pycolmap: `pip install pycolmap`
- Or install COLMAP CLI and add to PATH

### Out of GPU memory
- Use `gpu_mode: cpu_only`
- Or use `gpu_mode: force_offload` to free SD model VRAM first

## License

MIT License - See LICENSE file

## Credits

- [COLMAP](https://colmap.github.io/) — Johannes L. Schönberger
- [pycolmap](https://github.com/colmap/colmap/tree/main/pycolmap) — COLMAP Python bindings
