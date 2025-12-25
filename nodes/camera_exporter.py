"""
COLMAP Camera Exporter Node

Exports camera data to various formats:
- JSON: Universal format with all data
- CSV: Spreadsheet-compatible
- Alembic (.abc): Camera animation
- FBX: 3D scene with camera
- Nuke .chan: VFX compositing
- OpenCV YAML: Computer vision
"""

import os
from pathlib import Path
from typing import Tuple, Any, Optional

import numpy as np

# ComfyUI imports
try:
    import folder_paths
    OUTPUT_DIR = Path(folder_paths.get_output_directory())
except ImportError:
    OUTPUT_DIR = Path("./output")


class COLMAPCameraExporter:
    """
    Export camera data to various formats.
    
    Supports JSON, CSV, Alembic, FBX, Nuke .chan, and OpenCV YAML.
    """
    
    CATEGORY = "COLMAP"
    FUNCTION = "export"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_path", "status")
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_data": ("CAMERA_DATA",),
                "format": ([
                    "json", "csv", "alembic", "fbx", 
                    "nuke_chan", "opencv_yaml", "colmap_text"
                ], {
                    "default": "json"
                }),
                "filename": ("STRING", {
                    "default": "camera_data"
                }),
            },
            "optional": {
                "include_sparse_points": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include sparse points in export (where supported)"
                }),
                "include_motion": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include motion analysis in export"
                }),
                "coordinate_system": ([
                    "keep_original", "colmap", "blender", "opengl", 
                    "opencv", "unreal", "unity", "maya", "houdini", "usd"
                ], {
                    "default": "keep_original",
                    "tooltip": "Convert coordinates before export"
                }),
                "frame_offset": ("INT", {
                    "default": 0,
                    "min": -1000,
                    "max": 1000,
                    "tooltip": "Offset to add to frame numbers (e.g., -2 shifts frame 3 → frame 1)"
                }),
                "auto_frame_offset": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-offset frames to start from frame 1"
                }),
            }
        }
    
    def export(
        self,
        camera_data: Any,
        format: str,
        filename: str,
        include_sparse_points: bool = True,
        include_motion: bool = True,
        coordinate_system: str = "keep_original",
        frame_offset: int = 0,
        auto_frame_offset: bool = True
    ) -> Tuple[str, str]:
        """Export camera data to specified format."""
        
        if not camera_data or not camera_data.extrinsics:
            return ("", "ERROR: No camera data to export")
        
        # Calculate and apply frame offset
        calculated_offset = frame_offset
        if camera_data.extrinsics:
            min_frame = min(ext.frame_index for ext in camera_data.extrinsics)
            
            if auto_frame_offset and min_frame > 1:
                # Auto-offset to start from frame 1
                calculated_offset = -(min_frame - 1)
                print(f"[COLMAP Export] Auto frame offset: {calculated_offset} (min frame was {min_frame})")
            
            if calculated_offset != 0:
                print(f"[COLMAP Export] Applying frame offset: {calculated_offset}")
                # Apply offset to all extrinsics
                for ext in camera_data.extrinsics:
                    ext.frame_index += calculated_offset
                # Apply to motion data too
                for motion in camera_data.motion:
                    motion.frame_index += calculated_offset
        
        # Apply coordinate transform if needed
        if coordinate_system != "keep_original":
            camera_data = self._transform_coordinates(camera_data, coordinate_system)
        
        # Generate output path
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == "json":
                output_path = self._export_json(
                    camera_data, filename, include_sparse_points, include_motion
                )
            elif format == "csv":
                output_path = self._export_csv(camera_data, filename, include_motion)
            elif format == "alembic":
                output_path = self._export_alembic(
                    camera_data, filename, include_sparse_points
                )
            elif format == "fbx":
                output_path = self._export_fbx(
                    camera_data, filename, include_sparse_points
                )
            elif format == "nuke_chan":
                output_path = self._export_nuke_chan(camera_data, filename)
            elif format == "opencv_yaml":
                output_path = self._export_opencv_yaml(camera_data, filename)
            elif format == "colmap_text":
                output_path = self._export_colmap_text(camera_data, filename)
            else:
                return ("", f"ERROR: Unknown format: {format}")
            
            status = f"Exported {len(camera_data.extrinsics)} cameras to {format}"
            if calculated_offset != 0:
                status += f" (frame offset: {calculated_offset})"
            return (str(output_path), status)
            
        except Exception as e:
            import traceback
            return ("", f"ERROR: {str(e)}\n{traceback.format_exc()}")
    
    def _transform_coordinates(
        self,
        camera_data: Any,
        target_system: str
    ) -> Any:
        """Transform camera data to target coordinate system."""
        from ..utils import (
            CoordinateSystem, get_coordinate_transform,
            transform_position, transform_rotation_matrix,
            rotation_matrix_to_quaternion
        )
        
        source = CoordinateSystem(camera_data.coordinate_system)
        target = CoordinateSystem(target_system)
        
        if source == target:
            return camera_data
        
        transform = get_coordinate_transform(source, target)
        
        # Transform extrinsics
        for ext in camera_data.extrinsics:
            ext.position = transform_position(ext.position, transform)
            ext.rotation_matrix = transform_rotation_matrix(ext.rotation_matrix, transform)
            ext.quaternion = rotation_matrix_to_quaternion(ext.rotation_matrix)
        
        # Transform sparse points
        for point in camera_data.sparse_points:
            point.xyz = transform_position(point.xyz, transform)
        
        camera_data.coordinate_system = target_system
        return camera_data
    
    def _export_json(
        self,
        camera_data: Any,
        filename: str,
        include_points: bool,
        include_motion: bool
    ) -> Path:
        """Export to JSON format."""
        import json
        
        output_path = OUTPUT_DIR / f"{filename}.json"
        
        data = camera_data.to_dict()
        
        if include_points and camera_data.sparse_points:
            data["sparse_points"] = [p.to_dict() for p in camera_data.sparse_points]
        
        if include_motion and camera_data.motion:
            data["motion"] = [m.to_dict() for m in camera_data.motion]
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_path
    
    def _export_csv(
        self,
        camera_data: Any,
        filename: str,
        include_motion: bool
    ) -> Path:
        """Export to CSV format."""
        output_path = OUTPUT_DIR / f"{filename}.csv"
        
        headers = [
            "frame", "image_name",
            "pos_x", "pos_y", "pos_z",
            "rot_rx", "rot_ry", "rot_rz",
            "quat_w", "quat_x", "quat_y", "quat_z"
        ]
        
        if include_motion and camera_data.motion:
            headers.extend([
                "pan", "tilt", "roll",
                "dolly", "truck", "crane",
                "speed", "motion_type"
            ])
        
        lines = [",".join(headers)]
        
        for ext in camera_data.extrinsics:
            rx, ry, rz = ext.rotation_euler
            quat = ext.quaternion if ext.quaternion is not None else [1, 0, 0, 0]
            
            row = [
                str(ext.frame_index),
                ext.image_name,
                f"{ext.position[0]:.6f}",
                f"{ext.position[1]:.6f}",
                f"{ext.position[2]:.6f}",
                f"{rx:.4f}",
                f"{ry:.4f}",
                f"{rz:.4f}",
                f"{quat[0]:.6f}",
                f"{quat[1]:.6f}",
                f"{quat[2]:.6f}",
                f"{quat[3]:.6f}",
            ]
            
            if include_motion and camera_data.motion:
                motion = camera_data.get_motion(ext.frame_index)
                if motion:
                    row.extend([
                        f"{motion.pan:.4f}",
                        f"{motion.tilt:.4f}",
                        f"{motion.roll:.4f}",
                        f"{motion.dolly:.6f}",
                        f"{motion.truck:.6f}",
                        f"{motion.crane:.6f}",
                        f"{motion.speed:.6f}",
                        motion.motion_type.value,
                    ])
                else:
                    row.extend(["0", "0", "0", "0", "0", "0", "0", "static"])
            
            lines.append(",".join(row))
        
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        
        return output_path
    
    def _export_alembic(
        self,
        camera_data: Any,
        filename: str,
        include_points: bool
    ) -> Path:
        """Export to Alembic format using Blender."""
        import shutil
        import os
        
        output_path = OUTPUT_DIR / f"{filename}.abc"
        
        # Generate Blender Python script
        script = self._generate_blender_alembic_script(
            camera_data, str(output_path), include_points
        )
        
        # Save script
        script_path = OUTPUT_DIR / f"{filename}_export_abc.py"
        with open(script_path, 'w') as f:
            f.write(script)
        
        # Find Blender - check known paths first
        blender_path = self._find_blender()
        
        if blender_path:
            try:
                import subprocess
                print(f"[COLMAP] Running Blender from: {blender_path}")
                result = subprocess.run(
                    [blender_path, "--background", "--python", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0 and output_path.exists():
                    print(f"[COLMAP] Alembic exported successfully: {output_path}")
                    script_path.unlink()  # Cleanup
                    return output_path
                else:
                    print(f"[COLMAP] Blender Alembic export failed: {result.stderr[:500]}")
            except subprocess.TimeoutExpired:
                print("[COLMAP] Blender export timed out")
            except Exception as e:
                print(f"[COLMAP] Blender error: {e}")
        else:
            print("[COLMAP] Blender not found")
        
        # Fallback
        print(f"[COLMAP] Alembic export requires Blender. Script saved to: {script_path}")
        json_path = self._export_json(camera_data, filename, include_points, True)
        print(f"[COLMAP] JSON fallback exported to: {json_path}")
        return script_path
    
    def _find_blender(self) -> Optional[str]:
        """Find Blender executable, checking known paths first."""
        import shutil
        import os
        
        # Known Blender installation paths
        known_paths = [
            # SAM3DBody bundled Blender
            "/workspace/ComfyUI/custom_nodes/ComfyUI-SAM3DBody/lib/blender/blender-4.2.3-linux-x64/blender",
        ]
        
        # Also search SAM3DBody lib folder for any Blender version
        sam3d_blender_dir = Path("/workspace/ComfyUI/custom_nodes/ComfyUI-SAM3DBody/lib/blender")
        if sam3d_blender_dir.exists():
            for blender_dir in sam3d_blender_dir.iterdir():
                if blender_dir.is_dir():
                    potential_blender = blender_dir / "blender"
                    if potential_blender.exists():
                        known_paths.insert(0, str(potential_blender))
        
        # Check known paths
        for path in known_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        
        # Fall back to system PATH
        return shutil.which("blender")
    
    def _generate_blender_alembic_script(
        self,
        camera_data: Any,
        output_path: str,
        include_points: bool
    ) -> str:
        """Generate Blender Python script for Alembic export."""
        import json
        
        # Serialize camera data for the script
        intrinsics = camera_data.intrinsics.to_dict()
        frames_data = []
        for ext in camera_data.extrinsics:
            frames_data.append({
                "frame": ext.frame_index,
                "position": ext.position.tolist(),
                "rotation_matrix": ext.rotation_matrix.tolist(),
            })
        
        points_data = []
        if include_points:
            for p in camera_data.sparse_points[:10000]:  # Limit for performance
                points_data.append({
                    "xyz": p.xyz.tolist(),
                    "rgb": p.rgb.tolist() if p.rgb is not None else [128, 128, 128]
                })
        
        script = f'''
import bpy
import mathutils
import json

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Camera data
intrinsics = {json.dumps(intrinsics)}
frames = {json.dumps(frames_data)}
points = {json.dumps(points_data)}
fps = {camera_data.fps}
output_path = "{output_path}"

# Set scene FPS
bpy.context.scene.render.fps = int(fps)

# Create camera
cam_data = bpy.data.cameras.new("COLMAPCamera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)

# Set camera intrinsics
cam_data.lens = intrinsics.get("focal_length_mm", 35)
cam_data.sensor_width = 36.0  # Full-frame equivalent

# Animate camera
for frame_data in frames:
    frame = frame_data["frame"]
    pos = frame_data["position"]
    rot_matrix = frame_data["rotation_matrix"]
    
    # Set position
    cam_obj.location = mathutils.Vector(pos)
    cam_obj.keyframe_insert(data_path="location", frame=frame)
    
    # Set rotation from matrix
    mat = mathutils.Matrix(rot_matrix)
    cam_obj.rotation_mode = 'QUATERNION'
    cam_obj.rotation_quaternion = mat.to_quaternion()
    cam_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)

# Create point cloud if requested
if points:
    mesh = bpy.data.meshes.new("PointCloud")
    verts = [tuple(p["xyz"]) for p in points]
    mesh.from_pydata(verts, [], [])
    mesh.update()
    
    pc_obj = bpy.data.objects.new("PointCloud", mesh)
    bpy.context.collection.objects.link(pc_obj)
    
    # Add vertex colors
    if not mesh.vertex_colors:
        mesh.vertex_colors.new()

# Set frame range
if frames:
    bpy.context.scene.frame_start = min(f["frame"] for f in frames)
    bpy.context.scene.frame_end = max(f["frame"] for f in frames)

# Export Alembic
bpy.ops.wm.alembic_export(
    filepath=output_path,
    selected=False,
    start=bpy.context.scene.frame_start,
    end=bpy.context.scene.frame_end,
)

print(f"SUCCESS: Exported to {{output_path}}")
'''
        return script
    
    def _export_fbx(
        self,
        camera_data: Any,
        filename: str,
        include_points: bool
    ) -> Path:
        """Export to FBX format using Blender."""
        output_path = OUTPUT_DIR / f"{filename}.fbx"
        script_path = OUTPUT_DIR / f"{filename}_export_fbx.py"
        
        # Generate Blender script with coordinate transforms
        script = self._generate_blender_fbx_script(
            camera_data, str(output_path), include_points
        )
        
        # Always save the script for reference
        with open(script_path, 'w') as f:
            f.write(script)
        
        # Find Blender
        blender_path = self._find_blender()
        
        if blender_path:
            try:
                import subprocess
                print(f"[COLMAP] Running Blender from: {blender_path}")
                result = subprocess.run(
                    [blender_path, "--background", "--python", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                # Always print Blender output for debugging
                if result.stdout:
                    # Filter to show relevant lines
                    for line in result.stdout.split('\n'):
                        if any(kw in line.lower() for kw in ['sparse', 'point', 'mesh', 'vertex', 'locator', 'success', 'error', 'export']):
                            print(f"[Blender] {line}")
                
                if result.returncode == 0 and output_path.exists():
                    print(f"[COLMAP] FBX exported successfully: {output_path}")
                    return output_path
                else:
                    print(f"[COLMAP] Blender export failed (return code: {result.returncode})")
                    if result.stderr:
                        print(f"[COLMAP] Blender stderr: {result.stderr[:500]}")
                    if result.stdout:
                        print(f"[COLMAP] Blender stdout: {result.stdout[:500]}")
            except subprocess.TimeoutExpired:
                print("[COLMAP] Blender export timed out")
            except Exception as e:
                print(f"[COLMAP] Blender error: {e}")
        else:
            print("[COLMAP] Blender not found in PATH or known locations")
        
        # Fallback: return the script path with instructions
        print(f"[COLMAP] FBX export requires Blender. Script saved to: {script_path}")
        print(f"[COLMAP] Run manually: blender --background --python {script_path}")
        
        # Also export JSON as a universal fallback
        json_path = self._export_json(camera_data, filename, include_points, True)
        print(f"[COLMAP] JSON fallback exported to: {json_path}")
        
        return script_path
    
    def _generate_blender_fbx_script(
        self,
        camera_data: Any,
        output_path: str,
        include_points: bool
    ) -> str:
        """Generate Blender script for FBX export with coordinate transforms."""
        import json
        
        coord_sys = getattr(camera_data, 'coordinate_system', 'colmap')
        
        intrinsics = camera_data.intrinsics.to_dict()
        frames_data = []
        for ext in camera_data.extrinsics:
            pos = ext.position.tolist() if hasattr(ext.position, 'tolist') else list(ext.position)
            rot = ext.rotation_matrix.tolist() if hasattr(ext.rotation_matrix, 'tolist') else [list(r) for r in ext.rotation_matrix]
            frames_data.append({
                "frame": ext.frame_index,
                "position": pos,
                "rotation_matrix": rot,
            })
        
        # Include sparse points if requested
        points_data = []
        if include_points and camera_data.sparse_points:
            print(f"[COLMAP FBX] Including {len(camera_data.sparse_points)} sparse points")
            for p in camera_data.sparse_points[:5000]:  # Limit points
                xyz = p.xyz.tolist() if hasattr(p.xyz, 'tolist') else list(p.xyz)
                rgb = [128, 128, 128]
                if p.rgb is not None:
                    rgb = p.rgb.tolist() if hasattr(p.rgb, 'tolist') else list(p.rgb)
                points_data.append({"xyz": xyz, "rgb": rgb})
            print(f"[COLMAP FBX] Serialized {len(points_data)} points for export")
        else:
            print(f"[COLMAP FBX] No sparse points to include (include_points={include_points}, has_points={bool(camera_data.sparse_points)}, count={len(camera_data.sparse_points) if camera_data.sparse_points else 0})")
        
        script = f'''#!/usr/bin/env python3
"""
COLMAP Camera FBX Export Script
Generated by ComfyUI-COLMAP
Run with: blender --background --python {Path(output_path).stem}_export_fbx.py
"""
import bpy
import mathutils
import math

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Camera data from COLMAP
intrinsics = {json.dumps(intrinsics, indent=2)}
frames = {json.dumps(frames_data)}
sparse_points = {json.dumps(points_data)}
fps = {camera_data.fps}
output_path = "{output_path}"
coord_system = "{coord_sys}"

print(f"Exporting {{len(frames)}} camera frames to {{output_path}}")
print(f"Coordinate system: {{coord_system}}")

# Set scene FPS
bpy.context.scene.render.fps = int(fps)
bpy.context.scene.render.resolution_x = intrinsics.get("width", 1920)
bpy.context.scene.render.resolution_y = intrinsics.get("height", 1080)

# Coordinate system transforms
# COLMAP: +X right, +Y down, +Z forward (OpenCV convention)
# Blender: +X right, +Y forward, +Z up
# Maya: +X right, +Y up, -Z forward

def get_transform_matrix(target_system):
    """Get 4x4 transform from COLMAP to target system."""
    if target_system in ("blender", "opengl"):
        # Flip Y and Z
        return mathutils.Matrix((
            (1, 0, 0, 0),
            (0, 0, 1, 0),
            (0, -1, 0, 0),
            (0, 0, 0, 1)
        ))
    elif target_system == "maya":
        # Maya Y-up, Z-forward (toward viewer)
        return mathutils.Matrix((
            (1, 0, 0, 0),
            (0, 0, 1, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 1)
        ))
    elif target_system == "unreal":
        # Unreal: X-forward, Y-right, Z-up, centimeters
        return mathutils.Matrix((
            (0, 100, 0, 0),
            (100, 0, 0, 0),
            (0, 0, 100, 0),
            (0, 0, 0, 1)
        ))
    elif target_system == "unity":
        # Unity: left-handed, Y-up
        return mathutils.Matrix((
            (-1, 0, 0, 0),
            (0, 0, 1, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 1)
        ))
    elif target_system == "houdini":
        # Houdini: Y-up, similar to Maya
        return mathutils.Matrix((
            (1, 0, 0, 0),
            (0, 0, 1, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 1)
        ))
    else:
        # Keep as-is (colmap, opencv)
        return mathutils.Matrix.Identity(4)

# Since we're in Blender, always transform from source to Blender first
# Then FBX exporter handles Blender -> target format
if coord_system == "colmap" or coord_system == "opencv":
    transform = get_transform_matrix("blender")
elif coord_system == "maya":
    # Data already in Maya coords, convert to Blender for FBX export
    # Maya->Blender: swap Y<->Z
    transform = mathutils.Matrix((
        (1, 0, 0, 0),
        (0, 0, -1, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 1)
    ))
else:
    transform = mathutils.Matrix.Identity(4)

# Create camera
cam_data = bpy.data.cameras.new("COLMAPCamera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

# Set camera properties
cam_data.lens = intrinsics.get("focal_length_mm", 35)
cam_data.sensor_width = 36.0  # Full-frame equivalent

# Animate camera
for frame_data in frames:
    frame_num = frame_data["frame"]
    pos = mathutils.Vector(frame_data["position"])
    rot_mat = mathutils.Matrix(frame_data["rotation_matrix"]).to_3x3()
    
    # Apply coordinate transform
    pos = transform @ pos
    rot_mat = transform.to_3x3() @ rot_mat
    
    # Set keyframes
    cam_obj.location = pos
    cam_obj.keyframe_insert(data_path="location", frame=frame_num)
    
    cam_obj.rotation_mode = 'QUATERNION'
    cam_obj.rotation_quaternion = rot_mat.to_quaternion()
    cam_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)

# Set frame range
if frames:
    bpy.context.scene.frame_start = min(f["frame"] for f in frames)
    bpy.context.scene.frame_end = max(f["frame"] for f in frames)

# Add sparse points as point cloud
print(f"sparse_points list length: {{len(sparse_points)}}")
if sparse_points and len(sparse_points) > 0:
    print(f"Adding {{len(sparse_points)}} sparse points to scene")
    
    # Create a parent empty for all points
    points_parent = bpy.data.objects.new("SparsePointsGroup", None)
    points_parent.empty_display_type = 'PLAIN_AXES'
    points_parent.empty_display_size = 0.01
    bpy.context.collection.objects.link(points_parent)
    
    # Create mesh from vertices - this creates a point cloud
    mesh = bpy.data.meshes.new("SparsePointCloud")
    point_obj = bpy.data.objects.new("SparsePoints", mesh)
    point_obj.parent = points_parent
    bpy.context.collection.objects.link(point_obj)
    
    vertices = []
    for i, p in enumerate(sparse_points):
        v = mathutils.Vector(p["xyz"])
        v = transform @ v
        vertices.append(v)
        if i < 3:
            print(f"  Point {{i}}: {{v}}")
    
    print(f"Created {{len(vertices)}} vertices")
    
    # Create mesh - use edges to make points visible in Maya
    # Connect each point to itself (degenerate edge) or create small line segments
    edges = []
    # Create a simple line through first few points for visibility
    for i in range(min(len(vertices) - 1, 1000)):
        edges.append((i, i + 1))
    
    mesh.from_pydata(vertices, edges, [])
    mesh.update()
    
    print(f"Mesh created with {{len(mesh.vertices)}} vertices, {{len(mesh.edges)}} edges")
    
    # Also create locators for a subset of points (more visible in Maya)
    # Limit to 500 locators to avoid performance issues
    locator_count = min(len(sparse_points), 500)
    step = max(1, len(sparse_points) // locator_count)
    
    for i in range(0, len(sparse_points), step):
        if i >= len(sparse_points):
            break
        p = sparse_points[i]
        v = mathutils.Vector(p["xyz"])
        v = transform @ v
        
        # Create empty as locator
        loc = bpy.data.objects.new(f"pt_{{i}}", None)
        loc.empty_display_type = 'PLAIN_AXES'
        loc.empty_display_size = 0.02
        loc.location = v
        loc.parent = points_parent
        bpy.context.collection.objects.link(loc)
    
    print(f"Created {{locator_count}} locators for visibility")
else:
    print("No sparse points to add (list is empty)")

# Export FBX
print(f"Exporting FBX to: {{output_path}}")
bpy.ops.export_scene.fbx(
    filepath=output_path,
    use_selection=False,
    bake_anim=True,
    bake_anim_use_all_bones=False,
    bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=False,
    object_types={{'CAMERA', 'MESH', 'EMPTY'}},
    add_leaf_bones=False,
)

print(f"SUCCESS: Exported {{len(frames)}} camera frames to {{output_path}}")
'''
        return script
    
    def _export_nuke_chan(
        self,
        camera_data: Any,
        filename: str
    ) -> Path:
        """Export to Nuke .chan format."""
        output_path = OUTPUT_DIR / f"{filename}.chan"
        
        # .chan format: frame tx ty tz rx ry rz
        lines = []
        for ext in camera_data.extrinsics:
            rx, ry, rz = ext.rotation_euler
            line = (
                f"{ext.frame_index} "
                f"{ext.position[0]:.6f} {ext.position[1]:.6f} {ext.position[2]:.6f} "
                f"{rx:.4f} {ry:.4f} {rz:.4f}"
            )
            lines.append(line)
        
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        
        return output_path
    
    def _export_opencv_yaml(
        self,
        camera_data: Any,
        filename: str
    ) -> Path:
        """Export to OpenCV YAML format."""
        output_path = OUTPUT_DIR / f"{filename}.yaml"
        
        # OpenCV-style YAML
        lines = ["%YAML:1.0", "---"]
        
        # Intrinsics
        intr = camera_data.intrinsics
        K = intr.to_matrix()
        lines.append("camera_matrix: !!opencv-matrix")
        lines.append("   rows: 3")
        lines.append("   cols: 3")
        lines.append("   dt: d")
        lines.append(f"   data: [{', '.join(f'{x:.6f}' for x in K.flatten())}]")
        
        # Distortion coefficients
        dist = [intr.k1, intr.k2, intr.p1, intr.p2, intr.k3]
        lines.append("distortion_coefficients: !!opencv-matrix")
        lines.append("   rows: 1")
        lines.append("   cols: 5")
        lines.append("   dt: d")
        lines.append(f"   data: [{', '.join(f'{x:.6f}' for x in dist)}]")
        
        # Image size
        lines.append(f"image_width: {intr.width}")
        lines.append(f"image_height: {intr.height}")
        
        # Extrinsics for each frame
        lines.append("frames:")
        for ext in camera_data.extrinsics:
            lines.append(f"  - frame: {ext.frame_index}")
            lines.append(f"    image: \"{ext.image_name}\"")
            lines.append("    rvec: !!opencv-matrix")
            lines.append("       rows: 3")
            lines.append("       cols: 1")
            lines.append("       dt: d")
            # Convert rotation matrix to Rodrigues vector
            import cv2
            rvec, _ = cv2.Rodrigues(ext.rotation_matrix)
            lines.append(f"       data: [{', '.join(f'{x:.6f}' for x in rvec.flatten())}]")
            lines.append("    tvec: !!opencv-matrix")
            lines.append("       rows: 3")
            lines.append("       cols: 1")
            lines.append("       dt: d")
            tvec = -ext.rotation_matrix @ ext.position
            lines.append(f"       data: [{', '.join(f'{x:.6f}' for x in tvec)}]")
        
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        
        return output_path
    
    def _export_colmap_text(
        self,
        camera_data: Any,
        filename: str
    ) -> Path:
        """Export to COLMAP text format."""
        output_dir = OUTPUT_DIR / filename
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # cameras.txt
        intr = camera_data.intrinsics
        camera_line = (
            f"1 PINHOLE {intr.width} {intr.height} "
            f"{intr.focal_length_x} {intr.focal_length_y} "
            f"{intr.principal_point_x} {intr.principal_point_y}"
        )
        with open(output_dir / "cameras.txt", 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"# Number of cameras: 1\n")
            f.write(camera_line + "\n")
        
        # images.txt
        image_lines = []
        for ext in camera_data.extrinsics:
            quat = ext.quaternion if ext.quaternion is not None else [1, 0, 0, 0]
            tvec = -ext.rotation_matrix @ ext.position
            line = (
                f"{ext.frame_index + 1} "
                f"{quat[0]} {quat[1]} {quat[2]} {quat[3]} "
                f"{tvec[0]} {tvec[1]} {tvec[2]} "
                f"1 {ext.image_name}"
            )
            image_lines.append(line)
            image_lines.append("")  # Empty line for 2D points (not exported)
        
        with open(output_dir / "images.txt", 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write("\n".join(image_lines))
        
        # points3D.txt
        point_lines = []
        for i, p in enumerate(camera_data.sparse_points):
            rgb = p.rgb if p.rgb is not None else [128, 128, 128]
            line = (
                f"{i + 1} "
                f"{p.xyz[0]} {p.xyz[1]} {p.xyz[2]} "
                f"{rgb[0]} {rgb[1]} {rgb[2]} "
                f"{p.error}"
            )
            point_lines.append(line)
        
        with open(output_dir / "points3D.txt", 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
            f.write("\n".join(point_lines))
        
        return output_dir
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
