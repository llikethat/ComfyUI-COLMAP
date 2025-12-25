"""
COLMAP Camera Visualizer Node

Generates visual previews of camera tracking data within ComfyUI:
- Camera path visualization (top/side/3D views)
- Sparse point cloud preview
- Motion analysis charts
"""

import numpy as np
from typing import Tuple, Any, Optional, List
import torch


class COLMAPCameraVisualizer:
    """
    Visualize camera tracking data as images.
    
    Creates visual representations of:
    - Camera trajectory (path through 3D space)
    - Camera frustums showing view direction
    - Sparse point cloud
    - Motion graphs
    """
    
    CATEGORY = "COLMAP"
    FUNCTION = "visualize"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("trajectory_view", "motion_graph", "stats")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_data": ("CAMERA_DATA",),
            },
            "optional": {
                "view_type": (["top", "side", "front", "perspective"], {
                    "default": "perspective",
                    "tooltip": "Camera view angle for trajectory"
                }),
                "image_width": ("INT", {
                    "default": 800,
                    "min": 200,
                    "max": 2048,
                }),
                "image_height": ("INT", {
                    "default": 600,
                    "min": 200,
                    "max": 2048,
                }),
                "show_frustums": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show camera frustum cones"
                }),
                "show_points": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show sparse 3D points"
                }),
                "path_color": (["red", "green", "blue", "yellow", "cyan", "magenta", "white"], {
                    "default": "cyan"
                }),
                "background": (["black", "white", "gray"], {
                    "default": "black"
                }),
            }
        }
    
    def visualize(
        self,
        camera_data: Any,
        view_type: str = "perspective",
        image_width: int = 800,
        image_height: int = 600,
        show_frustums: bool = True,
        show_points: bool = True,
        path_color: str = "cyan",
        background: str = "black"
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Generate visualization of camera data."""
        
        if not camera_data or not camera_data.extrinsics:
            # Return empty images
            empty = torch.zeros((1, image_height, image_width, 3), dtype=torch.float32)
            return (empty, empty, "No camera data available")
        
        # Generate trajectory visualization
        trajectory_img = self._render_trajectory(
            camera_data, view_type, image_width, image_height,
            show_frustums, show_points, path_color, background
        )
        
        # Generate motion graph
        motion_img = self._render_motion_graph(
            camera_data, image_width, image_height // 2, background
        )
        
        # Generate stats string
        stats = self._generate_stats(camera_data)
        
        return (trajectory_img, motion_img, stats)
    
    def _get_color(self, name: str) -> Tuple[int, int, int]:
        """Get RGB color from name."""
        colors = {
            "red": (255, 80, 80),
            "green": (80, 255, 80),
            "blue": (80, 80, 255),
            "yellow": (255, 255, 80),
            "cyan": (80, 255, 255),
            "magenta": (255, 80, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "gray": (128, 128, 128),
        }
        return colors.get(name, (255, 255, 255))
    
    def _render_trajectory(
        self,
        camera_data: Any,
        view_type: str,
        width: int,
        height: int,
        show_frustums: bool,
        show_points: bool,
        path_color: str,
        background: str
    ) -> torch.Tensor:
        """Render camera trajectory as image."""
        
        # Create image
        bg_color = self._get_color(background)
        img = np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        # Extract camera positions
        positions = []
        directions = []
        for ext in camera_data.extrinsics:
            positions.append(ext.position)
            # Camera looks along -Z in camera space, transform to world
            R = ext.rotation_matrix
            look_dir = -R[:, 2]  # Third column, negated
            directions.append(look_dir)
        
        if not positions:
            return self._numpy_to_tensor(img)
        
        positions = np.array(positions)
        directions = np.array(directions)
        
        # Get sparse points if available
        points_3d = []
        if show_points and camera_data.sparse_points:
            for p in camera_data.sparse_points[:2000]:  # Limit for performance
                points_3d.append(p.xyz)
            points_3d = np.array(points_3d) if points_3d else None
        else:
            points_3d = None
        
        # Project to 2D based on view type
        if view_type == "top":
            # X-Z plane (bird's eye view)
            proj_pos = positions[:, [0, 2]]
            proj_dir = directions[:, [0, 2]]
            if points_3d is not None:
                proj_points = points_3d[:, [0, 2]]
            else:
                proj_points = None
        elif view_type == "side":
            # X-Y plane
            proj_pos = positions[:, [0, 1]]
            proj_dir = directions[:, [0, 1]]
            if points_3d is not None:
                proj_points = points_3d[:, [0, 1]]
            else:
                proj_points = None
        elif view_type == "front":
            # Y-Z plane
            proj_pos = positions[:, [2, 1]]
            proj_dir = directions[:, [2, 1]]
            if points_3d is not None:
                proj_points = points_3d[:, [2, 1]]
            else:
                proj_points = None
        else:  # perspective
            # Simple perspective projection
            proj_pos = self._perspective_project(positions)
            proj_dir = self._perspective_project(positions + directions * 0.5) - proj_pos
            if points_3d is not None:
                proj_points = self._perspective_project(points_3d)
            else:
                proj_points = None
        
        # Normalize to image coordinates
        all_points = [proj_pos]
        if proj_points is not None:
            all_points.append(proj_points)
        all_pts = np.vstack(all_points)
        
        min_pt = all_pts.min(axis=0)
        max_pt = all_pts.max(axis=0)
        range_pt = max_pt - min_pt
        range_pt[range_pt == 0] = 1  # Avoid division by zero
        
        margin = 50
        scale = min((width - 2*margin) / range_pt[0], (height - 2*margin) / range_pt[1])
        
        def to_pixel(pt):
            normalized = (pt - min_pt) * scale
            return (int(margin + normalized[0]), int(height - margin - normalized[1]))
        
        # Draw sparse points first (background)
        if proj_points is not None:
            point_color = (60, 60, 60) if background == "black" else (200, 200, 200)
            for pt in proj_points:
                px, py = to_pixel(pt)
                if 0 <= px < width and 0 <= py < height:
                    img[py, px] = point_color
        
        # Draw camera path
        path_rgb = self._get_color(path_color)
        for i in range(len(proj_pos) - 1):
            p1 = to_pixel(proj_pos[i])
            p2 = to_pixel(proj_pos[i + 1])
            self._draw_line(img, p1, p2, path_rgb, thickness=2)
        
        # Draw camera positions and frustums
        for i, (pos, direction) in enumerate(zip(proj_pos, proj_dir)):
            px, py = to_pixel(pos)
            
            # Camera position marker
            t = i / max(len(proj_pos) - 1, 1)
            marker_color = (
                int(255 * (1-t) + 80 * t),
                int(80 * (1-t) + 255 * t),
                int(80)
            )
            self._draw_circle(img, (px, py), 4, marker_color, filled=True)
            
            # Frustum direction
            if show_frustums:
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 0:
                    dir_normalized = direction / dir_norm * 20
                    end_pt = (int(px + dir_normalized[0]), int(py - dir_normalized[1]))
                    self._draw_line(img, (px, py), end_pt, (200, 200, 50), thickness=1)
        
        # Draw frame numbers for first and last
        if positions.shape[0] > 0:
            start_frame = camera_data.extrinsics[0].frame_index
            end_frame = camera_data.extrinsics[-1].frame_index
            
            self._draw_text(img, f"F{start_frame}", to_pixel(proj_pos[0]), (255, 255, 255))
            self._draw_text(img, f"F{end_frame}", to_pixel(proj_pos[-1]), (255, 255, 255))
        
        # Draw legend
        self._draw_text(img, f"View: {view_type.upper()}", (10, 20), (200, 200, 200))
        self._draw_text(img, f"Frames: {len(positions)}", (10, 40), (200, 200, 200))
        
        return self._numpy_to_tensor(img)
    
    def _perspective_project(self, points: np.ndarray, fov: float = 60) -> np.ndarray:
        """Simple perspective projection."""
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        # Camera at origin looking down -Z
        # Rotate points to get a nice view angle
        angle_y = np.radians(30)
        angle_x = np.radians(20)
        
        Ry = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])
        
        rotated = (Rx @ Ry @ points.T).T
        
        # Move camera back
        center = rotated.mean(axis=0)
        rotated = rotated - center
        
        # Project
        z = rotated[:, 2] + 5  # Offset to avoid division by zero
        z[z < 0.1] = 0.1
        
        f = 1.0 / np.tan(np.radians(fov / 2))
        x = rotated[:, 0] * f / z
        y = rotated[:, 1] * f / z
        
        return np.column_stack([x, y])
    
    def _render_motion_graph(
        self,
        camera_data: Any,
        width: int,
        height: int,
        background: str
    ) -> torch.Tensor:
        """Render motion analysis as graph."""
        
        bg_color = self._get_color(background)
        img = np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        if len(camera_data.extrinsics) < 2:
            return self._numpy_to_tensor(img)
        
        # Calculate motion metrics
        positions = np.array([ext.position for ext in camera_data.extrinsics])
        frames = [ext.frame_index for ext in camera_data.extrinsics]
        
        # Speed (distance between consecutive frames)
        speeds = []
        for i in range(1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[i-1])
            speeds.append(dist)
        
        if not speeds:
            return self._numpy_to_tensor(img)
        
        speeds = np.array(speeds)
        max_speed = speeds.max() if speeds.max() > 0 else 1
        
        # Draw graph
        margin = 40
        graph_width = width - 2 * margin
        graph_height = height - 2 * margin
        
        # Axes
        axis_color = (150, 150, 150)
        self._draw_line(img, (margin, height - margin), (width - margin, height - margin), axis_color)
        self._draw_line(img, (margin, margin), (margin, height - margin), axis_color)
        
        # Plot speed
        points = []
        for i, speed in enumerate(speeds):
            x = margin + int(i / max(len(speeds) - 1, 1) * graph_width)
            y = height - margin - int(speed / max_speed * graph_height * 0.9)
            points.append((x, y))
        
        # Draw line graph
        line_color = (80, 255, 80)
        for i in range(len(points) - 1):
            self._draw_line(img, points[i], points[i+1], line_color, thickness=2)
        
        # Labels
        self._draw_text(img, "Camera Speed", (width // 2 - 40, 15), (200, 200, 200))
        self._draw_text(img, f"Max: {max_speed:.2f}", (width - 100, 15), (150, 150, 150))
        self._draw_text(img, "Frame", (width // 2, height - 10), (150, 150, 150))
        
        return self._numpy_to_tensor(img)
    
    def _generate_stats(self, camera_data: Any) -> str:
        """Generate statistics string."""
        stats = []
        stats.append(f"=== Camera Tracking Stats ===")
        stats.append(f"Registered frames: {len(camera_data.extrinsics)} / {camera_data.total_frames}")
        stats.append(f"Sparse points: {len(camera_data.sparse_points)}")
        stats.append(f"FPS: {camera_data.fps}")
        
        if camera_data.intrinsics:
            intr = camera_data.intrinsics
            stats.append(f"\nIntrinsics:")
            stats.append(f"  Resolution: {intr.width} x {intr.height}")
            
            # Focal length - try different attribute names
            focal_mm = getattr(intr, 'focal_length_mm', None)
            if focal_mm is None:
                # Calculate from pixels if available
                fx = getattr(intr, 'focal_length_x', getattr(intr, 'fx', None))
                if fx and intr.width:
                    # Assume 36mm sensor width for mm conversion
                    focal_mm = fx * 36.0 / intr.width
            if focal_mm:
                stats.append(f"  Focal length: {focal_mm:.1f}mm")
            
            # FOV - calculate if not present
            fov_h = getattr(intr, 'fov_h', None)
            fov_v = getattr(intr, 'fov_v', None)
            if fov_h is None or fov_v is None:
                fx = getattr(intr, 'focal_length_x', getattr(intr, 'fx', None))
                fy = getattr(intr, 'focal_length_y', getattr(intr, 'fy', fx))
                if fx and intr.width:
                    import math
                    fov_h = 2 * math.degrees(math.atan(intr.width / (2 * fx)))
                    fov_v = 2 * math.degrees(math.atan(intr.height / (2 * fy))) if fy else fov_h * intr.height / intr.width
            if fov_h and fov_v:
                stats.append(f"  FOV: {fov_h:.1f}° x {fov_v:.1f}°")
        
        if camera_data.extrinsics:
            positions = np.array([ext.position for ext in camera_data.extrinsics])
            total_dist = 0
            for i in range(1, len(positions)):
                total_dist += np.linalg.norm(positions[i] - positions[i-1])
            
            stats.append(f"\nMotion:")
            stats.append(f"  Total travel: {total_dist:.2f} units")
            stats.append(f"  Bounding box: {positions.max(axis=0) - positions.min(axis=0)}")
        
        if camera_data.mean_reprojection_error:
            stats.append(f"\nQuality:")
            stats.append(f"  Mean reprojection error: {camera_data.mean_reprojection_error:.3f}px")
        
        return "\n".join(stats)
    
    def _draw_line(
        self,
        img: np.ndarray,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 1
    ):
        """Draw line using Bresenham's algorithm."""
        x1, y1 = p1
        x2, y2 = p2
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        h, w = img.shape[:2]
        
        while True:
            # Draw with thickness
            for ty in range(-thickness//2, thickness//2 + 1):
                for tx in range(-thickness//2, thickness//2 + 1):
                    px, py = x1 + tx, y1 + ty
                    if 0 <= px < w and 0 <= py < h:
                        img[py, px] = color
            
            if x1 == x2 and y1 == y2:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
    
    def _draw_circle(
        self,
        img: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        color: Tuple[int, int, int],
        filled: bool = False
    ):
        """Draw circle."""
        cx, cy = center
        h, w = img.shape[:2]
        
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if filled:
                    if dist <= radius and 0 <= x < w and 0 <= y < h:
                        img[y, x] = color
                else:
                    if abs(dist - radius) < 1 and 0 <= x < w and 0 <= y < h:
                        img[y, x] = color
    
    def _draw_text(
        self,
        img: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        color: Tuple[int, int, int]
    ):
        """Draw simple text (basic implementation)."""
        # Simple 3x5 font for digits and basic chars
        # For production, use PIL or cv2.putText
        x, y = pos
        h, w = img.shape[:2]
        
        # Just draw a colored rectangle as placeholder
        # In production, use proper font rendering
        text_width = len(text) * 6
        if 0 <= x < w and 0 <= y < h:
            for i, char in enumerate(text):
                cx = x + i * 6
                if 0 <= cx < w - 5:
                    # Simple dot pattern for visibility
                    for dy in range(5):
                        for dx in range(4):
                            if 0 <= y + dy < h and 0 <= cx + dx < w:
                                # Create simple patterns
                                if (dx + dy) % 2 == 0:
                                    img[y + dy, cx + dx] = color
    
    def _numpy_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert numpy image to ComfyUI tensor format."""
        # Normalize to 0-1 float
        img_float = img.astype(np.float32) / 255.0
        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        return torch.from_numpy(img_float).unsqueeze(0)


# Node registration
NODE_CLASS_MAPPINGS = {
    "COLMAPCameraVisualizer": COLMAPCameraVisualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "COLMAPCameraVisualizer": "COLMAP Camera Visualizer",
}
