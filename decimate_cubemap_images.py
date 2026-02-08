#!/usr/bin/env python3
"""
Cubemap Image Decimation Tool for 3DGS Training

Intelligently decimates cubemap images based on view frustum point cloud overlap
and camera baseline distance. Frames with high overlap and small baseline are
considered redundant and can be removed.

Algorithm (改善案A):
1. Group images by frame (6 directions per frame: top/front/right/back/left/bottom)
2. For each frame:
   a. Get point cloud IDs visible in each direction's view frustum
   b. Calculate overlap ratio with previous frame (per direction)
   c. Calculate camera baseline distance
   d. If overlap > threshold AND baseline < threshold → mark as redundant
3. Ensure minimum observation count for each 3D point

Usage:
    python decimate_cubemap_images.py \
        --input ./colmap_dataset/ \
        --output ./decimated_dataset/ \
        --overlap-threshold 0.8 \
        --baseline-threshold 0.1 \
        --min-observations 3

Dependencies:
    pip install numpy open3d
"""

import argparse
import pickle
import re
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    camera_id: int
    model: str
    width: int
    height: int
    params: List[float]  # [fx, fy, cx, cy] for PINHOLE
    
    @property
    def fx(self) -> float:
        return self.params[0]
    
    @property
    def fy(self) -> float:
        return self.params[1]
    
    @property
    def cx(self) -> float:
        return self.params[2]
    
    @property
    def cy(self) -> float:
        return self.params[3]


@dataclass
class ImagePose:
    """Image pose (extrinsics) from COLMAP images.txt."""
    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str
    
    @property
    def quaternion(self) -> np.ndarray:
        """Return quaternion as [qw, qx, qy, qz]."""
        return np.array([self.qw, self.qx, self.qy, self.qz])
    
    @property
    def translation(self) -> np.ndarray:
        """Return translation vector (world-to-camera)."""
        return np.array([self.tx, self.ty, self.tz])
    
    @property
    def rotation_matrix(self) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix (world-to-camera)."""
        q = self.quaternion
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
        ])
        return R
    
    @property
    def camera_position(self) -> np.ndarray:
        """Get camera position in world coordinates."""
        R = self.rotation_matrix
        t = self.translation
        return -R.T @ t


@dataclass
class Frame:
    """A frame consisting of 6 cubemap directions."""
    frame_name: str
    images: Dict[str, ImagePose] = field(default_factory=dict)  # direction -> ImagePose
    visible_points: Dict[str, Set[int]] = field(default_factory=dict)  # direction -> set of point IDs
    
    @property
    def camera_position(self) -> np.ndarray:
        """Get camera position (same for all directions in a frame)."""
        if self.images:
            return next(iter(self.images.values())).camera_position
        return np.zeros(3)
    
    @property
    def all_visible_points(self) -> Set[int]:
        """Get all visible points across all directions."""
        all_points = set()
        for points in self.visible_points.values():
            all_points.update(points)
        return all_points


def parse_cameras_txt(cameras_path: Path) -> Dict[int, CameraIntrinsics]:
    """Parse COLMAP cameras.txt file."""
    cameras = {}
    with open(cameras_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[camera_id] = CameraIntrinsics(
                camera_id=camera_id,
                model=model,
                width=width,
                height=height,
                params=params,
            )
    return cameras


def parse_images_txt(images_path: Path) -> Dict[int, ImagePose]:
    """Parse COLMAP images.txt file."""
    images = {}
    with open(images_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        
        parts = line.split()
        if len(parts) >= 10:
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = int(parts[8])
            name = parts[9]
            
            images[image_id] = ImagePose(
                image_id=image_id,
                qw=qw, qx=qx, qy=qy, qz=qz,
                tx=tx, ty=ty, tz=tz,
                camera_id=camera_id,
                name=name,
            )
            i += 2  # Skip the POINTS2D line
        else:
            i += 1
    
    return images


def parse_points3d_txt(points3d_path: Path) -> np.ndarray:
    """Parse COLMAP points3D.txt and return Nx3 array of points."""
    points = []
    point_ids = []
    with open(points3d_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                point_id = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                points.append([x, y, z])
                point_ids.append(point_id)
    return np.array(points), np.array(point_ids)


def load_points3d_ply(ply_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load points from PLY file. Returns (points, point_ids)."""
    if not HAS_OPEN3D:
        raise ImportError("open3d is required to load PLY files")
    
    pc = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pc.points)
    point_ids = np.arange(1, len(points) + 1)  # Generate sequential IDs
    return points, point_ids


def get_visible_points_in_frustum(
    points: np.ndarray,
    point_ids: np.ndarray,
    camera: CameraIntrinsics,
    image_pose: ImagePose,
    fov_margin: float = 0.0,
    min_depth: float = 0.1,
    max_depth: float = 100.0,
) -> Set[int]:
    """
    Get point IDs visible within the camera's view frustum.
    
    Args:
        points: Nx3 array of 3D points in world coordinates
        point_ids: Array of point IDs
        camera: Camera intrinsics
        image_pose: Camera pose (extrinsics)
        fov_margin: Extra margin around FoV (in pixels)
        min_depth: Minimum depth for visibility
        max_depth: Maximum depth for visibility
    
    Returns:
        Set of visible point IDs
    """
    if len(points) == 0:
        return set()
    
    # Transform points to camera coordinates
    R = image_pose.rotation_matrix
    t = image_pose.translation
    
    # World to camera: p_cam = R @ p_world + t
    points_cam = (R @ points.T).T + t
    
    # Filter by depth
    z = points_cam[:, 2]
    depth_mask = (z > min_depth) & (z < max_depth)
    
    # Project to image plane
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    
    u = camera.fx * (x / z) + camera.cx
    v = camera.fy * (y / z) + camera.cy
    
    # Check if within image bounds (with margin)
    margin = fov_margin
    in_bounds = (
        (u >= -margin) & (u < camera.width + margin) &
        (v >= -margin) & (v < camera.height + margin)
    )
    
    # Combine masks
    visible_mask = depth_mask & in_bounds
    
    # Return visible point IDs
    visible_ids = set(point_ids[visible_mask])
    return visible_ids


def compute_visible_points_worker(
    task: Tuple[str, str, int, Dict[str, Any], np.ndarray, np.ndarray, float]
) -> Tuple[str, str, Set[int]]:
    """
    Worker function for parallel visible points computation.
    
    Args:
        task: Tuple of (frame_name, direction, camera_id, camera_dict, 
              image_pose_dict, points, point_ids, max_depth)
    
    Returns:
        Tuple of (frame_name, direction, visible_point_ids)
    """
    frame_name, direction, camera_dict, image_pose_dict, points, point_ids, max_depth = task
    
    # Reconstruct camera intrinsics
    camera = CameraIntrinsics(
        camera_id=camera_dict["camera_id"],
        model=camera_dict["model"],
        width=camera_dict["width"],
        height=camera_dict["height"],
        params=camera_dict["params"],
    )
    
    # Reconstruct image pose
    image_pose = ImagePose(
        image_id=image_pose_dict["image_id"],
        qw=image_pose_dict["qw"],
        qx=image_pose_dict["qx"],
        qy=image_pose_dict["qy"],
        qz=image_pose_dict["qz"],
        tx=image_pose_dict["tx"],
        ty=image_pose_dict["ty"],
        tz=image_pose_dict["tz"],
        camera_id=image_pose_dict["camera_id"],
        name=image_pose_dict["name"],
    )
    
    # Calculate visible points
    visible = get_visible_points_in_frustum(
        points, point_ids, camera, image_pose,
        max_depth=max_depth,
    )
    
    return (frame_name, direction, visible)


def image_pose_to_dict(image_pose: ImagePose) -> Dict[str, Any]:
    """Convert ImagePose to serializable dict for multiprocessing."""
    return {
        "image_id": image_pose.image_id,
        "qw": image_pose.qw,
        "qx": image_pose.qx,
        "qy": image_pose.qy,
        "qz": image_pose.qz,
        "tx": image_pose.tx,
        "ty": image_pose.ty,
        "tz": image_pose.tz,
        "camera_id": image_pose.camera_id,
        "name": image_pose.name,
    }


def camera_to_dict(camera: CameraIntrinsics) -> Dict[str, Any]:
    """Convert CameraIntrinsics to serializable dict for multiprocessing."""
    return {
        "camera_id": camera.camera_id,
        "model": camera.model,
        "width": camera.width,
        "height": camera.height,
        "params": camera.params,
    }


def extract_frame_name(image_name: str) -> Tuple[str, str]:
    """
    Extract frame name and direction from image name.
    
    Expected format: {frame_name}_{direction}.jpg
    E.g., "frame_0001_front.jpg" -> ("frame_0001", "front")
    """
    # Match pattern: anything_direction.ext where direction is one of the cubemap directions
    directions = ["top", "front", "right", "back", "left", "bottom"]
    
    stem = Path(image_name).stem
    for direction in directions:
        if stem.endswith(f"_{direction}"):
            frame_name = stem[:-len(direction)-1]
            return frame_name, direction
    
    # Fallback: treat entire stem as frame name
    return stem, "unknown"


def group_images_by_frame(images: Dict[int, ImagePose]) -> Dict[str, Frame]:
    """Group images by frame name."""
    frames: Dict[str, Frame] = {}
    
    for image_id, image_pose in images.items():
        frame_name, direction = extract_frame_name(image_pose.name)
        
        if frame_name not in frames:
            frames[frame_name] = Frame(frame_name=frame_name)
        
        frames[frame_name].images[direction] = image_pose
    
    return frames


def calculate_overlap_ratio(
    points_curr: Set[int],
    points_prev: Set[int],
) -> float:
    """
    Calculate overlap ratio between two sets of visible points.
    
    Returns: |intersection| / |curr| if curr is not empty, else 0.0
    """
    if not points_curr:
        return 0.0
    
    intersection = points_curr & points_prev
    return len(intersection) / len(points_curr)


def calculate_frame_redundancy(
    frame_curr: Frame,
    frame_prev: Frame,
    directions: List[str],
) -> Tuple[float, float, Dict[str, float]]:
    """
    Calculate redundancy metrics between two frames.
    
    Returns:
        - average_overlap: Average overlap ratio across all directions
        - baseline_distance: Distance between camera positions
        - per_direction_overlap: Dict of overlap ratios per direction
    """
    per_direction_overlap = {}
    valid_overlaps = []
    
    for direction in directions:
        if direction in frame_curr.visible_points and direction in frame_prev.visible_points:
            overlap = calculate_overlap_ratio(
                frame_curr.visible_points[direction],
                frame_prev.visible_points[direction],
            )
            per_direction_overlap[direction] = overlap
            valid_overlaps.append(overlap)
    
    average_overlap = np.mean(valid_overlaps) if valid_overlaps else 0.0
    baseline_distance = np.linalg.norm(frame_curr.camera_position - frame_prev.camera_position)
    
    return average_overlap, baseline_distance, per_direction_overlap


def count_point_observations(
    frames: Dict[str, Frame],
    kept_frame_names: Set[str],
) -> Dict[int, int]:
    """
    Count how many times each point is observed across kept frames.
    
    Returns: Dict mapping point_id -> observation_count
    """
    point_counts: Dict[int, int] = defaultdict(int)
    
    for frame_name in kept_frame_names:
        frame = frames[frame_name]
        for point_id in frame.all_visible_points:
            point_counts[point_id] += 1
    
    return point_counts


def compute_frame_redundancy_worker(
    task: Tuple[int, str, Dict[str, Set[int]], np.ndarray, List[str], List[Tuple[str, Dict[str, Set[int]], np.ndarray]]]
) -> Tuple[int, str, float, float]:
    """
    Worker function for parallel frame redundancy computation.
    
    Args:
        task: (frame_index, frame_name, visible_points, camera_pos, directions, 
               list of (other_name, other_visible_points, other_camera_pos))
    
    Returns:
        (frame_index, frame_name, max_overlap, min_baseline)
    """
    frame_idx, frame_name, visible_points, camera_pos, directions, others = task
    
    max_overlap = 0.0
    min_baseline = float('inf')
    
    for other_name, other_visible_points, other_camera_pos in others:
        # Calculate overlap per direction
        valid_overlaps = []
        for direction in directions:
            if direction in visible_points and direction in other_visible_points:
                curr_points = visible_points[direction]
                other_points = other_visible_points[direction]
                if curr_points:
                    intersection = curr_points & other_points
                    overlap = len(intersection) / len(curr_points)
                    valid_overlaps.append(overlap)
        
        avg_overlap = np.mean(valid_overlaps) if valid_overlaps else 0.0
        baseline = np.linalg.norm(camera_pos - other_camera_pos)
        
        max_overlap = max(max_overlap, avg_overlap)
        min_baseline = min(min_baseline, baseline)
    
    if min_baseline == float('inf'):
        min_baseline = 0.0
    
    return (frame_idx, frame_name, max_overlap, min_baseline)


def decimate_frames(
    frames: Dict[str, Frame],
    directions: List[str],
    overlap_threshold: float = 0.8,
    baseline_threshold: float = 0.1,
    min_observations: int = 3,
    window_size: int = 1,
    verbose: bool = True,
    show_stats: bool = True,
) -> Tuple[Set[str], Set[str]]:
    """
    Decimate frames based on redundancy while ensuring minimum observation coverage.
    
    Args:
        frames: Dict of frame_name -> Frame
        directions: List of direction names
        overlap_threshold: Frames with overlap > this are candidates for removal
        baseline_threshold: Frames with baseline < this are candidates for removal
        min_observations: Each point should be observed at least this many times
        window_size: Compare each frame with this many frames before/after (1=adjacent only)
        verbose: Print progress
        show_stats: Show overlap and baseline statistics for threshold tuning
    
    Returns:
        - kept_frames: Set of frame names to keep
        - removed_frames: Set of frame names to remove
    """
    # Sort frames by name to process in order
    sorted_frame_names = sorted(frames.keys())
    
    if len(sorted_frame_names) == 0:
        return set(), set()
    
    # Start with all frames
    kept_frames: Set[str] = set(sorted_frame_names)
    removed_frames: Set[str] = set()
    
    if verbose:
        print(f"Starting decimation with {len(sorted_frame_names)} frames")
        print(f"  Overlap threshold: {overlap_threshold}")
        print(f"  Baseline threshold: {baseline_threshold}")
        print(f"  Min observations: {min_observations}")
        print(f"  Window size: {window_size} (compare with +/-{window_size} frames)")
    
    # First pass: calculate all redundancy metrics and identify candidates
    # Compare each frame with frames within the window
    all_metrics: List[Tuple[str, float, float]] = []  # (frame_name, max_overlap, min_baseline)
    redundant_candidates: List[Tuple[str, float, float]] = []  # (frame_name, max_overlap, min_baseline)
    
    if verbose:
        print(f"  Comparing {len(sorted_frame_names)} frames (window_size={window_size})...")
    
    for i in range(len(sorted_frame_names)):
        frame_curr = frames[sorted_frame_names[i]]
        
        # Compare with frames within window (before and after)
        max_overlap = 0.0
        min_baseline = float('inf')
        
        for offset in range(-window_size, window_size + 1):
            if offset == 0:
                continue  # Skip self
            j = i + offset
            if j < 0 or j >= len(sorted_frame_names):
                continue  # Out of bounds
            
            frame_other = frames[sorted_frame_names[j]]
            
            # Calculate overlap per direction
            valid_overlaps = []
            for direction in directions:
                if direction in frame_curr.visible_points and direction in frame_other.visible_points:
                    curr_points = frame_curr.visible_points[direction]
                    other_points = frame_other.visible_points[direction]
                    if curr_points:
                        intersection = curr_points & other_points
                        overlap = len(intersection) / len(curr_points)
                        valid_overlaps.append(overlap)
            
            avg_overlap = np.mean(valid_overlaps) if valid_overlaps else 0.0
            baseline = np.linalg.norm(frame_curr.camera_position - frame_other.camera_position)
            
            max_overlap = max(max_overlap, avg_overlap)
            min_baseline = min(min_baseline, baseline)
        
        if min_baseline == float('inf'):
            min_baseline = 0.0
        
        all_metrics.append((sorted_frame_names[i], max_overlap, min_baseline))
        
        # Check if frame is redundant
        if max_overlap > overlap_threshold and min_baseline < baseline_threshold:
            redundant_candidates.append((sorted_frame_names[i], max_overlap, min_baseline))
        
        # Progress
        if verbose and (i + 1) % 100 == 0:
            print(f"    Progress: {i+1}/{len(sorted_frame_names)} frames ({(i+1)*100//len(sorted_frame_names)}%)")
    
    if verbose:
        print(f"    Progress: {len(sorted_frame_names)}/{len(sorted_frame_names)} frames (100%)")
    
    # Show statistics for threshold tuning
    if show_stats and all_metrics:
        overlaps = [m[1] for m in all_metrics]
        baselines = [m[2] for m in all_metrics]
        
        print(f"\n=== Frame Pair Statistics (for threshold tuning) ===")
        print(f"  Overlap ratio (higher = more similar):")
        print(f"    Min: {min(overlaps):.4f}, Max: {max(overlaps):.4f}, Avg: {np.mean(overlaps):.4f}")
        print(f"    Percentiles: 25%={np.percentile(overlaps, 25):.4f}, 50%={np.percentile(overlaps, 50):.4f}, 75%={np.percentile(overlaps, 75):.4f}")
        print(f"  Baseline distance (lower = camera moved less):")
        print(f"    Min: {min(baselines):.4f}, Max: {max(baselines):.4f}, Avg: {np.mean(baselines):.4f}")
        print(f"    Percentiles: 25%={np.percentile(baselines, 25):.4f}, 50%={np.percentile(baselines, 50):.4f}, 75%={np.percentile(baselines, 75):.4f}")
        
        # Show sample frame pairs
        print(f"\n  Sample frame pairs (first 5):")
        for frame_name, overlap, baseline in all_metrics[:5]:
            status = "REDUNDANT" if overlap > overlap_threshold and baseline < baseline_threshold else ""
            print(f"    {frame_name}: overlap={overlap:.4f}, baseline={baseline:.4f} {status}")
        
        # Count how many would be removed at different thresholds
        print(f"\n  Threshold analysis:")
        # Generate baseline thresholds relative to config value
        bl_thresholds = [
            baseline_threshold / 8,
            baseline_threshold / 4,
            baseline_threshold / 2,
            baseline_threshold / 1,
            baseline_threshold * 2,
            baseline_threshold * 4,
            baseline_threshold * 8,
        ]
        for ov_th in [0.3, 0.5, 0.7, 0.9]:
            for bl_th in bl_thresholds:
                count = sum(1 for _, ov, bl in all_metrics if ov > ov_th and bl < bl_th)
                if count > 0:
                    print(f"    overlap>{ov_th}, baseline<{bl_th:.4f}: {count} frames ({count/len(sorted_frame_names)*100:.1f}%)")
        print()
    
    if verbose:
        print(f"Found {len(redundant_candidates)} redundant candidates")
    
    # Second pass: remove redundant frames while ensuring coverage
    # Sort by overlap (highest first) - remove most redundant first
    redundant_candidates.sort(key=lambda x: x[1], reverse=True)
    
    for frame_name, overlap, baseline in redundant_candidates:
        # Temporarily remove frame
        test_kept = kept_frames - {frame_name}
        
        # Check if removal would violate min_observations constraint
        point_counts = count_point_observations(frames, test_kept)
        
        # Get all points visible from the frame being removed
        frame = frames[frame_name]
        all_points_in_frame = frame.all_visible_points
        
        # Check if any point would fall below min_observations
        can_remove = True
        for point_id in all_points_in_frame:
            if point_counts.get(point_id, 0) < min_observations:
                can_remove = False
                break
        
        if can_remove:
            kept_frames.remove(frame_name)
            removed_frames.add(frame_name)
            if verbose:
                print(f"  Removing frame '{frame_name}' (overlap={overlap:.3f}, baseline={baseline:.4f})")
    
    if verbose:
        print(f"\nDecimation complete:")
        print(f"  Kept: {len(kept_frames)} frames")
        print(f"  Removed: {len(removed_frames)} frames")
        print(f"  Reduction: {len(removed_frames) / len(sorted_frame_names) * 100:.1f}%")
    
    return kept_frames, removed_frames


def write_decimated_output(
    input_dir: Path,
    output_dir: Path,
    cameras: Dict[int, CameraIntrinsics],
    images: Dict[int, ImagePose],
    frames: Dict[str, Frame],
    kept_frames: Set[str],
    copy_images: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Write decimated dataset to output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_output_dir = output_dir / "images"
    images_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect image IDs to keep
    kept_image_ids: Set[int] = set()
    for frame_name in kept_frames:
        frame = frames[frame_name]
        for image_pose in frame.images.values():
            kept_image_ids.add(image_pose.image_id)
    
    # Write cameras.txt (same as input)
    cameras_txt = output_dir / "cameras.txt"
    with open(cameras_txt, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for cam_id, cam in cameras.items():
            params_str = " ".join(str(p) for p in cam.params)
            f.write(f"{cam_id} {cam.model} {cam.width} {cam.height} {params_str}\n")
    
    # Write images.txt (only kept images)
    images_txt = output_dir / "images.txt"
    kept_images = {img_id: img for img_id, img in images.items() if img_id in kept_image_ids}
    
    with open(images_txt, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(kept_images)}\n")
        for img_id in sorted(kept_images.keys()):
            img = kept_images[img_id]
            f.write(
                f"{img_id} {img.qw} {img.qx} {img.qy} {img.qz} "
                f"{img.tx} {img.ty} {img.tz} {img.camera_id} {img.name}\n"
            )
            f.write(" \n")  # Empty POINTS2D line
    
    # Copy points3D files (preserve original data including colors, errors, tracks)
    input_points3d_txt = input_dir / "points3D.txt"
    input_points3d_ply = input_dir / "points3D.ply"
    
    if input_points3d_txt.exists():
        shutil.copy2(input_points3d_txt, output_dir / "points3D.txt")
        if verbose:
            print(f"  Copied points3D.txt")
    
    if input_points3d_ply.exists():
        shutil.copy2(input_points3d_ply, output_dir / "points3D.ply")
        if verbose:
            print(f"  Copied points3D.ply")
    
    # Copy images
    input_images_dir = input_dir / "images"
    copied_count = 0
    if copy_images and input_images_dir.exists():
        for img_id in kept_image_ids:
            img = images[img_id]
            src_path = input_images_dir / img.name
            dst_path = images_output_dir / img.name
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                copied_count += 1
    
    # Copy masks if they exist
    input_masks_dir = input_dir / "masks"
    masks_output_dir = output_dir / "masks"
    copied_masks_count = 0
    if input_masks_dir.exists():
        masks_output_dir.mkdir(parents=True, exist_ok=True)
        for img_id in kept_image_ids:
            img = images[img_id]
            # Mask filename: replace .jpg with .png
            mask_name = Path(img.name).stem + ".png"
            src_path = input_masks_dir / mask_name
            dst_path = masks_output_dir / mask_name
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                copied_masks_count += 1
    
    if verbose:
        print(f"\nOutput written to {output_dir}")
        print(f"  cameras.txt: {len(cameras)} cameras")
        print(f"  images.txt: {len(kept_images)} images")
        if copy_images:
            print(f"  Copied {copied_count} images")
        if copied_masks_count > 0:
            print(f"  Copied {copied_masks_count} masks")
    
    return {
        "num_cameras": len(cameras),
        "num_images": len(kept_images),
        "copied_images": copied_count,
        "copied_masks": copied_masks_count,
    }


def save_visibility_cache(
    cache_path: Path,
    frames_visibility: Dict[str, Dict[str, Set[int]]],
    max_depth: float,
    skip_directions: List[str],
    num_points: int,
    num_frames: int,
) -> None:
    """
    Save visible points cache to pickle file.
    
    Args:
        cache_path: Path to save cache file
        frames_visibility: Dict of frame_name -> {direction -> set of point IDs}
        max_depth: max_depth used for calculation
        skip_directions: List of directions that were skipped
        num_points: Number of points used
        num_frames: Number of frames processed
    """
    cache_data = {
        "version": 2,
        "max_depth": max_depth,
        "skip_directions": skip_directions,
        "num_points": num_points,
        "num_frames": num_frames,
        "frames_visibility": {
            frame_name: {
                direction: list(point_ids)  # Convert set to list for serialization
                for direction, point_ids in directions.items()
            }
            for frame_name, directions in frames_visibility.items()
        },
    }
    
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)


def load_visibility_cache(
    cache_path: Path,
) -> Tuple[Dict[str, Dict[str, Set[int]]], Dict[str, Any]]:
    """
    Load visible points cache from pickle file.
    
    Returns:
        Tuple of (frames_visibility, metadata)
    """
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)
    
    # Convert lists back to sets
    frames_visibility = {
        frame_name: {
            direction: set(point_ids)
            for direction, point_ids in directions.items()
        }
        for frame_name, directions in cache_data["frames_visibility"].items()
    }
    
    metadata = {
        "version": cache_data.get("version", 0),
        "max_depth": cache_data.get("max_depth"),
        "skip_directions": cache_data.get("skip_directions", []),
        "num_points": cache_data.get("num_points"),
        "num_frames": cache_data.get("num_frames"),
    }
    
    return frames_visibility, metadata


def prompt_cache_usage(cache_path: Path, metadata: Dict[str, Any], verbose: bool = True) -> bool:
    """
    Prompt user whether to use cached visibility data.
    
    Returns:
        True to use cache, False to recalculate
    """
    if not verbose:
        return True  # In quiet mode, use cache by default
    
    print(f"\nFound cached visibility data: {cache_path}")
    print(f"  Cache info:")
    print(f"    - max_depth: {metadata.get('max_depth')}")
    print(f"    - skip_directions: {metadata.get('skip_directions', [])}")
    print(f"    - num_points: {metadata.get('num_points')}")
    print(f"    - num_frames: {metadata.get('num_frames')}")
    
    while True:
        response = input("\nUse cached data? [Y]es / [N]o (recalculate): ").strip().lower()
        if response in ("", "y", "yes"):
            return True
        elif response in ("n", "no"):
            return False
        else:
            print("Please enter 'y' or 'n'")


def decimate_cubemap_images(
    input_dir: Path,
    output_dir: Path,
    overlap_threshold: float = 0.8,
    baseline_threshold: float = 0.1,
    min_observations: int = 3,
    window_size: int = 1,
    skip_directions: Optional[List[str]] = None,
    copy_images: bool = True,
    max_depth: float = 100.0,
    num_workers: int = 4,
    cache_file: Optional[Path] = None,
    use_cache: Optional[bool] = None,
    verbose: bool = True,
    no_stats: bool = False,
) -> Dict[str, Any]:
    """
    Main function to decimate cubemap images.
    
    Args:
        input_dir: Directory with COLMAP format data (cameras.txt, images.txt, points3D.txt/ply)
        output_dir: Output directory for decimated dataset
        overlap_threshold: Frames with overlap > threshold are redundancy candidates
        baseline_threshold: Frames with baseline < threshold are redundancy candidates
        min_observations: Ensure each 3D point is observed at least this many times
        skip_directions: List of directions to skip (e.g., ["bottom", "top"])
        copy_images: Whether to copy image files to output
        max_depth: Maximum depth for point visibility check
        num_workers: Number of parallel worker processes
        cache_file: Path to cache file (default: input_dir/visibility_cache.pkl)
        use_cache: True=use cache without prompt, False=recalculate, None=prompt user
        verbose: Print progress
    
    Returns:
        Statistics dictionary
    """
    # Load data
    cameras_path = input_dir / "cameras.txt"
    images_path = input_dir / "images.txt"
    points3d_txt_path = input_dir / "points3D.txt"
    points3d_ply_path = input_dir / "points3D.ply"
    
    if not cameras_path.exists():
        raise FileNotFoundError(f"cameras.txt not found in {input_dir}")
    if not images_path.exists():
        raise FileNotFoundError(f"images.txt not found in {input_dir}")
    
    if verbose:
        print(f"Loading data from {input_dir}")
    
    cameras = parse_cameras_txt(cameras_path)
    images = parse_images_txt(images_path)
    
    if verbose:
        print(f"  Loaded {len(cameras)} cameras, {len(images)} images")
    
    # Load points
    if points3d_ply_path.exists() and HAS_OPEN3D:
        points, point_ids = load_points3d_ply(points3d_ply_path)
        if verbose:
            print(f"  Loaded {len(points)} points from points3D.ply")
    elif points3d_txt_path.exists():
        points, point_ids = parse_points3d_txt(points3d_txt_path)
        if verbose:
            print(f"  Loaded {len(points)} points from points3D.txt")
    else:
        raise FileNotFoundError(f"No points3D.txt or points3D.ply found in {input_dir}")
    
    # Group images by frame
    frames = group_images_by_frame(images)
    if verbose:
        print(f"  Grouped into {len(frames)} frames")
    
    # Define directions
    all_directions = ["top", "front", "right", "back", "left", "bottom"]
    if skip_directions is None:
        skip_directions = []
    directions = [d for d in all_directions if d not in skip_directions]
    
    if verbose and skip_directions:
        print(f"  Skipping directions: {skip_directions}")
        print(f"  Using directions: {directions}")
    
    # Determine cache file path
    if cache_file is None:
        cache_file = input_dir / "visibility_cache.pkl"
    
    # Check if cache exists and decide whether to use it
    use_cached_data = False
    cached_visibility = None
    
    if cache_file.exists():
        try:
            cached_visibility, cache_metadata = load_visibility_cache(cache_file)
            
            # Check if cache is compatible
            cache_compatible = (
                cache_metadata.get("max_depth") == max_depth and
                set(cache_metadata.get("skip_directions", [])) == set(skip_directions) and
                cache_metadata.get("num_points") == len(points) and
                cache_metadata.get("num_frames") == len(frames)
            )
            
            if not cache_compatible:
                if verbose:
                    print(f"\nCache parameters do not match current settings:")
                    print(f"  Cache: max_depth={cache_metadata.get('max_depth')}, skip_directions={cache_metadata.get('skip_directions', [])}")
                    print(f"  Current: max_depth={max_depth}, skip_directions={skip_directions}")
                    print(f"  Cache: {cache_metadata.get('num_points')} points, {cache_metadata.get('num_frames')} frames")
                    print(f"  Current: {len(points)} points, {len(frames)} frames")
                    print("  Cache will be recalculated.")
                use_cached_data = False
            elif use_cache is True:
                # Explicitly use cache
                use_cached_data = True
                if verbose:
                    print(f"Using cached visibility data: {cache_file}")
            elif use_cache is False:
                # Explicitly recalculate
                use_cached_data = False
                if verbose:
                    print("Recalculating visibility data (--recalculate specified)")
            else:
                # Prompt user
                use_cached_data = prompt_cache_usage(cache_file, cache_metadata, verbose)
        
        except Exception as exc:
            if verbose:
                print(f"Warning: Failed to load cache: {exc}")
            use_cached_data = False
    
    if use_cached_data and cached_visibility is not None:
        # Apply cached visibility data to frames
        for frame_name, frame in frames.items():
            if frame_name in cached_visibility:
                frame.visible_points = cached_visibility[frame_name]
        
        if verbose:
            print(f"Loaded visibility data for {len(cached_visibility)} frames from cache")
    else:
        # Calculate visible points for each direction in each frame (parallel)
        if verbose:
            print(f"Calculating visible points per view (using {num_workers} workers)...")
        
        # Prepare tasks for parallel processing
        tasks = []
        for frame_name, frame in frames.items():
            for direction, image_pose in frame.images.items():
                if direction not in directions:
                    continue
                
                camera = cameras[image_pose.camera_id]
                task = (
                    frame_name,
                    direction,
                    camera_to_dict(camera),
                    image_pose_to_dict(image_pose),
                    points,
                    point_ids,
                    max_depth,
                )
                tasks.append(task)
        
        total_views = len(tasks)
        completed_views = 0
        
        # Process in parallel
        if num_workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(compute_visible_points_worker, task): task for task in tasks}
                
                for future in as_completed(futures):
                    try:
                        frame_name, direction, visible = future.result()
                        frames[frame_name].visible_points[direction] = visible
                        completed_views += 1
                        
                        if verbose and completed_views % 100 == 0:
                            print(f"  Processed {completed_views}/{total_views} views...")
                    except Exception as exc:
                        if verbose:
                            print(f"  Error processing view: {exc}")
        else:
            # Single-threaded fallback
            for task in tasks:
                frame_name, direction, visible = compute_visible_points_worker(task)
                frames[frame_name].visible_points[direction] = visible
                completed_views += 1
        
        if verbose:
            print(f"  Processed {completed_views} views")
            # Print sample stats
            sample_frame = next(iter(frames.values()))
            for direction, visible in sample_frame.visible_points.items():
                print(f"    Sample '{sample_frame.frame_name}' {direction}: {len(visible)} visible points")
        
        # Save cache
        try:
            frames_visibility = {
                frame_name: dict(frame.visible_points)
                for frame_name, frame in frames.items()
            }
            save_visibility_cache(
                cache_file,
                frames_visibility,
                max_depth,
                skip_directions,
                len(points),
                len(frames),
            )
            if verbose:
                print(f"  Saved visibility cache to: {cache_file}")
        except Exception as exc:
            if verbose:
                print(f"  Warning: Failed to save cache: {exc}")
    
    # Decimate frames
    kept_frames, removed_frames = decimate_frames(
        frames=frames,
        directions=directions,
        overlap_threshold=overlap_threshold,
        baseline_threshold=baseline_threshold,
        min_observations=min_observations,
        window_size=window_size,
        verbose=verbose,
        show_stats=not no_stats,
    )
    
    # Write output
    result = write_decimated_output(
        input_dir=input_dir,
        output_dir=output_dir,
        cameras=cameras,
        images=images,
        frames=frames,
        kept_frames=kept_frames,
        copy_images=copy_images,
        verbose=verbose,
    )
    
    result["kept_frames"] = len(kept_frames)
    result["removed_frames"] = len(removed_frames)
    result["total_frames"] = len(frames)
    
    return result


def load_config(config_path: Path = Path("config_decimate.txt")) -> Dict[str, Any]:
    """Load configuration from config file."""
    config = {}
    if not config_path.exists():
        return config
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                if "=" not in line:
                    continue
                
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes
                if value and len(value) >= 2:
                    if (value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"):
                        value = value[1:-1]
                
                if value.lower() in ("true", "yes", "1"):
                    config[key] = True
                elif value.lower() in ("false", "no", "0", ""):
                    config[key] = False
                elif value:
                    config[key] = value
    except Exception as e:
        print(f"Warning: Failed to read config: {e}")
    
    return config


def main() -> int:
    config = load_config()
    
    parser = argparse.ArgumentParser(
        description="Decimate cubemap images based on view frustum overlap and baseline distance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        required="input" not in config,
        default=Path(config["input"]) if "input" in config else None,
        help="Input directory with COLMAP format data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required="output" not in config,
        default=Path(config["output"]) if "output" in config else None,
        help="Output directory for decimated dataset",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=float(config.get("overlap-threshold", 0.8)),
        help="Overlap threshold for redundancy detection (0.0-1.0)",
    )
    parser.add_argument(
        "--baseline-threshold",
        type=float,
        default=float(config.get("baseline-threshold", 0.1)),
        help="Baseline distance threshold for redundancy detection",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=int(config.get("min-observations", 3)),
        help="Minimum observation count per 3D point",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=int(config.get("window-size", 1)),
        help="Compare each frame with +/-N frames (1=adjacent only, higher=more aggressive decimation)",
    )
    parser.add_argument(
        "--skip-directions",
        type=str,
        default=config.get("skip-directions", ""),
        help="Comma-separated list of directions to skip (e.g., 'bottom' or 'top,bottom'). Valid: top,front,right,back,left,bottom",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        default=config.get("no-copy-images", False),
        help="Don't copy image files to output",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=float(config.get("max-depth", 100.0)),
        help="Maximum depth for point visibility check",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(config.get("num-workers", 4)),
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path(config["cache-file"]) if "cache-file" in config else None,
        help="Path to visibility cache file (default: input_dir/visibility_cache.pkl)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=False,
        help="Use cached visibility data without prompting (if available and compatible)",
    )
    parser.add_argument(
        "--recalculate",
        action="store_true",
        default=False,
        help="Force recalculation of visibility data (ignore cache)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=config.get("quiet", False),
        help="Suppress progress output",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        default=config.get("no-stats", False),
        help="Suppress statistics output for threshold tuning",
    )
    
    args = parser.parse_args()
    
    # Determine cache usage mode
    if args.use_cache and args.recalculate:
        print("Error: Cannot specify both --use-cache and --recalculate")
        return 1
    
    use_cache_mode = None  # Prompt user
    if args.use_cache:
        use_cache_mode = True
    elif args.recalculate:
        use_cache_mode = False
    
    # Parse skip-directions
    valid_directions = {"top", "front", "right", "back", "left", "bottom"}
    skip_directions_list = []
    if args.skip_directions:
        skip_directions_list = [d.strip().lower() for d in args.skip_directions.split(",") if d.strip()]
        invalid_dirs = set(skip_directions_list) - valid_directions
        if invalid_dirs:
            print(f"Error: Invalid directions: {invalid_dirs}. Valid: {valid_directions}")
            return 1
    
    if not args.input.is_dir():
        print(f"Error: Input directory not found: {args.input}")
        return 1
    
    try:
        result = decimate_cubemap_images(
            input_dir=args.input,
            output_dir=args.output,
            overlap_threshold=args.overlap_threshold,
            baseline_threshold=args.baseline_threshold,
            min_observations=args.min_observations,
            window_size=args.window_size,
            skip_directions=skip_directions_list,
            copy_images=not args.no_copy_images,
            max_depth=args.max_depth,
            num_workers=args.num_workers,
            cache_file=args.cache_file,
            use_cache=use_cache_mode,
            verbose=not args.quiet,
            no_stats=args.no_stats,
        )
        
        if not args.quiet:
            print("\n=== Decimation Summary ===")
            print(f"Total frames: {result['total_frames']}")
            print(f"Kept frames: {result['kept_frames']}")
            print(f"Removed frames: {result['removed_frames']}")
            print(f"Reduction: {result['removed_frames'] / result['total_frames'] * 100:.1f}%")
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
