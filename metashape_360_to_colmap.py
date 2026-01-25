#!/usr/bin/env python3
"""
Metashape equirectangular XML → COLMAP converter

- Reads Metashape XML camera poses (equirectangular capture) and optional PLY.
- For each equirectangular frame, slices four 90° views (front/right/back/left)
  into square crops and saves them under <output>/images/.
- Writes COLMAP-compatible cameras.txt, images.txt, and points3D.txt.

Usage example:
    python metashape_360_to_colmap.py \
        --images ./equirect/ \
        --xml ./cameras.xml \
        --output ./colmap_dataset/ \
        --ply ./dense.ply

Dependencies:
    pip install numpy pillow opencv-python
    Optional: pip install open3d (for points3D handling)
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import cv2
from PIL import Image

try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_OPEN3D = False


def find_param(calib_xml: ET.Element, param_name: str) -> float:
    """Find a parameter in calibration XML, return 0.0 if not found."""
    param = calib_xml.find(param_name)
    if param is not None and param.text:
        return float(param.text)
    return 0.0


def parse_metashape_xml(xml_path: Path) -> Dict[str, Any]:
    """Parse Metashape XML and return sensors, components, and cameras."""
    xml_tree = ET.parse(xml_path)
    root = xml_tree.getroot()
    chunk = root[0]
    sensors = chunk.find("sensors")

    if sensors is None:
        raise ValueError("No sensors found in Metashape XML")

    calibrated_sensors = [
        sensor for sensor in sensors.iter("sensor")
        if sensor.get("type") == "spherical" or sensor.find("calibration")
    ]
    if not calibrated_sensors:
        raise ValueError("No calibrated sensor found in Metashape XML")

    sensor_types = [s.get("type") for s in calibrated_sensors]
    if sensor_types.count(sensor_types[0]) != len(sensor_types):
        raise ValueError("All sensors must share the same type")

    sensor_type = sensor_types[0]
    if sensor_type != "spherical":
        raise ValueError(f"Expected equirectangular (spherical) sensors, got {sensor_type}")

    sensor_dict: Dict[str, Dict[str, float]] = {}
    for sensor in calibrated_sensors:
        s: Dict[str, float] = {}
        resolution = sensor.find("resolution")
        if resolution is None:
            raise ValueError("Resolution not found in Metashape XML")

        s["w"] = int(resolution.get("width"))
        s["h"] = int(resolution.get("height"))

        calib = sensor.find("calibration")
        if calib is None:
            s["fl_x"] = s["w"] / 2.0
            s["fl_y"] = s["h"]
            s["cx"] = s["w"] / 2.0
            s["cy"] = s["h"] / 2.0
        else:
            f = calib.find("f")
            if f is None or f.text is None:
                raise ValueError("Focal length not found in Metashape XML")
            s["fl_x"] = s["fl_y"] = float(f.text)
            s["cx"] = find_param(calib, "cx") + s["w"] / 2.0
            s["cy"] = find_param(calib, "cy") + s["h"] / 2.0
            s["k1"] = find_param(calib, "k1")
            s["k2"] = find_param(calib, "k2")
            s["k3"] = find_param(calib, "k3")
            s["k4"] = find_param(calib, "k4")
            s["p1"] = find_param(calib, "p1")
            s["p2"] = find_param(calib, "p2")

        sensor_dict[sensor.get("id")] = s

    components = chunk.find("components")
    component_dict: Dict[str, np.ndarray] = {}
    if components is not None:
        for component in components.iter("component"):
            transform = component.find("transform")
            if transform is None:
                continue

            rotation = transform.find("rotation")
            translation = transform.find("translation")
            scale = transform.find("scale")

            r = np.eye(3) if rotation is None or rotation.text is None else np.array(
                [float(x) for x in rotation.text.split()]).reshape((3, 3))
            t = np.zeros(3) if translation is None or translation.text is None else np.array(
                [float(x) for x in translation.text.split()])
            s = 1.0 if scale is None or scale.text is None else float(scale.text)

            m = np.eye(4)
            m[:3, :3] = r
            m[:3, 3] = t / s
            component_dict[component.get("id")] = m

    cameras = chunk.find("cameras")
    if cameras is None:
        raise ValueError("No cameras found in Metashape XML")

    return {
        "sensor_dict": sensor_dict,
        "component_dict": component_dict,
        "cameras": cameras,
    }


def get_direction_rotation_matrix(direction: str) -> np.ndarray:
    """Rotation matrix for a 90° yaw step (front/right/back/left)."""
    yaw_deg = direction_yaw_deg(direction)
    yaw = np.radians(yaw_deg)
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    return np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y],
    ])


def direction_yaw_deg(direction: str) -> float:
    """Canonical yaw (deg) for each cardinal crop; keep shared between remap and extrinsics."""
    yaw_angles = {
        "front": 0.0,
        "right": -90.0,
        "back": 180.0,
        "left": 90.0,
    }
    return yaw_angles[direction]


def quaternion_from_matrix(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to normalized quaternion (x, y, z, w)."""
    trace = np.trace(R)
    if trace > 0:
        S = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / S
        x = (R[2, 1] - R[1, 2]) * S
        y = (R[0, 2] - R[2, 0]) * S
        z = (R[1, 0] - R[0, 1]) * S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def crop_direction(
    equirect_image: Image.Image,
    direction: str,
    crop_size: int,
    fov_deg: float = 90.0,
    flip_vertical: bool = True,
) -> Image.Image:
    """Rectilinear 90° crop from equirectangular using cv2.remap (no wrap seams)."""
    yaw_deg = direction_yaw_deg(direction)
    # Prepare output grid (pixel centers).
    w_out = h_out = crop_size
    fx = fy = (w_out / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    cx = cy = (w_out - 1) / 2.0
    u, v = np.meshgrid(np.arange(w_out, dtype=np.float32), np.arange(h_out, dtype=np.float32))
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = np.ones_like(x)
    dirs = np.stack([x, y, z], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    # Yaw rotation per face (right-handed, yaw about +Y).
    yaw = np.deg2rad(yaw_deg)
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]], dtype=np.float32)
    dirs = dirs @ R_y.T

    # Convert direction vectors to equirectangular UV.
    lon = np.arctan2(dirs[..., 0], dirs[..., 2])  # [-pi, pi]
    lat = np.arctan2(dirs[..., 1], np.sqrt(dirs[..., 0] ** 2 + dirs[..., 2] ** 2))  # [-pi/2, pi/2]

    width, height = equirect_image.size
    map_x = (lon / (2 * np.pi) + 0.5) * float(width)
    map_y = (0.5 - lat / np.pi) * float(height)
    if flip_vertical:
        map_y = (0.5 + lat / np.pi) * float(height)
    map_y = np.clip(map_y, 0.0, float(height - 1))

    # Remap (wrap horizontally, clamp vertically).
    equirect_np = np.array(equirect_image.convert("RGB"))
    sampled = cv2.remap(
        equirect_np,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )

    return Image.fromarray(sampled, mode="RGB")


def convert_metashape_to_colmap(
    images_dir: Path,
    xml_path: Path,
    output_dir: Optional[Path] = None,
    ply_path: Optional[Path] = None,
    crop_size: int = 512,
    fov_deg: float = 90.0,
    max_images: Optional[int] = None,
    flip_vertical: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Convert Metashape equirectangular data to COLMAP format."""
    if output_dir is None:
        output_dir = xml_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)
    images_output_dir = output_dir / "images"
    images_output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Parsing Metashape XML: {xml_path}")

    xml_data = parse_metashape_xml(xml_path)
    sensor_dict = xml_data["sensor_dict"]
    component_dict = xml_data["component_dict"]
    cameras_xml = xml_data["cameras"]

    image_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp",".JPG", ".JPEG", ".PNG", ".TIFF", ".TIF", ".WEBP"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))

    image_filename_map = {img_path.stem: img_path for img_path in image_files}
    image_filename_map.update({img_path.name: img_path for img_path in image_files})

    if verbose:
        print(f"Found {len(image_files)} equirectangular images in {images_dir}")

    camera_id = 1  # single shared intrinsic entry
    fx = fy = (crop_size / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    cx = cy = crop_size / 2.0
    cameras_colmap = {
        camera_id: {
            "width": crop_size,
            "height": crop_size,
            "model": "PINHOLE",
            "params": [fx, fy, cx, cy],
        }
    }

    images_colmap: Dict[int, Dict[str, Any]] = {}
    image_id = 1
    processed_images = 0
    processed_cameras = 0
    num_skipped = 0
    directions = ["front", "right", "back", "left"]

    for camera in cameras_xml.iter("camera"):
        if max_images is not None and processed_cameras >= max_images:
            break
        camera_label = camera.get("label")
        if not camera_label:
            continue

        if camera_label not in image_filename_map:
            camera_label_no_ext = camera_label.split(".")[0]
            if camera_label_no_ext not in image_filename_map:
                if verbose:
                    print(f"  Skipping {camera_label}: no matching image")
                num_skipped += 1
                continue
            camera_label = camera_label_no_ext

        sensor_id = camera.get("sensor_id")
        if sensor_id not in sensor_dict:
            if verbose:
                print(f"  Skipping {camera_label}: no sensor calibration")
            num_skipped += 1
            continue

        transform_elem = camera.find("transform")
        if transform_elem is None or transform_elem.text is None:
            if verbose:
                print(f"  Skipping {camera_label}: no transform")
            num_skipped += 1
            continue

        transform = np.array([float(x) for x in transform_elem.text.split()]).reshape((4, 4))

        component_id = camera.get("component_id")
        if component_id in component_dict:
            transform = component_dict[component_id] @ transform

        src_image_path = image_filename_map[camera_label]
        try:
            equirect_image = Image.open(src_image_path)
            if equirect_image.mode != "RGB":
                equirect_image = equirect_image.convert("RGB")
        except Exception as exc:  # pragma: no cover - IO guard
            if verbose:
                print(f"  Skipping {camera_label}: failed to load image ({exc})")
            num_skipped += 1
            continue

        R_c2w = transform[:3, :3]
        t_c2w = transform[:3, 3]

        base_name = Path(camera_label).stem

        for direction in directions:
            cropped = crop_direction(
                equirect_image,
                direction,
                crop_size,
                fov_deg=fov_deg,
                flip_vertical=flip_vertical,
            )
            output_image_name = f"{base_name}_{direction}.jpg"
            output_image_path = images_output_dir / output_image_name
            cropped.save(output_image_path, quality=95)

            R_dir = get_direction_rotation_matrix(direction)
            R_c2w_dir = R_c2w @ R_dir  # align extrinsics with the rotated crop

            R_w2c = R_c2w_dir.T
            t_w2c = -R_w2c @ t_c2w
            q = quaternion_from_matrix(R_w2c)

            images_colmap[image_id] = {
                "quat": q,  # [x, y, z, w]
                "tvec": t_w2c,
                "camera_id": camera_id,
                "name": output_image_name,
            }
            image_id += 1
            processed_images += 1
        processed_cameras += 1

    if verbose:
        print(f"Processed {processed_images} cropped images")
        if max_images is not None:
            print(f"  (Stopped after {processed_cameras} source images due to --max-images)")
        if num_skipped > 0:
            print(f"Skipped {num_skipped} camera(s) with missing data")

    cameras_txt = output_dir / "cameras.txt"
    with open(cameras_txt, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras_colmap)}\n")
        for cam_id, cam_data in cameras_colmap.items():
            params_str = " ".join(str(p) for p in cam_data["params"])
            f.write(
                f"{cam_id} {cam_data['model']} {cam_data['width']} {cam_data['height']} {params_str}\n"
            )

    images_txt = output_dir / "images.txt"
    with open(images_txt, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images_colmap)}\n")
        for img_id in sorted(images_colmap.keys()):
            img_data = images_colmap[img_id]
            q = img_data["quat"]
            t = img_data["tvec"]
            f.write(
                f"{img_id} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]} {img_data['camera_id']} {img_data['name']}\n"
            )
            f.write(" \n") #LFS needs one space

    points3d_data = []
    if ply_path is not None and ply_path.exists() and HAS_OPEN3D:
        if verbose:
            print(f"Processing point cloud: {ply_path}")

        pc = o3d.io.read_point_cloud(str(ply_path))
        points3d = np.asarray(pc.points)
        colors = np.asarray(pc.colors) if pc.has_colors() else None

        comp_transform = None
        if len(component_dict) == 1:
            comp_transform = next(iter(component_dict.values()))
        elif len(component_dict) > 1:
            comp_transform = next(iter(component_dict.values()))
            if verbose:
                print("  Multiple components detected; using the first component transform")

        if comp_transform is not None:
            points_h = np.hstack([points3d, np.ones((len(points3d), 1))])
            points3d = (comp_transform @ points_h.T).T[:, :3]

        for idx, point in enumerate(points3d, start=1):
            x, y, z = point
            if colors is not None:
                r = int(colors[idx - 1, 0] * 255)
                g = int(colors[idx - 1, 1] * 255)
                b = int(colors[idx - 1, 2] * 255)
            else:
                r = g = b = 128
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            points3d_data.append((idx, x, y, z, r, g, b, 0.0, ""))

        output_ply = output_dir / "points3D.ply"
        pc.points = o3d.utility.Vector3dVector(points3d)
        o3d.io.write_point_cloud(str(output_ply), pc)
        if verbose:
            print(f"  Wrote transformed point cloud to {output_ply}")
    elif ply_path is not None and not HAS_OPEN3D:
        if verbose:
            print("open3d is not installed; skipping PLY to points3D.txt conversion")

    points3d_txt = output_dir / "points3D.txt"
    with open(points3d_txt, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points3d_data)}\n")
        for pid, x, y, z, r, g, b, err, track in points3d_data:
            f.write(f"{pid} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {err} {track}\n")

    if verbose:
        print(f"Wrote cameras.txt, images.txt, points3D.txt to {output_dir}")

    return {
        "num_images": len(images_colmap),
        "num_cameras": len(cameras_colmap),
        "num_skipped": num_skipped,
        "num_points3d": len(points3d_data),
        "crop_size": crop_size,
        "output_dir": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Metashape equirectangular XML to COLMAP format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images", type=Path, required=True, help="Directory with equirectangular images")
    parser.add_argument("--xml", type=Path, required=True, help="Path to Metashape cameras.xml")
    parser.add_argument("--output", type=Path, required=True, help="Output directory (COLMAP layout)")
    parser.add_argument("--ply", type=Path, default=None, help="Optional PLY to export points3D.txt")
    parser.add_argument("--crop-size", type=int, default=1920, help="Crop size for 90° views")
    parser.add_argument("--fov-deg", type=float, default=90.0, help="Horizontal FoV for rectilinear crops")
    parser.add_argument(
        "--flip-vertical",
        action="store_true",
        default=True,
        help="Flip vertical direction (invert latitude) when sampling equirect (default: on)",
    )
    parser.add_argument(
        "--no-flip-vertical",
        action="store_false",
        dest="flip_vertical",
        help="Disable vertical flip if your data is already upright",
    )
    parser.add_argument("--max-images", type=int, default=10000, help="Optional limit on number of equirectangular images to process (for quick tests)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    if not args.images.is_dir():
        print(f"Error: Images directory not found: {args.images}")
        return 1
    if not args.xml.is_file():
        print(f"Error: XML file not found: {args.xml}")
        return 1
    if args.ply and not args.ply.is_file():
        print(f"Error: PLY file not found: {args.ply}")
        return 1

    try:
        result = convert_metashape_to_colmap(
            images_dir=args.images,
            xml_path=args.xml,
            output_dir=args.output,
            ply_path=args.ply,
            crop_size=args.crop_size,
            fov_deg=args.fov_deg,
            flip_vertical=args.flip_vertical,
            max_images=args.max_images,
            verbose=not args.quiet,
        )
        if not args.quiet:
            print("\nConversion complete!")
            print(f"  Cropped images: {result['num_images']}")
            print(f"  Cameras: {result['num_cameras']}")
            print(f"  Points3D: {result['num_points3d']}")
            print(f"  Skipped: {result['num_skipped']}")
            print(f"  Crop size: {result['crop_size']}x{result['crop_size']}")
            print(f"  Output: {result['output_dir']}")
        return 0
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
