#!/usr/bin/env python3
"""
Metashape equirectangular XML → COLMAP converter

- Reads Metashape XML camera poses (equirectangular capture) and optional PLY.
- For each equirectangular frame, slices six 90° views (top/front/right/back/left/bottom)
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
import configparser
import multiprocessing
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import cv2
from PIL import Image

try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_OPEN3D = False

try:
    from ultralytics import YOLO

    HAS_YOLO = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_YOLO = False


def detect_image_format(image_path: Path) -> Tuple[str, str, int]:
    """Detect image format, PIL mode and bit depth from file.
    
    Returns:
        Tuple of (file_extension, mode_description, bits_per_channel)
        file_extension: e.g., '.png', '.jpg', '.tiff'
        mode_description: e.g., 'RGB', 'RGBA', 'RGB;16', 'RGBA;16', etc.
        bits_per_channel: 8, 16, or 32
    """
    # Try with OpenCV first for better 16-bit support
    import cv2
    cv_img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    
    # Determine file extension
    suffix = image_path.suffix.lower()
    if suffix in ['.jpg', '.jpeg']:
        ext = '.jpg'
    elif suffix in ['.png']:
        ext = '.png'
    elif suffix in ['.tiff', '.tif']:
        ext = '.tiff'
    elif suffix in ['.webp']:
        ext = '.webp'
    else:
        ext = suffix
    
    # Detect bit depth and channels from OpenCV
    bits_per_channel = 8  # default
    if cv_img is not None:
        dtype = cv_img.dtype
        channels = cv_img.shape[2] if len(cv_img.shape) == 3 else 1
        
        if dtype == np.uint16:
            bits_per_channel = 16
        elif dtype == np.uint32 or dtype == np.float32:
            bits_per_channel = 32
        
        # Construct PIL-compatible mode string for consistency
        if channels == 1:
            if bits_per_channel == 16:
                mode = 'I;16'
            elif bits_per_channel == 32:
                mode = 'F'
            else:
                mode = 'L'
        elif channels == 3:
            if bits_per_channel == 16:
                mode = 'RGB;16'
            else:
                mode = 'RGB'
        elif channels == 4:
            if bits_per_channel == 16:
                mode = 'RGBA;16'
            else:
                mode = 'RGBA'
        else:
            mode = 'RGB'
    else:
        # Fall back to PIL for mode detection
        img = Image.open(image_path)
        mode = img.mode
        if ';16' in mode or 'I;16' in mode:
            bits_per_channel = 16
        elif ';32F' in mode or 'F' in mode:
            bits_per_channel = 32
    
    return (ext, mode, bits_per_channel)


def open_image_preserving_bitdepth(image_path: str):
    """Open an image file, preserving 16-bit and higher bit depths.
    
    Uses OpenCV for reading to properly support 16-bit RGBA PNG files.
    Returns either a PIL Image (for 8-bit) or numpy array (for 16-bit+).
    """
    import cv2
    
    # Read with OpenCV to preserve bit depth
    cv_img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    
    if cv_img is None:
        # Fall back to PIL if OpenCV fails
        return Image.open(image_path)
    
    dtype = cv_img.dtype
    
    # For 16-bit and higher, keep as numpy array (OpenCV format: BGR/BGRA)
    if dtype == np.uint16 or dtype == np.uint32 or dtype == np.float32:
        # Convert BGR to RGB for consistency
        if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        elif len(cv_img.shape) == 3 and cv_img.shape[2] == 4:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
        # Return as numpy array - we'll handle it specially in crop_direction
        return cv_img
    
    # For 8-bit, convert to PIL Image
    if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    elif len(cv_img.shape) == 3 and cv_img.shape[2] == 4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
    
    return Image.fromarray(cv_img)




def get_output_format_extension(output_format: Optional[str], input_ext: str, input_mode: str) -> Tuple[str, bool]:
    """Get output file extension and whether to preserve alpha channel.
    
    Args:
        output_format: User-specified format (e.g., 'png', 'auto') or None for auto-detect
        input_ext: Input file extension (e.g., '.png')
        input_mode: Input PIL mode (e.g., 'RGBA')
    
    Returns:
        Tuple of (extension, preserve_alpha)
    """
    if output_format == 'auto' or output_format is None:
        # Auto-detect: use input format
        return (input_ext, 'A' in input_mode)
    
    # User specified format
    output_format = output_format.lower().strip('.')
    
    if output_format in ['jpg', 'jpeg']:
        return ('.jpg', False)
    elif output_format == 'png':
        return ('.png', True)
    elif output_format in ['tiff', 'tif']:
        return ('.tiff', True)
    elif output_format == 'webp':
        return ('.webp', True)
    else:
        # Unknown format, use input format
        return (input_ext, 'A' in input_mode)


def find_param(calib_xml: ET.Element, param_name: str) -> float:
    """Find a parameter in calibration XML, return 0.0 if not found."""
    param = calib_xml.find(param_name)
    if param is not None and param.text:
        return float(param.text)
    return 0.0


def numpy_to_pil_image_16bit(array: np.ndarray) -> Image.Image:
    """Convert numpy array (from OpenCV) to PIL Image, handling 16-bit properly."""
    if array.dtype == np.uint16:
        channels = array.shape[2] if len(array.shape) == 3 else 1
        
        if channels == 4:
            # 16-bit RGBA
            # PIL doesn't have built-in RGBA;16 mode, so we use raw mode
            size = (array.shape[1], array.shape[0])
            # Convert to bytes in the right format
            if array.flags['C_CONTIGUOUS']:
                data = array.tobytes()
            else:
                data = np.ascontiguousarray(array).tobytes()
            # Create PIL Image with raw mode
            img = Image.frombytes('RGBA;16', size, data)
            return img
        elif channels == 3:
            # 16-bit RGB
            size = (array.shape[1], array.shape[0])
            if array.flags['C_CONTIGUOUS']:
                data = array.tobytes()
            else:
                data = np.ascontiguousarray(array).tobytes()
            img = Image.frombytes('RGB;16', size, data)
            return img
        elif channels == 1:
            # 16-bit grayscale
            size = (array.shape[1], array.shape[0])
            if array.flags['C_CONTIGUOUS']:
                data = array.tobytes()
            else:
                data = np.ascontiguousarray(array).tobytes()
            img = Image.frombytes('I;16', size, data)
            return img
    elif array.dtype == np.uint8:
        return Image.fromarray(array)
    else:
        return Image.fromarray(array)


def numpy_to_image_preserving_bitdepth(array: np.ndarray, mode: str):
    """Convert numpy array to PIL Image or keep as numpy array for bit depth preservation.
    
    PIL doesn't support modes like RGBA;16, so for 16-bit+ we return numpy arrays.
    For 8-bit modes, returns PIL Image.
    """
    # For 16-bit and higher, PIL doesn't support the mode strings, so return numpy array
    if ';16' in mode or 'I;16' in mode or ';32F' in mode or mode == 'F':
        # Ensure correct dtype
        if mode in ['RGBA;16', 'RGB;16', 'I;16']:
            return array.astype(np.uint16) if array.dtype != np.uint16 else array
        elif mode in ['F', 'RGB;32F', 'RGBA;32F']:
            return array.astype(np.float32) if array.dtype != np.float32 else array
        return array
    
    # For 8-bit modes, convert to PIL Image
    try:
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)
        return Image.fromarray(array, mode=mode)
    except:
        # Fallback
        return Image.fromarray(array)


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
    """Rotation matrix for direction views: yaw for cardinal directions, pitch for top/bottom."""
    yaw_deg = direction_yaw_deg(direction)
    pitch_deg = direction_pitch_deg(direction)
    
    # First apply yaw rotation (about Y-axis)
    yaw = np.radians(yaw_deg)
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    R_yaw = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y],
    ])
    
    # Then apply pitch rotation (about X-axis)
    pitch = np.radians(pitch_deg)
    cos_p = np.cos(pitch)
    sin_p = np.sin(pitch)
    R_pitch = np.array([
        [1, 0, 0],
        [0, cos_p, -sin_p],
        [0, sin_p, cos_p],
    ])
    
    return R_yaw @ R_pitch


def direction_yaw_deg(direction: str) -> float:
    """Canonical yaw (deg) for each crop; keep shared between remap and extrinsics."""
    yaw_angles = {
        "top": 0.0,
        "front": 0.0,
        "right": -90.0,
        "back": 180.0,
        "left": 90.0,
        "bottom": 0.0,
    }
    return yaw_angles[direction]


def direction_pitch_deg(direction: str) -> float:
    """Canonical pitch (deg) for each crop; 90° for top, -90° for bottom, 0° for others."""
    pitch_angles = {
        "top": 90.0,
        "front": 0.0,
        "right": 0.0,
        "back": 0.0,
        "left": 0.0,
        "bottom": -90.0,
    }
    return pitch_angles[direction]


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


def create_overexposure_mask(
    image: Image.Image,
    threshold: int = 250,
    dilate_pixels: int = 5,
) -> np.ndarray:
    """Detect white-blown-out (overexposed) pixels and return a binary mask.
    
    Pixels where ALL of R, G, B channels exceed the threshold are considered
    overexposed. The result is optionally dilated to cover fringe artifacts.
    
    Args:
        image: PIL Image in RGB mode.
        threshold: Minimum value (0-255) for all channels to be considered
                   blown-out.  Default 250.
        dilate_pixels: Radius of morphological dilation applied to the raw
                       overexposure mask.  Set to 0 to disable.
    
    Returns:
        uint8 numpy array (H, W) where 255 = overexposed, 0 = normal.
    """
    img_np = np.array(image)  # (H, W, 3)
    # All three channels above threshold
    blown = np.all(img_np >= threshold, axis=-1).astype(np.uint8) * 255
    
    if dilate_pixels > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilate_pixels * 2 + 1, dilate_pixels * 2 + 1),
        )
        blown = cv2.dilate(blown, kernel, iterations=1)
    
    return blown


def create_person_mask_from_yolo(
    image_path: str,
    yolo_model: Any,
    yolo_conf: float = 0.25,
    invert_mask: bool = False,
    class_ids: Optional[list] = None,
    mask_overexposure: bool = False,
    overexposure_threshold: int = 250,
    overexposure_dilate: int = 5,
) -> Image.Image:
    """Create a binary mask for detected objects using YOLO.
    
    Optionally also masks white-blown-out (overexposed) pixels.
    
    Args:
        image_path: Path to the equirectangular image
        yolo_model: YOLO model instance
        yolo_conf: Minimum YOLO confidence score (0.0-1.0) to keep detections.
        invert_mask: If True, object=white(255) and background=black(0).
                     If False, object=black(0) and background=white(255) for 3DGS training.
        class_ids: List of YOLO class IDs to include in mask. If None, uses [0] (person only).
                   Common COCO classes: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck, etc.
        mask_overexposure: If True, also mask pixels that are white-blown-out.
        overexposure_threshold: Pixel value threshold (0-255) for overexposure detection.
        overexposure_dilate: Dilation radius for overexposure mask (pixels).
    """
    # Load image
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Default to person class if not specified
    if class_ids is None:
        class_ids = [0]
    
    # Run YOLO detection
    results = yolo_model(image, verbose=False, conf=yolo_conf)
    
    # Create binary mask
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    
    # Process detections for specified classes
    for result in results:
        if result.masks is not None:
            for i, cls in enumerate(result.boxes.cls):
                if int(cls) in class_ids:
                    mask_data = result.masks.data[i].cpu().numpy()
                    # Resize mask to original image size
                    mask_resized = cv2.resize(mask_data, (image.width, image.height))
                    mask = np.maximum(mask, (mask_resized * 255).astype(np.uint8))
    
    # Combine with overexposure mask if enabled
    if mask_overexposure:
        overexposure = create_overexposure_mask(
            image,
            threshold=overexposure_threshold,
            dilate_pixels=overexposure_dilate,
        )
        mask = np.maximum(mask, overexposure)
    
    # Default: person=white(255), background=black(0)
    # Without invert_mask: person=black(0), background=white(255) for 3DGS training
    if not invert_mask:
        mask = 255 - mask
    
    return Image.fromarray(mask, mode="L")


def generate_mask_and_save(
    image_path: str,
    output_mask_path: str,
    yolo_model_path: str,
    yolo_conf: float = 0.25,
    invert_mask: bool = False,
    class_ids: Optional[list] = None,
    mask_overexposure: bool = False,
    overexposure_threshold: int = 250,
    overexposure_dilate: int = 5,
) -> Tuple[str, str]:
    """Generate object mask and save to file (for parallel processing).
    
    Args:
        image_path: Path to the equirectangular image
        output_mask_path: Path to save the mask
        yolo_model_path: Path to YOLO model
        yolo_conf: Minimum YOLO confidence score (0.0-1.0) to keep detections
        invert_mask: Whether to invert the mask
        class_ids: List of YOLO class IDs to include in mask
        mask_overexposure: Whether to also mask overexposed pixels
        overexposure_threshold: Pixel value threshold for overexposure detection
        overexposure_dilate: Dilation radius for overexposure mask
    
    Returns:
        Tuple of (image_path, output_mask_path)
    """
    if not HAS_YOLO:
        raise ImportError("ultralytics is required")
    
    # Load YOLO model in worker process
    yolo_model = YOLO(yolo_model_path)
    
    # Generate mask
    mask = create_person_mask_from_yolo(
        image_path, yolo_model, yolo_conf, invert_mask, class_ids,
        mask_overexposure=mask_overexposure,
        overexposure_threshold=overexposure_threshold,
        overexposure_dilate=overexposure_dilate,
    )
    
    # Save mask
    mask.save(output_mask_path)
    
    return (image_path, output_mask_path)


def crop_and_save_image(
    image_path: str,
    direction: str,
    crop_size: int,
    output_image_path: str,
    fov_deg: float = 90.0,
    flip_vertical: bool = True,
    mask_image_path: Optional[str] = None,
    output_mask_path: Optional[str] = None,
    yaw_offset: float = 0.0,
    preserve_alpha: bool = False,
    verbose: bool = False,
) -> Tuple[str, str, str, np.ndarray]:
    """Crop equirectangular image and save. Optionally crop and save mask from file path. Returns (direction, output_name, output_path, metadata).
    
    Preserves original image mode, bit depth, and alpha channel through the cropping process.
    Bit depth is detected per-image to handle mixed bit depth datasets correctly.
    """
    # Use OpenCV-based reader to properly handle 16-bit RGBA
    equirect_image = open_image_preserving_bitdepth(image_path)
    
    # Detect if it's a numpy array (16-bit+) or PIL Image (8-bit)
    is_numpy_array = isinstance(equirect_image, np.ndarray)
    
    # Verify input file info
    try:
        input_file = Path(image_path)
        input_size_bytes = input_file.stat().st_size
    except:
        input_size_bytes = 0
    
    # Detect this specific image's bit depth (important for mixed bit-depth datasets)
    if is_numpy_array:
        # NumPy array from OpenCV (16-bit or higher)
        dtype = equirect_image.dtype
        channels = equirect_image.shape[2] if len(equirect_image.shape) == 3 else 1
        
        if dtype == np.uint16:
            is_16bit = True
            is_32bit = False
            if channels == 1:
                original_mode = 'I;16'
            elif channels == 3:
                original_mode = 'RGB;16'
            elif channels == 4:
                original_mode = 'RGBA;16'
            else:
                original_mode = 'RGB;16'
        elif dtype == np.uint32:
            is_16bit = False
            is_32bit = True
            original_mode = 'F'
        else:
            is_16bit = False
            is_32bit = False
            original_mode = 'RGB'
    else:
        # PIL Image
        original_mode = equirect_image.mode
        is_16bit = ';16' in original_mode or 'I;16' in original_mode
        is_32bit = ';32F' in original_mode or 'F' in original_mode
    
    # Always write debug log (for GUI visibility even when not verbose on console)
    # DEBUG LOG DISABLED
    
    if verbose:
        print(f"  Cropping {Path(image_path).name} ({direction}): input mode={original_mode}")
    
    # Convert numpy array to PIL Image if needed
    if is_numpy_array:
        # Keep as numpy array for 16-bit processing - PIL doesn't support RGBA;16 mode
        # We'll convert to PIL only at save time
        pass  # Keep equirect_image as numpy array
    
    # Only convert if output format doesn't support the current bit depth
    # Don't force conversion - the crop_direction function will preserve the original mode and bit depth
    
    cropped = crop_direction(
        equirect_image,
        direction,
        crop_size,
        fov_deg=fov_deg,
        flip_vertical=flip_vertical,
        yaw_offset=yaw_offset,
    )
    
    # Log post-crop mode to debug file
    # DEBUG LOG DISABLED
    
    if verbose and isinstance(cropped, Image.Image) and cropped.mode != original_mode:
        print(f"    Mode changed after crop: {original_mode} → {cropped.mode}")
    
    # Save with appropriate format settings based on output file extension
    output_path_lower = output_image_path.lower()
    
    # Handle numpy array (16-bit) separately - use OpenCV for saving
    if isinstance(cropped, np.ndarray):
        try:
            import cv2
            if cropped.dtype == np.uint16:
                # Convert from RGB to BGR for OpenCV if RGB/RGBA
                if len(cropped.shape) == 3 and cropped.shape[2] == 3:
                    cropped_cv = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
                elif len(cropped.shape) == 3 and cropped.shape[2] == 4:
                    cropped_cv = cv2.cvtColor(cropped, cv2.COLOR_RGBA2BGRA)
                else:
                    cropped_cv = cropped
                
                # Save with OpenCV (preserves 16-bit)
                success = cv2.imwrite(output_image_path, cropped_cv)
                if verbose and success:
                    print(f"    Saved as 16-bit PNG with OpenCV")
                try:
                    debug_log = Path(output_image_path).parent / "DEBUG_BITDEPTH.log"
                    with open(debug_log, 'a') as f:
                        f.write(f"  Saved as 16-bit with OpenCV (cv2.imwrite), file={Path(output_image_path).name}\n")
                        f.flush()
                except:
                    pass
                return
        except Exception as e:
            # If OpenCV saving fails, fall through to try PIL
            pass
    
    # PIL Image saving (8-bit)
    # Failsafe: if cropped is still a numpy array, convert to PIL first
    if isinstance(cropped, np.ndarray):
        # Convert numpy array to PIL Image
        if cropped.dtype == np.uint8:
            if len(cropped.shape) == 3 and cropped.shape[2] == 4:
                cropped = Image.fromarray(cropped, 'RGBA')
            elif len(cropped.shape) == 3 and cropped.shape[2] == 3:
                cropped = Image.fromarray(cropped, 'RGB')
            else:
                cropped = Image.fromarray(cropped)
        else:
            # Shouldn't reach here - 16-bit should have been handled above
            return
    
    if output_path_lower.endswith('.jpg') or output_path_lower.endswith('.jpeg'):
        # JPG doesn't support alpha or bit depths > 8, convert to RGB 8-bit
        if isinstance(cropped, Image.Image) and cropped.mode not in ['RGB', 'L']:
            cropped = cropped.convert('RGB')
        cropped.save(output_image_path, quality=100, optimize=True)
        if verbose:
            print(f"    Saved as JPG (8-bit, no alpha)")
    elif output_path_lower.endswith('.png'):
        # PNG supports alphaand bit depths
        if isinstance(cropped, Image.Image):
            cropped.save(output_image_path, optimize=True)
            if verbose:
                print(f"    Saved as PNG ({cropped.mode})")
    elif output_path_lower.endswith(('.tiff', '.tif')):
        # TIFF supports all modes and bit depths - preserve as-is
        if isinstance(cropped, Image.Image):
            cropped.save(output_image_path, compression='none')
            if verbose:
                print(f"    Saved as TIFF ({cropped.mode})")
    elif output_path_lower.endswith('.webp'):
        # WebP - convert to RGB/RGBA if needed but preserve 8-bit
        if isinstance(cropped, Image.Image):
            if cropped.mode not in ['RGB', 'RGBA', 'L']:
                if 'A' in cropped.mode:
                    cropped = cropped.convert('RGBA')
                else:
                    cropped = cropped.convert('RGB')
            cropped.save(output_image_path, quality=100)
            if verbose:
                print(f"    Saved as WebP ({cropped.mode})")
    else:
        # Default: try to save as-is
        if isinstance(cropped, Image.Image):
            cropped.save(output_image_path, quality=100)
            if verbose:
                print(f"    Saved as {Path(output_image_path).suffix} ({cropped.mode})")
    
    # Crop and save mask if provided (masks are always 8-bit)
    if mask_image_path is not None and output_mask_path is not None:
        try:
            mask_image = Image.open(mask_image_path)
            cropped_mask = crop_direction(
                mask_image,
                direction,
                crop_size,
                fov_deg=fov_deg,
                flip_vertical=flip_vertical,
                yaw_offset=yaw_offset,
            )
            # cropped_mask should be PIL Image for masks (8-bit)
            if isinstance(cropped_mask, Image.Image):
                cropped_mask.save(output_mask_path)
        except Exception as e:
            pass
    
    # Verify output file mode by re-opening it
    # DEBUG LOG DISABLED
    
    output_name = Path(output_image_path).name
    return (direction, output_name, output_image_path, np.array([]))


def crop_direction(
    equirect_image,  # Can be PIL Image or numpy array
    direction: str,
    crop_size: int,
    fov_deg: float = 90.0,
    flip_vertical: bool = True,
    yaw_offset: float = 0.0,
):
    """Rectilinear 90° crop from equirectangular using cv2.remap (cube map layout).
    
    Extracts 6 directions (top/front/right/back/left/bottom) like a cube map unfolding.
    Preserves alpha channel and bit depth (8-bit, 16-bit, 32-bit) if present in the input image.
    
    Accepts both PIL Image and numpy array inputs (for 16-bit+ support).
    
    Args:
        yaw_offset: Additional yaw rotation in degrees to apply to the crop direction.
                   Use this to rotate the cubemap extraction angle per frame.
    """
    # Handle both PIL Image and numpy array inputs
    is_numpy_input = isinstance(equirect_image, np.ndarray)
    
    if is_numpy_input:
        # Numpy array (16-bit from OpenCV)
        equirect_np_temp = equirect_image
        height, width = equirect_image.shape[:2]
        channels = equirect_image.shape[2] if len(equirect_image.shape) == 3 else 1
        
        # Infer mode from numpy dtype and channels
        if equirect_image.dtype == np.uint16:
            if channels == 4:
                original_mode = 'RGBA;16'
            elif channels == 3:
                original_mode = 'RGB;16'
            else:
                original_mode = 'I;16'
        elif equirect_image.dtype == np.uint8:
            if channels == 4:
                original_mode = 'RGBA'
            elif channels == 3:
                original_mode = 'RGB'
            else:
                original_mode = 'L'
        else:
            original_mode = 'RGB'
    else:
        # PIL Image
        width, height = equirect_image.size
        original_mode = equirect_image.mode
        equirect_np_temp = np.array(equirect_image)
    
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

    # Apply rotation matrix for this direction (both yaw and pitch) plus yaw offset.
    R = get_direction_rotation_matrix(direction).astype(np.float32)
    
    # Apply additional yaw offset rotation
    if yaw_offset != 0.0:
        yaw_rad = np.radians(yaw_offset)
        cos_y = np.cos(yaw_rad)
        sin_y = np.sin(yaw_rad)
        R_yaw_offset = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y],
        ], dtype=np.float32)
        R = R_yaw_offset @ R
    
    dirs = dirs @ R.T

    # Convert direction vectors to equirectangular UV.
    lon = np.arctan2(dirs[..., 0], dirs[..., 2])  # [-pi, pi]
    lat = np.arctan2(dirs[..., 1], np.sqrt(dirs[..., 0] ** 2 + dirs[..., 2] ** 2))  # [-pi/2, pi/2]

    map_x = (lon / (2 * np.pi) + 0.5) * float(width)
    map_y = (0.5 - lat / np.pi) * float(height)
    if flip_vertical:
        map_y = (0.5 + lat / np.pi) * float(height)
    map_y = np.clip(map_y, 0.0, float(height - 1))

    # Determine original mode and bit depth (don't convert yet, preserve original mode)
    has_alpha = 'A' in original_mode
    is_16bit = ';16' in original_mode or 'I;16' in original_mode
    is_32bit = ';32F' in original_mode or 'F' in original_mode
    
    # Remap with bit depth preservation
    if has_alpha and len(equirect_np_temp.shape) == 3 and equirect_np_temp.shape[2] == 4:
        # Split into color and alpha channels
        color_channels = equirect_np_temp[:, :, :3]
        alpha_channel = equirect_np_temp[:, :, 3]
        
        # Remap color channels
        sampled_color = cv2.remap(
            color_channels,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        
        # Remap alpha channel
        sampled_alpha = cv2.remap(
            alpha_channel,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        
        # Combine back - convert to target dtype with proper scaling
        if is_16bit:
            # For 16-bit, clip to valid range and convert
            sampled_color = np.clip(sampled_color, 0, 65535).astype(np.uint16)
            sampled_alpha = np.clip(sampled_alpha, 0, 65535).astype(np.uint16)
        elif is_32bit:
            # For 32-bit float, ensure it's float32
            sampled_color = sampled_color.astype(np.float32)
            sampled_alpha = sampled_alpha.astype(np.float32)
        else:
            # For 8-bit, clip to valid range and convert
            sampled_color = np.clip(sampled_color, 0, 255).astype(np.uint8)
            sampled_alpha = np.clip(sampled_alpha, 0, 255).astype(np.uint8)
        
        sampled = np.dstack((sampled_color, sampled_alpha))
    else:
        # No alpha channel, just remap the color channels
        sampled = cv2.remap(
            equirect_np_temp,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        
        # Convert to target dtype with proper scaling
        if is_16bit:
            # For 16-bit, clip to valid range and convert
            sampled = np.clip(sampled, 0, 65535).astype(np.uint16)
        elif is_32bit:
            # For 32-bit float, ensure it's float32
            sampled = sampled.astype(np.float32)
        else:
            # For 8-bit, clip to valid range and convert
            sampled = np.clip(sampled, 0, 255).astype(np.uint8)
    
    # Return with original mode preserved using helper function
    return numpy_to_image_preserving_bitdepth(sampled, original_mode)


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
    num_workers: int = 4,
    skip_component_transform_for_ply: bool = True,
    skip_directions: Optional[list] = None,
    generate_masks: bool = False,
    yolo_model_path: str = "yolo11n-seg.pt",
    yolo_conf: float = 0.25,
    invert_mask: bool = False,
    yaw_offset_per_frame: float = 0.0,
    range_images: Optional[Tuple[int, int]] = None,
    yolo_classes: Optional[list] = None,
    rotate_z180: bool = False,
    mask_overexposure: bool = False,
    overexposure_threshold: int = 250,
    overexposure_dilate: int = 5,
    output_format: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert Metashape equirectangular data to COLMAP format.
    
    Args:
        skip_directions: List of directions to skip (e.g., ["bottom", "top"]).
                         Valid directions: top, front, right, back, left, bottom.
        yaw_offset_per_frame: Yaw rotation offset (degrees) to add per frame.
                              E.g., 45.0 means frame 0 has 0° offset, frame 1 has 45°, etc.
                              This can improve 3DGS training stability by diversifying view angles.
        range_images: Tuple of (start_index, end_index) to process only a range of images.
                      E.g., (10, 50) processes images from index 10 to 50 (inclusive).
                      If None, processes all images (up to max_images if specified).
        yolo_classes: List of YOLO class IDs to include in mask. If None, uses [0] (person only).
                      Common COCO classes: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck, etc.
        yolo_conf: Minimum YOLO confidence score (0.0-1.0) to keep detections.
        rotate_z180: If True, rotate the entire scene 180° around Z-axis.
                     This fixes coordinate system differences between COLMAP and PostShot.
        mask_overexposure: If True, also mask white-blown-out (overexposed) pixels.
        overexposure_threshold: Pixel value threshold (0-255) for overexposure detection.
        overexposure_dilate: Dilation radius (pixels) for overexposure mask.
        output_format: Output image format ('png', 'jpg', 'tiff', 'webp', 'auto', or None).
                      'auto' or None means use the same format as input images (default).
    """
    if output_dir is None:
        output_dir = xml_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)
    images_output_dir = output_dir / "images"
    images_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create masks directory if mask generation is enabled
    masks_output_dir = None
    tmp_masks_dir = None
    yolo_model = None
    if generate_masks:
        if not HAS_YOLO:
            raise ImportError("ultralytics is required for mask generation. Install with: pip install ultralytics")
        masks_output_dir = output_dir / "masks"
        masks_output_dir.mkdir(parents=True, exist_ok=True)
        tmp_masks_dir = output_dir / "tmp"
        tmp_masks_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Loading YOLO model: {yolo_model_path}")
        yolo_model = YOLO(yolo_model_path)

    if verbose:
        print(f"Parsing Metashape XML: {xml_path}")

    xml_data = parse_metashape_xml(xml_path)
    sensor_dict = xml_data["sensor_dict"]
    component_dict = xml_data["component_dict"]
    cameras_xml = xml_data["cameras"]

    image_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))

    image_filename_map = {img_path.stem: img_path for img_path in image_files}
    image_filename_map.update({img_path.name: img_path for img_path in image_files})

    if verbose:
        print(f"Found {len(image_files)} equirectangular images in {images_dir}")
        print(f"Component dict size: {len(component_dict)}")
        if component_dict:
            for comp_id, comp_mat in component_dict.items():
                print(f"  Component '{comp_id}': {comp_mat}")

    # Detect input image format and determine output format
    input_ext = ".jpg"  # default
    input_mode = "RGB"  # default
    input_bit_depth = 8  # default
    preserve_alpha = False
    if image_files:
        first_image = image_files[0]
        input_ext, input_mode, input_bit_depth = detect_image_format(first_image)
        # ALWAYS print this to console so user can see it even from GUI
        print(f"Detected input format: {input_ext} (mode: {input_mode}, {input_bit_depth}-bit)")
        if verbose:
            print(f"Detected input format: {input_ext} (mode: {input_mode}, {input_bit_depth}-bit)")
    
    # Get output format extension and alpha preservation setting
    output_ext, preserve_alpha = get_output_format_extension(output_format, input_ext, input_mode)
    # ALWAYS print this to console
    print(f"Output format: {output_ext}, preserve alpha: {preserve_alpha}, bit depth: {input_bit_depth}-bit")
    if verbose:
        print(f"Output format: {output_ext}, preserve alpha: {preserve_alpha}, bit depth: {input_bit_depth}-bit")


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
    all_directions = ["top", "front", "right", "back", "left", "bottom"]
    if skip_directions is None:
        skip_directions = []
    directions = [d for d in all_directions if d not in skip_directions]
    
    if verbose and skip_directions:
        print(f"  Skipping directions: {skip_directions}")
        print(f"  Using directions: {directions}")

    # Collect all crop tasks for parallel processing
    crop_tasks = []  # List of (src_image_path, base_name, direction, R_c2w, t_c2w, camera_label)
    camera_metadata = []  # Store metadata for later processing
    equirect_mask_paths = {}  # Cache for generated mask file paths {image_path: tmp_mask_path}
    equirect_images_to_process = []  # List of (src_image_path, base_name) for mask generation
    frame_index = 0  # Track frame index for yaw offset calculation
    camera_index = 0  # Track camera index for range filtering

    # Determine range bounds
    range_start = range_images[0] if range_images else 0
    range_end = range_images[1] if range_images else None

    for camera in cameras_xml.iter("camera"):
        # Check if we've reached the max_images limit
        if max_images is not None and processed_cameras >= max_images:
            break
        
        # Check if we're within the specified range
        if camera_index < range_start:
            camera_index += 1
            continue
        if range_end is not None and camera_index > range_end:
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
            if verbose and processed_cameras == 0:
                print(f"First camera '{camera_label}' component_id: {component_id} (found in component_dict)")
        elif verbose and processed_cameras == 0:
            print(f"First camera '{camera_label}' component_id: {component_id} (NOT found in component_dict)")

        src_image_path = image_filename_map[camera_label]
        try:
            # Test if image can be loaded
            test_img = Image.open(src_image_path)
            test_img.close()
        except Exception as exc:  # pragma: no cover - IO guard
            if verbose:
                print(f"  Skipping {camera_label}: failed to load image ({exc})")
            num_skipped += 1
            continue

        R_c2w = transform[:3, :3]
        t_c2w = transform[:3, 3]

        base_name = Path(camera_label).stem

        # Debug output for first valid camera
        if verbose and processed_cameras == 0:
            print(f"First valid camera '{camera_label}':")
            print(f"  Raw transform (after component): \n{transform}")
            print(f"  R_c2w:\n{R_c2w}")
            print(f"  t_c2w: {t_c2w}")

        # Collect equirectangular images for mask generation
        if generate_masks and str(src_image_path) not in equirect_mask_paths:
            equirect_images_to_process.append((str(src_image_path), base_name))

        # Calculate yaw offset for this frame
        current_yaw_offset = frame_index * yaw_offset_per_frame

        # Queue tasks for each direction
        for direction in directions:
            output_image_name = f"{base_name}_{direction}{output_ext}"
            output_image_path = str(images_output_dir / output_image_name)
            crop_tasks.append((str(src_image_path), direction, crop_size, output_image_path, fov_deg, flip_vertical, current_yaw_offset, preserve_alpha))
            camera_metadata.append((base_name, direction, R_c2w, t_c2w, current_yaw_offset, output_ext))

        processed_cameras += 1
        frame_index += 1
        camera_index += 1

    # Generate masks in parallel if requested
    if generate_masks and equirect_images_to_process:
        if verbose:
            print(f"Generating {len(equirect_images_to_process)} masks with YOLO (parallel)...")
        
        mask_generation_tasks = []
        for src_image_path, base_name in equirect_images_to_process:
            tmp_mask_name = f"{base_name}_mask.png"
            tmp_mask_path = str(tmp_masks_dir / tmp_mask_name)
            mask_generation_tasks.append((
                src_image_path, tmp_mask_path, yolo_model_path, yolo_conf, invert_mask, yolo_classes,
                mask_overexposure, overexposure_threshold, overexposure_dilate,
            ))
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for task in mask_generation_tasks:
                futures.append(
                    executor.submit(
                        generate_mask_and_save,
                        task[0],
                        task[1],
                        task[2],
                        task[3],
                        task[4],
                        task[5],
                        task[6],
                        task[7],
                        task[8],
                    )
                )

            total_masks = len(futures)
            report_interval = max(1, total_masks // 20)
            next_report = report_interval
            use_inline_progress = sys.stdout.isatty()

            if verbose and use_inline_progress:
                print(f"  Mask progress: 0/{total_masks} (0.0%)", end="\r", flush=True)
            
            for idx, future in enumerate(futures):
                try:
                    image_path, mask_path = future.result()
                    equirect_mask_paths[image_path] = mask_path
                except Exception as exc:
                    if verbose:
                        print(f"  Error generating mask {idx + 1}: {exc}")
                    continue

                completed = idx + 1
                if verbose and (completed >= next_report or completed == total_masks):
                    progress_msg = (
                        f"  Mask progress: {completed}/{total_masks} "
                        f"({(completed / total_masks) * 100:.1f}%)"
                    )
                    if use_inline_progress:
                        print(progress_msg, end="\r", flush=True)
                    else:
                        print(progress_msg)
                    while next_report <= completed:
                        next_report += report_interval

            if verbose and use_inline_progress:
                print()
        
        if verbose:
            print(f"  Completed {len(equirect_mask_paths)} masks")

    # Process crops in parallel
    if crop_tasks:
        if verbose:
            print(f"Cropping {len(crop_tasks)} images...")
        
        # Create a debug log in the output directory with main process info
        # DEBUG LOG DISABLED
        
        # Parallel processing for both RGB images and masks
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for task in crop_tasks:
                src_image_path, direction, crop_size_val, output_image_path, fov_deg_val, flip_vertical_val, yaw_offset_val, preserve_alpha_val = task
                
                # Determine mask file path if masks are enabled
                mask_file_path = None
                output_mask_path = None
                if generate_masks:
                    mask_file_path = equirect_mask_paths.get(src_image_path)
                    if mask_file_path is not None:
                        # Replace the output image extension with .png for mask
                        output_image_name = Path(output_image_path).name
                        base_output_name = output_image_name
                        for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.webp']:
                            if output_image_name.endswith(ext):
                                base_output_name = output_image_name[:-len(ext)]
                                break
                        output_mask_name = f"{base_output_name}.png"
                        output_mask_path = str(masks_output_dir / output_mask_name)
                
                futures.append(
                    executor.submit(
                        crop_and_save_image,
                        src_image_path,
                        direction,
                        crop_size_val,
                        output_image_path,
                        fov_deg_val,
                        flip_vertical_val,
                        mask_file_path,
                        output_mask_path,
                        yaw_offset_val,
                        preserve_alpha_val,
                        verbose,
                    )
                )

            total_crops = len(futures)
            # Report progress in small, readable increments (about every 5%).
            report_interval = max(1, total_crops // 20)
            next_report = report_interval
            use_inline_progress = sys.stdout.isatty()

            if verbose and use_inline_progress:
                print(f"  Cropping progress: 0/{total_crops} (0.0%)", end="\r", flush=True)
            
            for idx, future in enumerate(futures):
                try:
                    future.result()
                except Exception as exc:
                    if verbose:
                        print(f"  Error processing crop {idx + 1}: {exc}")
                    continue

                completed = idx + 1
                if verbose and (completed >= next_report or completed == total_crops):
                    progress_msg = (
                        f"  Cropping progress: {completed}/{total_crops} "
                        f"({(completed / total_crops) * 100:.1f}%)"
                    )
                    if use_inline_progress:
                        print(progress_msg, end="\r", flush=True)
                    else:
                        print(progress_msg)
                    while next_report <= completed:
                        next_report += report_interval

            if verbose and use_inline_progress:
                # Finish the inline progress line before printing subsequent logs.
                print()

    # Build images_colmap from results
    for idx, (base_name, direction, R_c2w, t_c2w, yaw_offset, output_ext) in enumerate(camera_metadata):
        output_image_name = f"{base_name}_{direction}{output_ext}"
        
        R_dir = get_direction_rotation_matrix(direction)
        
        # Apply yaw offset rotation to match the cropped image
        if yaw_offset != 0.0:
            yaw_rad = np.radians(yaw_offset)
            cos_y = np.cos(yaw_rad)
            sin_y = np.sin(yaw_rad)
            R_yaw_offset = np.array([
                [cos_y, 0, sin_y],
                [0, 1, 0],
                [-sin_y, 0, cos_y],
            ])
            R_dir = R_yaw_offset @ R_dir
        
        R_c2w_dir = R_c2w @ R_dir  # align extrinsics with the rotated crop

        R_w2c = R_c2w_dir.T
        t_w2c = -R_w2c @ t_c2w

        # Apply 180° rotation around Z-axis for PostShot coordinate system
        if rotate_z180:
            R_z180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
            R_w2c = R_w2c @ R_z180
            # t_w2c stays the same (derivation: t_new = -(R@Rz) @ (Rz@c) = -R@c = t)

        q = quaternion_from_matrix(R_w2c)

        images_colmap[image_id] = {
            "quat": q,  # [x, y, z, w]
            "tvec": t_w2c,
            "camera_id": camera_id,
            "name": output_image_name,
        }
        image_id += 1
        processed_images += 1

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

        if comp_transform is not None and not skip_component_transform_for_ply:
            if verbose:
                print(f"  Component transform being applied to points:")
                print(f"    Comp transform:\n{comp_transform}")
            points_h = np.hstack([points3d, np.ones((len(points3d), 1))])
            points3d_original = points3d.copy()
            points3d = (comp_transform @ points_h.T).T[:, :3]
            if verbose:
                print(f"    First 3 points before: {points3d_original[:3]}")
                print(f"    First 3 points after: {points3d[:3]}")
        elif skip_component_transform_for_ply and verbose:
            print(f"  Skipping component transform for PLY (--skip-component-transform-for-ply enabled)")

        # Apply 180° rotation around Z-axis for PostShot coordinate system
        if rotate_z180:
            R_z180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
            points3d = (R_z180 @ points3d.T).T
            if verbose:
                print("  Applied 180° Z-axis rotation to point cloud (for PostShot)")

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
        batch_size = 10000
        point_lines = []
        for pid, x, y, z, r, g, b, err, track in points3d_data:
            point_lines.append(f"{pid} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {err} {track}\n")
            if len(point_lines) >= batch_size:
                f.write("".join(point_lines))
                point_lines.clear()

        if point_lines:
            f.write("".join(point_lines))

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


def load_config(config_path: Path = Path("config.txt")) -> Dict[str, Any]:
    """Load configuration from config.txt file.
    
    The config file should be in simple key=value format:
        images=./equirect/
        xml=./cameras.xml
        output=./colmap_dataset/
        ply=./dense.ply
        crop-size=1920
        fov-deg=90.0
        flip-vertical=True
        max-images=10000
        num-workers=4
        apply-component-transform-for-ply=False
        skip-directions=bottom
        generate-masks=False
        yolo-model=yolo11n-seg.pt
        yolo-classes=0
        yolo-conf=0.25
        invert-mask=False
        yaw-offset=0.0
        range-images=10-50
        rotate-z180=True
        quiet=False
    
    Paths with spaces should be enclosed in quotes (single or double):
        images="D:\\My Documents\\equirect frames"
        xml='C:/Program Files/project/cameras.xml'
    
    Quotes are automatically removed during parsing.
    """
    config = {}
    if not config_path.exists():
        return config
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                
                # Parse key=value
                if "=" not in line:
                    print(f"Warning: Ignoring malformed line {line_num} in config.txt: {line}")
                    continue
                
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                
                # Remove surrounding quotes if present (both single and double quotes)
                # This allows paths with spaces to be properly handled
                if value and len(value) >= 2:
                    if (value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"):
                        value = value[1:-1]
                
                # Convert value to appropriate type
                if value.lower() in ("true", "yes", "1"):
                    config[key] = True
                elif value.lower() in ("false", "no", "0", ""):
                    config[key] = False
                elif value:
                    config[key] = value
    except Exception as e:
        print(f"Warning: Failed to read config.txt: {e}")
    
    return config


def main() -> int:
    # Ensure logs flush promptly when this CLI is launched from the GUI.
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True, write_through=True)
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass

    # Load config.txt first
    config = load_config()
    
    parser = argparse.ArgumentParser(
        description="Convert Metashape equirectangular XML to COLMAP format. "
                    "Options can be specified via config.txt file (key=value format). "
                    "Command-line arguments take precedence over config.txt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Apply config.txt values as defaults, convert Path types
    parser.add_argument(
        "--images",
        type=Path,
        required="images" not in config,
        default=Path(config["images"]) if "images" in config else None,
        help="Directory with equirectangular images"
    )
    parser.add_argument(
        "--xml",
        type=Path,
        required="xml" not in config,
        default=Path(config["xml"]) if "xml" in config else None,
        help="Path to Metashape cameras.xml"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required="output" not in config,
        default=Path(config["output"]) if "output" in config else None,
        help="Output directory (COLMAP layout)"
    )
    parser.add_argument(
        "--ply",
        type=Path,
        default=Path(config["ply"]) if "ply" in config else None,
        help="Optional PLY to export points3D.txt"
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=int(config["crop-size"]) if "crop-size" in config else 1920,
        help="Crop size for 90° views"
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=float(config["fov-deg"]) if "fov-deg" in config else 90.0,
        help="Horizontal FoV for rectilinear crops"
    )
    parser.add_argument(
        "--flip-vertical",
        action="store_true",
        default=config.get("flip-vertical", True) if isinstance(config.get("flip-vertical"), bool) else True,
        help="Flip vertical direction (invert latitude) when sampling equirect (default: on)",
    )
    parser.add_argument(
        "--no-flip-vertical",
        action="store_false",
        dest="flip_vertical",
        help="Disable vertical flip if your data is already upright",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=int(config["max-images"]) if "max-images" in config else 10000,
        help="Optional limit on number of equirectangular images to process (for quick tests)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(config["num-workers"]) if "num-workers" in config else 4,
        help="Number of worker processes for parallel image cropping"
    )
    parser.add_argument(
        "--apply-component-transform-for-ply",
        action="store_true",
        default=config.get("apply-component-transform-for-ply", False) if isinstance(config.get("apply-component-transform-for-ply"), bool) else False,
        help="Apply component transform for PLY (default: disabled, as PLY is usually pre-transformed in Metashape)"
    )
    parser.add_argument(
        "--skip-directions",
        type=str,
        default=config.get("skip-directions", ""),
        help="Comma-separated list of directions to skip (e.g., 'bottom' or 'top,bottom'). Valid: top,front,right,back,left,bottom"
    )
    parser.add_argument(
        "--generate-masks",
        action="store_true",
        default=config.get("generate-masks", False) if isinstance(config.get("generate-masks"), bool) else False,
        help="Generate person masks using YOLO and crop them alongside images"
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default=config.get("yolo-model", "yolo11n-seg.pt"),
        help="YOLO model path for mask generation (default: yolo11n-seg.pt)"
    )
    parser.add_argument(
        "--yolo-classes",
        type=str,
        default=config.get("yolo-classes", "0"),
        help="Comma-separated YOLO class IDs to include in mask (default: 0 for person). Common COCO classes: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck. Example: 0,2,5 for person, car, and bus."
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=float(config["yolo-conf"]) if "yolo-conf" in config else 0.25,
        help="Minimum YOLO confidence score (0.0-1.0) to keep detections (default: 0.25)"
    )
    parser.add_argument(
        "--invert-mask",
        action="store_true",
        default=config.get("invert-mask", False) if isinstance(config.get("invert-mask"), bool) else False,
        help="Use object=white(255), background=black(0). Default: object=black(0), background=white(255) for 3DGS training"
    )
    parser.add_argument(
        "--yaw-offset",
        type=float,
        default=float(config["yaw-offset"]) if "yaw-offset" in config else 0.0,
        help="Yaw rotation offset (degrees) to add per frame. E.g., 45.0 rotates cubemap extraction by 45° for each successive frame. This can improve 3DGS training stability by diversifying view angles."
    )
    parser.add_argument(
        "--rotate-z180",
        action="store_true",
        default=config.get("rotate-z180", True) if isinstance(config.get("rotate-z180"), bool) else True,
        help="Rotate the entire scene 180° around Z-axis for PostShot coordinate system compatibility (default: on)"
    )
    parser.add_argument(
        "--no-rotate-z180",
        action="store_false",
        dest="rotate_z180",
        help="Disable Z-axis 180° rotation"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=config.get("quiet", False) if isinstance(config.get("quiet"), bool) else False,
        help="Suppress progress output"
    )
    parser.add_argument(
        "--range-images",
        type=str,
        default=config.get("range-images"),
        help="Range of images to process (format: START-END, e.g., 10-50). Processes images from START to END (inclusive, 0-based index)."
    )
    parser.add_argument(
        "--mask-overexposure",
        action="store_true",
        default=config.get("mask-overexposure", False) if isinstance(config.get("mask-overexposure"), bool) else False,
        help="Also mask white-blown-out (overexposed) pixels in the equirectangular images"
    )
    parser.add_argument(
        "--overexposure-threshold",
        type=int,
        default=int(config["overexposure-threshold"]) if "overexposure-threshold" in config else 250,
        help="Pixel value threshold (0-255) above which all-channel pixels are considered overexposed (default: 250)"
    )
    parser.add_argument(
        "--overexposure-dilate",
        type=int,
        default=int(config["overexposure-dilate"]) if "overexposure-dilate" in config else 5,
        help="Dilation radius in pixels for the overexposure mask to cover fringe artifacts (default: 5)"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default=config.get("output-format", "auto"),
        choices=["auto", "jpg", "jpeg", "png", "tiff", "tif", "webp"],
        help="Output image format. 'auto' (default) uses the same format as input images, preserving bit depth and alpha channel when applicable."
    )

    args = parser.parse_args()
    
    # Parse yolo-classes if specified
    yolo_classes = None
    if args.yolo_classes:
        try:
            yolo_classes = [int(x.strip()) for x in args.yolo_classes.split(",")]
            if any(c < 0 for c in yolo_classes):
                print(f"Error: All YOLO class IDs must be non-negative integers.")
                return 1
        except ValueError:
            print(f"Error: Invalid yolo-classes format. Use comma-separated integers (e.g., 0,2,5).")
            return 1

    if not (0.0 <= args.yolo_conf <= 1.0):
        print("Error: --yolo-conf must be in range [0.0, 1.0].")
        return 1
    
    # Parse skip-directions if specified
    valid_directions = {"top", "front", "right", "back", "left", "bottom"}
    skip_directions_list = None
    if args.skip_directions:
        skip_directions_list = [d.strip().lower() for d in args.skip_directions.split(",") if d.strip()]
        invalid_dirs = set(skip_directions_list) - valid_directions
        if invalid_dirs:
            print(f"Error: Invalid directions: {invalid_dirs}. Valid: {valid_directions}")
            return 1
    
    # Parse range-images if specified
    range_images = None
    if args.range_images:
        try:
            if "-" in args.range_images:
                start_str, end_str = args.range_images.split("-", 1)
                range_start = int(start_str.strip())
                range_end = int(end_str.strip())
                if range_start < 0 or range_end < range_start:
                    print(f"Error: Invalid range-images format. START must be >= 0 and END must be >= START.")
                    return 1
                range_images = (range_start, range_end)
            else:
                print(f"Error: Invalid range-images format. Use START-END format (e.g., 10-50).")
                return 1
        except ValueError:
            print(f"Error: Invalid range-images format. Both START and END must be integers.")
            return 1

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
            num_workers=args.num_workers,
            skip_component_transform_for_ply=not args.apply_component_transform_for_ply,
            verbose=not args.quiet,
            skip_directions=skip_directions_list,
            generate_masks=args.generate_masks,
            yolo_model_path=args.yolo_model,
            yolo_conf=args.yolo_conf,
            invert_mask=args.invert_mask,
            yaw_offset_per_frame=args.yaw_offset,
            range_images=range_images,
            yolo_classes=yolo_classes,
            rotate_z180=args.rotate_z180,
            mask_overexposure=args.mask_overexposure,
            overexposure_threshold=args.overexposure_threshold,
            overexposure_dilate=args.overexposure_dilate,
            output_format=args.output_format,
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
    multiprocessing.freeze_support()
    raise SystemExit(main())
