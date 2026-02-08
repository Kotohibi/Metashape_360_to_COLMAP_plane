# Cubemap Image Decimation Tool

Intelligently decimate (thin out) cubemap images for 3D Gaussian Splatting (3DGS) training based on view frustum point cloud overlap and camera baseline distance.

## Overview

This tool removes redundant frames from a COLMAP dataset while ensuring 3D reconstruction quality by:
- Analyzing view frustum overlap (which 3D points are visible in each frame)
- Checking camera baseline distance (how far the camera moved)
- Guaranteeing minimum observation counts for all 3D points

## Algorithm (改善案A)

1. **Group images by frame**: Cubemap images are grouped by frame (6 directions per frame: top/front/right/back/left/bottom)
2. **Calculate visibility**: For each direction, determine which 3D points are visible in the view frustum
3. **Compare frames**: Within a window, compare each frame with neighboring frames
   - Calculate overlap ratio: `|intersection| / |current_frame_points|`
   - Calculate baseline distance: `||camera_pos_curr - camera_pos_other||`
4. **Identify redundant frames**: Frames with **high overlap AND small baseline** are candidates for removal
5. **Ensure coverage**: Remove frames only if all 3D points maintain minimum observation count

## Installation

```bash
pip install numpy open3d
```

## Usage

### Basic Usage

```bash
python decimate_cubemap_images.py \
    --input ./colmap_dataset/ \
    --output ./decimated_dataset/ \
    --overlap-threshold 0.5 \
    --baseline-threshold 0.5 \
    --min-observations 3
```

### Using Configuration File

1. Copy the example configuration:
   ```bash
   cp config_decimate.txt.example config_decimate.txt
   ```

2. Edit `config_decimate.txt` with your settings

3. Run without arguments (reads from config):
   ```bash
   python decimate_cubemap_images.py
   ```

## Parameters

### Core Decimation Parameters

#### `overlap-threshold` (0.0-1.0)
Frames with average overlap **greater than** this value are candidates for removal.

- **Lower value** = more aggressive decimation (keep fewer frames)
  - Example: `0.3` → frames with 30%+ overlap are candidates
- **Higher value** = more conservative decimation (keep more frames)
  - Example: `0.9` → only frames with 90%+ overlap are candidates
- **Default**: `0.8`

#### `baseline-threshold` (world units)
Frames with camera movement **less than** this value are candidates for removal.

- **Larger value** = more aggressive decimation (keep fewer frames)
  - Example: `5.0` → cameras within 5 units are "close"
- **Smaller value** = more conservative decimation (keep more frames)
  - Example: `0.1` → only nearly stationary cameras are "close"
- **Default**: `0.1` (adjust based on scene scale)
- **Scene scale guide**:
  - Indoor (few meters): `0.1 - 0.3`
  - Building exterior (tens of meters): `0.5 - 2.0`
  - Large landscape (hundreds of meters): `5.0 - 10.0`

⚠️ **Important**: A frame is considered redundant only when **BOTH** conditions are true:
```
overlap > overlap_threshold  AND  baseline < baseline_threshold
```

#### `min-observations` (integer)
Minimum number of frames that must observe each 3D point.

- **Higher value** = safer for reconstruction quality, less decimation
  - Example: `5` → each point seen by at least 5 frames
- **Lower value** = more aggressive decimation, riskier
  - Example: `1` → each point seen by at least 1 frame
- **Default**: `3`

#### `window-size` (integer)
Number of frames before/after each frame to compare.

- **`1`** (default) = compare only with adjacent frames (fast)
  - frame_005 compares with: frame_004, frame_006
- **`3`** = compare with +/-3 frames (more aggressive)
  - frame_005 compares with: frame_002, 003, 004, 006, 007, 008
- **`5`** = compare with +/-5 frames (most aggressive, slower)

Higher values find more redundancy but increase processing time linearly.

### Direction Control

#### `skip-directions`
Comma-separated list of cubemap directions to skip.

Valid directions: `top`, `front`, `right`, `back`, `left`, `bottom`

Examples:
```bash
--skip-directions bottom           # Skip only bottom
--skip-directions top,bottom       # Skip top and bottom
```

### Performance Parameters

#### `num-workers` (integer)
Number of parallel worker processes for visibility calculation (CPU-intensive 3D projection).

- **Default**: `4`
- **Recommended**: Set to your CPU core count (e.g., `16` for 16-core CPU)

#### `max-depth` (world units)
Maximum depth for point visibility check. Points beyond this distance are not considered visible.

- **Default**: `100.0`
- Adjust based on scene scale

### Caching Options

#### `--use-cache`
Use cached visibility data without prompting (if available and compatible).

#### `--recalculate`
Force recalculation of visibility data, ignoring cache.

#### `--cache-file` (path)
Specify custom cache file location.
- **Default**: `<input_dir>/visibility_cache.pkl`

### Output Control

#### `--no-copy-images`
Don't copy image files to output directory (only write COLMAP txt files).

#### `--no-stats`
Suppress statistics output for threshold tuning.

#### `--quiet`
Suppress all progress output.

## Configuration File Format

```ini
# config_decimate.txt
input=./colmap_dataset/
output=./decimated_dataset/

overlap-threshold=0.8
baseline-threshold=0.5
min-observations=3
window-size=1

skip-directions=
no-copy-images=False
max-depth=100.0
num-workers=8
quiet=False
```

## Statistics Output

When running, the tool displays helpful statistics:

```
=== Frame Pair Statistics (for threshold tuning) ===
  Overlap ratio (higher = more similar):
    Min: 0.1234, Max: 0.7890, Avg: 0.4500
    Percentiles: 25%=0.30, 50%=0.45, 75%=0.60
  Baseline distance (lower = camera moved less):
    Min: 0.0500, Max: 2.3000, Avg: 0.8000
    Percentiles: 25%=0.20, 50%=0.50, 75%=1.00

  Sample frame pairs (first 5):
    frame_001: overlap=0.45, baseline=0.80
    frame_002: overlap=0.52, baseline=0.75 REDUNDANT
    ...

  Threshold analysis:
    overlap>0.3, baseline<0.0625: 12 frames (20.0%)
    overlap>0.5, baseline<0.2500: 8 frames (13.3%)
    overlap>0.7, baseline<1.0000: 3 frames (5.0%)
```

Use this information to tune your thresholds for desired decimation rate.

## Workflow

### Step 1: Initial Run with Statistics
```bash
python decimate_cubemap_images.py \
    --input ./data/sparse/0 \
    --output ./data/decimated/sparse/0 \
    --overlap-threshold 0.8 \
    --baseline-threshold 0.5 \
    --min-observations 3 \
    --recalculate
```

### Step 2: Analyze Statistics
Check the output statistics to understand:
- Typical overlap ratios in your dataset
- Typical baseline distances (adjust for scene scale!)
- Threshold analysis showing reduction rates

### Step 3: Tune Parameters
Based on statistics, adjust thresholds:
- Want 30% reduction? Find thresholds that match in analysis
- Scene too large? Increase `baseline-threshold`
- Scene too small? Decrease `baseline-threshold`

### Step 4: Run with Cache
```bash
python decimate_cubemap_images.py --use-cache
```

Visibility calculation is cached, so parameter tuning is fast!

## Examples

### Conservative Decimation (High Quality)
```bash
python decimate_cubemap_images.py \
    --overlap-threshold 0.9 \
    --baseline-threshold 0.1 \
    --min-observations 5
```
Removes only highly redundant frames with 90%+ overlap.

### Aggressive Decimation (Speed Priority)
```bash
python decimate_cubemap_images.py \
    --overlap-threshold 0.3 \
    --baseline-threshold 2.0 \
    --min-observations 1 \
    --window-size 3
```
Removes frames with 30%+ overlap and cameras within 2 units.

### Skip Bottom Direction
```bash
python decimate_cubemap_images.py \
    --skip-directions bottom
```
Useful if bottom images contain unwanted artifacts.

## Tips & Tricks

### 1. Use Statistics to Set Thresholds
Always run once with default settings and examine the statistics output. The threshold analysis tells you exactly how many frames will be removed at different settings.

### 2. Scene Scale Matters
The `baseline-threshold` is in world coordinate units. A threshold of `0.1` might mean:
- 10cm in a room scene
- 10m in a city scene
- 100m in a landscape scene

Check your COLMAP model scale first!

### 3. Cache Visibility for Fast Iteration
Visibility calculation is slow (minutes to hours). But it only depends on:
- `max-depth`
- `skip-directions`
- Point cloud and camera poses

Once calculated, you can tune `overlap-threshold`, `baseline-threshold`, `min-observations`, and `window-size` instantly using the cache.

### 4. Window Size Trade-offs
- `window-size=1`: Fast, catches adjacent redundant frames
- `window-size=3-5`: Better for videos with gradual movement
- `window-size>5`: Slower, diminishing returns

### 5. Check Reduction Rate
Aim for:
- **10-30% reduction**: Safe for most cases
- **30-50% reduction**: Acceptable if statistics look good
- **>50% reduction**: Check carefully, may lose quality

## Troubleshooting

### "Processing is stuck / very slow"
- Check if visibility calculation is running (CPU usage on all cores)
- For large datasets (>1000 frames), visibility calculation takes time
- Use `--use-cache` after first run

### "No frames removed"
Your thresholds are too strict. Check statistics output:
- Is your `overlap-threshold` too high? (Try lower values like 0.3-0.5)
- Is your `baseline-threshold` too small for your scene scale? (Try larger values)

### "Too many frames removed"
Your thresholds are too loose:
- Increase `overlap-threshold` (e.g., 0.7-0.9)
- Decrease `baseline-threshold`
- Increase `min-observations` (e.g., 5-10)

### "Cache incompatible" warning
Cache parameters don't match current settings. The tool will recalculate automatically. Use `--recalculate` to force this.

## Output

The tool creates a decimated COLMAP dataset:
```
output_dir/
├── cameras.txt          # Same as input
├── images.txt           # Only kept frames
├── points3D.txt         # Same as input (all points preserved)
├── points3D.ply         # Same as input (if exists)
├── images/              # Only kept images (if --no-copy-images not set)
│   ├── frame_001_front.jpg
│   ├── frame_001_right.jpg
│   └── ...
└── masks/               # Only kept masks (if input has masks/)
```

## Performance

Typical processing times (on 16-core CPU):

| Frames | Visibility Calc | Frame Comparison | Total |
|--------|----------------|------------------|-------|
| 50 | ~2 min | <1 sec | ~2 min |
| 100 | ~5 min | ~2 sec | ~5 min |
| 500 | ~20 min | ~10 sec | ~20 min |
| 1000 | ~40 min | ~30 sec | ~40 min |

**With cache, re-running with different thresholds takes seconds!**

## License

Same as parent project.
