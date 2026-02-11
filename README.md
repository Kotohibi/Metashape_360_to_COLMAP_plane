# Metashape 360° to COLMAP Converter

English | 日本語

## Overview / 概要
Convert Agisoft Metashape equirectangular (spherical) camera exports into COLMAP text format, while generating rectilinear crops (top/front/right/back/left/bottom) from each 360° frame. Optional PLY is converted to points3D.txt (via Open3D).

Refer to my X
- https://x.com/naribubu/status/2015376645360849394
- https://x.com/naribubu/status/2016517770876440901
- https://x.com/naribubu/status/2020138127084695876
- https://x.com/naribubu/status/2017883648075391214 (As for --yaw-offset otpion)

Refer to other URL
- https://lilea.net/lab/equirectangular-3dgs-with-licht-feld-studio/
- https://note.com/grand_loris6426/n/ne67d281505d5
- https://x.com/migero_usausagi/status/2017850407884632298 (As for --yaw-offset otpion)

## Features / 特長
- Equirectangular → Cubemap, 6 rectilinear 90° crops per frame (top/front/right/back/left/bottom), multi-process available
- Writes COLMAP `cameras.txt`, `images.txt`, `points3D.txt`
- Optional PLY transform/export (needs Open3D)
- Adjustable FoV and crop size; vertical flip for sampling equirect
- Optional image-count cap for quick tests
- Generate masks for human

## Requirements / 必要環境
- Metashape Standard (https://www.agisoft.com/features/standard-edition/)
- Python 3.9+
- pip: `numpy`, `pillow`, `opencv-python`
- Optional: `ultralytics`(for generate mask for human by YOLO), `open3d` (for PLY → points3D)

## Usage / 使い方
### SfM with Camera type as Spherical in Metashape
- [Tools] -> [Camera Calibration] -> [Camera type] -> [Spherical]
### Export Cameras as XML and Point cloud as PLY files from Metashape
- [File] -> [Export Cameras...] -> select xml type
- [File] -> [Export Point Cloud...] -> select ply type
### Python CLI example
```bash
python metashape_360_to_colmap.py \
  --images /path/to/equirect_frames \
  --xml /path/to/metashape_cameras.xml \ # Specify xml exported from Metashape
  --output /path/to/output_colmap \
  --ply /path/to/pointcloud.ply \ # Specify ply exported from Metashape
  --crop-size 1920 \ 
  --fov-deg 90 \
  --num-workers 4 \
  --max-images 50 \ # If you test quickly, specify small number. default 10000
  --yaw-offset 30 \ # If needed, rotate cubemap for each extraction to be more stable for 3DGS. default 0
  --generate-masks \ # Generate masks for specified objects
  --yolo-classes 0,2,5 # Mask person (0), car (2), and bus (5). Default: 0 (person only)
```

### Using Configuration File / 設定ファイルの使用
You can specify options in `config.txt` file instead of command-line arguments. Create a `config.txt` in the same directory as the script:

```
# config.txt example
images=./equirect/
xml=./cameras.xml
output=./colmap_dataset/
ply=./dense.ply
crop-size=1920
fov-deg=90.0
num-workers=4
max-images=10000
yaw-offset=30
generate-masks=True
```

**For paths with spaces (Windows users):** Enclose paths in quotes (single or double):
```
images="D:\My Documents\equirect frames"
xml="C:\Program Files\project\cameras.xml"
output='D:\Output Folder\colmap'
```

**Priority:** Command-line arguments > config.txt > default values

If you specify an option on the command line, it will override the value in config.txt. See [config.txt.example](config.txt.example) for all available options.

### Key options / 主なオプション
- `--images` (req): Directory of equirectangular images
- `--xml` (req): Metashape XML export (cameras)
- `--output` (req): Output folder (COLMAP text + crops in `images/`)
- `--ply`: Optional PLY to export `points3D.txt` and `points3D.ply`
- `--crop-size`: Crop resolution (square). Default 1920.
- `--fov-deg`: Horizontal FoV of rectilinear crops. Default 90.
- `--max-images`: Limit number of source equirects for quick tests (default 10000)
- `--range-images`: Range of images to process (format: `START-END`, e.g., `10-50`). Processes images from START to END (inclusive, 0-based index). Useful for processing specific subsets of images.
- `--num-workers`: Number of process for image reframing (default 4)
- `--skip-directions=`:Comma-separated list of directions to skip (top, front, right, back, left, bottom)
- `--generate-masks` : Generate masks for specified objects using YOLO
- `--yolo-classes`: Comma-separated YOLO class IDs to include in mask (default: 0 for person only). Common COCO classes: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck. Example: `--yolo-classes 0,2,5` for person, car, and bus.
- `--invert-mask` : Invert mask color from BLACK to WHITE
- `--yaw-offset`: Yaw rotation offset (degrees) per frame. E.g., `45.0` rotates cubemap extraction by 45° for each successive frame. This can improve 3DGS training stability by diversifying view angles. (default 0.0) 

### Outputs / 出力
- `output/ images/`: Cropped images (4 per input frame)
- `output/ masks/`: Mask images for specified objects if the option (--generate-masks) is specified
- `output/ tmp`: tmp folder for generating mask process. Able to delete after finishing
- `output/ cameras.txt`
- `output/ images.txt`
- `output/ points3D.txt` (+ `points3D.ply` when PLY given)

### YOLO COCO Classes Reference / YOLOクラスID参考
Common COCO dataset class IDs for `--yolo-classes`:
- 0: person
- 1: bicycle
- 2: car
- 3: motorcycle
- 5: bus
- 7: truck
- 16: dog
- 17: cat
- [Full COCO class list](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)

## Notes / 補足
- I confirmed that it worked with PostShot for 3DGS train.
- Only spherical sensors are supported; uses the first component transform when multiple are present.
- Intrinsics per crop are PINHOLE with `fx=fy=(w/2)/tan(fov/2)`, `cx=cy=w/2`.
- If orientations look wrong, verify top/front/right/back/left/bottom yaw definitions and FoV.

## License / ライセンス
MIT
