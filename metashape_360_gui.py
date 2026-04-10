#!/usr/bin/env python3
"""
Metashape 360° to COLMAP Converter - GUI Application

A graphical user interface for the metashape_360_to_colmap.py script.
Provides easy configuration and execution of the conversion process.
"""

import os
import sys
import subprocess
import threading
import queue
import multiprocessing
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path

# Default values matching config.txt.example
DEFAULTS = {
    "images": "./equirect/",
    "xml": "./cameras.xml",
    "ply": "./sparse_point_cloud.ply",
    "output": "./colmap_dataset/",
    "crop_size": 1920,
    "fov_deg": 90.0,
    "max_images": 10000,
    "range_start": 1,
    "range_end": 10000,
    "num_workers": 4,
    "generate_masks": False,
    "invert_mask": False,
    "yaw_offset": 0.0,
    "quiet": False,
    "yolo_classes": "0",
    "yolo_conf": 0.25,
    "mask_overexposure": False,
    "overexposure_threshold": 250,
    "overexposure_dilate": 5,
    "yolo_model": "yolo11m-seg.pt",
    "apply_component_transform": False,
    "flip_vertical": True,
    "rotate_z180": False,
    "language": "EN",
    "output_format": "auto",
}

VALID_DIRECTIONS = ["top", "front", "right", "back", "left", "bottom"]
YOLO_SEG_MODELS = [
    "yolo11n-seg.pt",
    "yolo11s-seg.pt",
    "yolo11m-seg.pt",
    "yolo11l-seg.pt",
    "yolo11x-seg.pt",
]

UI_TEXT = {
    "EN": {
        "app_title": "Metashape 360° to COLMAP Converter",
        "language": "Language:",
        "paths_section": "Input/Output Paths",
        "images_label": "Input Images Folder:",
        "xml_label": "Metashape XML:",
        "ply_label": "PLY File (Optional):",
        "output_label": "Output Folder:",
        "browse": "Browse...",
        "proc_section": "Processing Options",
        "crop_size": "Crop Size:",
        "fov": "FoV:",
        "max_images": "Max Images:",
        "workers": "Workers:",
        "range": "Image Range:",
        "range_start": "Start:",
        "range_end": "End:",
        "yaw_offset": "Yaw Offset:",
        "output_format": "Output Format:",
        "skip_section": "Skip Directions",
        "skip_note": "(Skip cubemap generation for selected directions)",
        "mask_section": "Mask Generation",
        "enable_yolo": "Enable YOLO mask generation",
        "yolo_classes": "YOLO Class IDs:",
        "yolo_conf": "YOLO Confidence:",
        "yolo_model": "YOLO Model:",
        "invert_mask": "Invert mask (object=white)",
        "enable_overexp": "Enable overexposure mask",
        "threshold": "Threshold (0-255):",
        "dilate": "Dilation Radius:",
        "advanced_section": "Dev Options(Don't change)",
        "advanced_show": "Show Dev Options",
        "advanced_hide": "Hide Dev Options",
        "flip_vertical": "Flip vertical (for equirect sampling)",
        "rotate_z180": "Rotate 180° around Z-axis (PostShot compatibility)",
        "apply_component": "Apply component transform for PLY",
        "quiet": "Quiet mode (suppress progress output)",
        "save_cfg": "Save Config",
        "load_cfg": "Load Config",
        "reset_defaults": "Reset Defaults",
        "run": "▶ Run Conversion",
        "stop": "■ Stop",
        "console_section": "Output Log",
        "clear_log": "Clear Log",
        "err_title": "Input Error",
        "err_images_required": "Please specify input images folder",
        "err_images_missing": "Input images folder not found: {path}",
        "err_xml_required": "Please specify Metashape XML file",
        "err_xml_missing": "XML file not found: {path}",
        "err_output_required": "Please specify output folder",
        "err_ply_missing": "PLY file not found: {path}",
        "cmd": "Command: {cmd}",
        "done_ok": "✓ Conversion completed successfully.",
        "done_ng": "✗ Conversion exited with error code {code}.",
        "error": "Error: {error}",
        "stopped": "Processing stopped.",
        "exit_confirm_title": "Confirm Exit",
        "exit_confirm_running": "A conversion is still running. Exit and force-stop the CLI process?",
        "reset_done": "Settings reset to defaults.",
        "saved": "Config saved: {path}",
        "auto_loaded": "Auto-loaded config.txt.",
        "load_error": "Config load error: {error}",
        "loaded": "Config loaded: {path}",
        "tip_images": "Folder containing equirectangular images",
        "tip_xml": "cameras.xml exported from Metashape",
        "tip_ply": "Point cloud file exported from Metashape (optional)",
        "tip_output": "Destination folder for COLMAP dataset output",
        "tip_crop_size": "Output image size (pixels)",
        "tip_fov": "Horizontal field of view for rectilinear crops (degrees)",
        "tip_max_images": "Maximum number of images to process (for testing)",
        "tip_workers": "Number of worker processes for parallel processing",
        "tip_range": "Index range of images to process (0-based)",
        "tip_yaw_offset": "Per-frame yaw rotation offset (degrees)",
        "tip_output_format": "Output image format. 'auto' uses the same format as input images (default), preserving bit depth and alpha channel",
        "tip_yolo_classes": "Comma-separated class IDs (0=person, 2=car, 3=motorcycle, 5=bus, 7=truck)",
        "tip_yolo_conf": "Minimum YOLO confidence score to keep detections (0.0-1.0)",
        "tip_overexp_threshold": "Treat pixels as overexposed when all RGB channels exceed this value",
        "tip_overexp_dilate": "Dilation amount to cover fringe artifacts around masks (pixels)",
        "tip_flip_vertical": "Flip Y-axis during equirectangular image sampling",
    },
    "JP": {
        "app_title": "Metashape 360° to COLMAP コンバーター",
        "language": "言語:",
        "paths_section": "入力/出力パス",
        "images_label": "入力画像フォルダ:",
        "xml_label": "Metashape XML:",
        "ply_label": "PLYファイル (任意):",
        "output_label": "出力フォルダ:",
        "browse": "参照...",
        "proc_section": "処理オプション",
        "crop_size": "クロップサイズ:",
        "fov": "視野角 (FoV):",
        "max_images": "最大画像数:",
        "workers": "ワーカー数:",
        "range": "画像範囲指定:",
        "range_start": "開始:",
        "range_end": "終了:",
        "yaw_offset": "Yawオフセット:",
        "output_format": "出力形式:",
        "skip_section": "スキップ方向",
        "skip_note": "(選択した方向のCubemap生成をスキップします)",
        "mask_section": "マスク生成",
        "enable_yolo": "YOLOマスク生成を有効化",
        "yolo_classes": "YOLOクラスID:",
        "yolo_conf": "YOLO信頼度閾値:",
        "yolo_model": "YOLOモデル:",
        "invert_mask": "マスク反転 (物体=白)",
        "enable_overexp": "露出オーバーマスクを有効化",
        "threshold": "閾値 (0-255):",
        "dilate": "膨張半径:",
        "advanced_section": "開発者向けオプション (変更不要)",
        "advanced_show": "開発者向けオプションを表示",
        "advanced_hide": "開発者向けオプションを隠す",
        "flip_vertical": "垂直フリップ (Equirect用)",
        "rotate_z180": "Z軸180°回転 (PostShot互換)",
        "apply_component": "PLYにコンポーネント変換を適用",
        "quiet": "静音モード (進捗非表示)",
        "save_cfg": "設定を保存",
        "load_cfg": "設定を読み込み",
        "reset_defaults": "デフォルトに戻す",
        "run": "▶ 変換を実行",
        "stop": "■ 停止",
        "console_section": "出力ログ",
        "clear_log": "ログをクリア",
        "err_title": "入力エラー",
        "err_images_required": "入力画像フォルダを指定してください",
        "err_images_missing": "入力画像フォルダが見つかりません: {path}",
        "err_xml_required": "Metashape XMLファイルを指定してください",
        "err_xml_missing": "XMLファイルが見つかりません: {path}",
        "err_output_required": "出力フォルダを指定してください",
        "err_ply_missing": "PLYファイルが見つかりません: {path}",
        "cmd": "実行コマンド: {cmd}",
        "done_ok": "✓ 変換が正常に完了しました。",
        "done_ng": "✗ 変換がエラーコード {code} で終了しました。",
        "error": "エラー: {error}",
        "stopped": "処理を停止しました。",
        "exit_confirm_title": "終了確認",
        "exit_confirm_running": "変換処理が実行中です。終了してCLIプロセスを強制停止しますか？",
        "reset_done": "設定をデフォルトに戻しました。",
        "saved": "設定を保存しました: {path}",
        "auto_loaded": "config.txt を自動読み込みしました。",
        "load_error": "設定ファイル読み込みエラー: {error}",
        "loaded": "設定を読み込みました: {path}",
        "tip_images": "Equirectangular画像が含まれるフォルダ",
        "tip_xml": "Metashapeからエクスポートしたcameras.xml",
        "tip_ply": "Metashapeからエクスポートした点群ファイル (任意)",
        "tip_output": "COLMAP形式のデータセット出力先",
        "tip_crop_size": "出力画像のサイズ (pixels)",
        "tip_fov": "Rectilinear cropsの水平視野角 (degrees)",
        "tip_max_images": "処理する最大画像数 (テスト用)",
        "tip_workers": "並列処理のワーカープロセス数",
        "tip_range": "処理する画像のインデックス範囲 (0ベース)",
        "tip_yaw_offset": "フレームごとのYaw回転オフセット (degrees)",
        "tip_output_format": "出力画像の形式。'auto'は入力画像と同じ形式を使用 (デフォルト)、ビット深度とアルファチャンネルを維持",
        "tip_yolo_classes": "カンマ区切りのクラスID (0=person, 2=car, 3=motorcycle, 5=bus, 7=truck)",
        "tip_yolo_conf": "検出を採用する最小YOLO信頼度スコア (0.0-1.0)",
        "tip_overexp_threshold": "全RGBチャンネルがこの値を超えるピクセルを露出オーバーとみなす",
        "tip_overexp_dilate": "マスク周辺のフリンジアーティファクトをカバーする膨張量 (pixels)",
        "tip_flip_vertical": "Equirectangular画像サンプリング時のY軸反転",
    },
}


class ToolTip:
    """Simple tooltip implementation for Tkinter widgets."""
    
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)
    
    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, _, cy = self.widget.bbox("insert") if hasattr(self.widget, 'bbox') else (0, 0, 0, 0)
        x = x + self.widget.winfo_rootx() + 25
        y = y + cy + self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                        background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                        font=("Segoe UI", 9))
        label.pack(ipadx=4)
    
    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class Metashape360GUI:
    """Main GUI Application class."""
    
    def __init__(self, root):
        self.root = root
        self.root.geometry("850x900")
        self.root.minsize(750, 700)
        self.is_closing = False
        self.main_canvas = None
        self.main_scrollbar = None
        self.console = None
        self.options_notebook = None
        
        # Output queue for async process output
        self.output_queue = queue.Queue()
        self.process = None
        
        # Variables for all settings
        self.init_variables()
        
        # Setup UI
        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close_requested)
        
        # Load config if exists
        self.load_config_file()
        
        # Start output polling
        self.poll_output()
    
    def init_variables(self):
        """Initialize all tkinter variables."""
        # Path variables
        self.var_images = tk.StringVar(value=DEFAULTS["images"])
        self.var_xml = tk.StringVar(value=DEFAULTS["xml"])
        self.var_ply = tk.StringVar(value=DEFAULTS["ply"])
        self.var_output = tk.StringVar(value=DEFAULTS["output"])
        
        # Processing options
        self.var_crop_size = tk.IntVar(value=DEFAULTS["crop_size"])
        self.var_fov_deg = tk.DoubleVar(value=DEFAULTS["fov_deg"])
        self.var_max_images = tk.IntVar(value=DEFAULTS["max_images"])
        self.var_range_enabled = tk.BooleanVar(value=False)
        self.var_range_start = tk.IntVar(value=DEFAULTS["range_start"])
        self.var_range_end = tk.IntVar(value=DEFAULTS["range_end"])
        self.var_num_workers = tk.IntVar(value=DEFAULTS["num_workers"])
        self.var_yaw_offset = tk.DoubleVar(value=DEFAULTS["yaw_offset"])
        self.var_output_format = tk.StringVar(value=DEFAULTS["output_format"])
        
        # Direction skip checkboxes
        self.var_skip_directions = {d: tk.BooleanVar(value=False) for d in VALID_DIRECTIONS}
        
        # Mask options
        self.var_generate_masks = tk.BooleanVar(value=DEFAULTS["generate_masks"])
        self.var_invert_mask = tk.BooleanVar(value=DEFAULTS["invert_mask"])
        self.var_yolo_classes = tk.StringVar(value=DEFAULTS["yolo_classes"])
        self.var_yolo_conf = tk.DoubleVar(value=DEFAULTS["yolo_conf"])
        self.var_yolo_model = tk.StringVar(value=DEFAULTS["yolo_model"])
        
        # Overexposure mask
        self.var_mask_overexposure = tk.BooleanVar(value=DEFAULTS["mask_overexposure"])
        self.var_overexposure_threshold = tk.IntVar(value=DEFAULTS["overexposure_threshold"])
        self.var_overexposure_dilate = tk.IntVar(value=DEFAULTS["overexposure_dilate"])
        
        # Advanced options
        self.var_flip_vertical = tk.BooleanVar(value=DEFAULTS["flip_vertical"])
        self.var_rotate_z180 = tk.BooleanVar(value=DEFAULTS["rotate_z180"])
        self.var_apply_component = tk.BooleanVar(value=DEFAULTS["apply_component_transform"])
        self.var_quiet = tk.BooleanVar(value=DEFAULTS["quiet"])
        self.var_language = tk.StringVar(value=DEFAULTS["language"])
        self.var_advanced_expanded = tk.BooleanVar(value=False)

    def t(self, key, **kwargs):
        """Get localized UI text."""
        lang = self.var_language.get() if hasattr(self, "var_language") else "EN"
        table = UI_TEXT.get(lang, UI_TEXT["EN"])
        value = table.get(key, UI_TEXT["EN"].get(key, key))
        return value.format(**kwargs) if kwargs else value

    def on_language_changed(self, event=None):
        """Rebuild UI when language changes."""
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the main UI layout."""
        old_log = ""
        old_options_tab_idx = 0
        if self.console is not None:
            old_log = self.console.get("1.0", tk.END)
        if self.options_notebook is not None:
            try:
                old_options_tab_idx = self.options_notebook.index(self.options_notebook.select())
            except Exception:
                old_options_tab_idx = 0
        if self.main_canvas is not None:
            self.main_canvas.destroy()
        if self.main_scrollbar is not None:
            self.main_scrollbar.destroy()
        self.root.unbind_all("<MouseWheel>")
        self.root.title(self.t("app_title"))

        # Main container with scrollbar
        self.main_canvas = tk.Canvas(self.root)
        self.main_scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        scrollable_frame = ttk.Frame(self.main_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        
        self.main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.main_scrollbar.set)
        
        # Enable mouse wheel scrolling
        def on_mousewheel(event):
            self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.main_canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.main_scrollbar.pack(side="right", fill="y")
        
        # Padding
        padx = 10
        pady = 5

        language_frame = ttk.Frame(scrollable_frame)
        language_frame.pack(fill="x", padx=padx, pady=(10, 5))
        ttk.Label(language_frame, text=self.t("language")).pack(side="left", padx=5)
        lang_combo = ttk.Combobox(
            language_frame,
            textvariable=self.var_language,
            values=["EN", "JP"],
            state="readonly",
            width=6,
        )
        lang_combo.pack(side="left")
        lang_combo.bind("<<ComboboxSelected>>", self.on_language_changed)
        
        # ===== File Paths Section =====
        paths_frame = ttk.LabelFrame(scrollable_frame, text=self.t("paths_section"), padding=10)
        paths_frame.pack(fill="x", padx=padx, pady=pady)
        
        # Images folder
        self.add_path_entry(paths_frame, self.t("images_label"), self.var_images,
                           is_folder=True, row=0,
                           tooltip=self.t("tip_images"))
        
        # XML file
        self.add_path_entry(paths_frame, self.t("xml_label"), self.var_xml,
                           is_folder=False, row=1, filetypes=[("XML files", "*.xml")],
                           tooltip=self.t("tip_xml"))
        
        # PLY file
        self.add_path_entry(paths_frame, self.t("ply_label"), self.var_ply,
                           is_folder=False, row=2, filetypes=[("PLY files", "*.ply")],
                           tooltip=self.t("tip_ply"))
        
        # Output folder
        self.add_path_entry(paths_frame, self.t("output_label"), self.var_output,
                           is_folder=True, row=3,
                           tooltip=self.t("tip_output"))
        
        # ===== Tabbed Options Section =====
        self.options_notebook = ttk.Notebook(scrollable_frame)
        self.options_notebook.pack(fill="x", padx=padx, pady=pady)

        proc_tab = ttk.Frame(self.options_notebook, padding=10)
        skip_tab = ttk.Frame(self.options_notebook, padding=10)
        mask_tab = ttk.Frame(self.options_notebook, padding=10)
        self.options_notebook.add(proc_tab, text=self.t("proc_section"))
        self.options_notebook.add(skip_tab, text=self.t("skip_section"))
        self.options_notebook.add(mask_tab, text=self.t("mask_section"))

        # Processing tab
        ttk.Label(proc_tab, text=self.t("crop_size")).grid(row=0, column=0, sticky="w", padx=5)
        crop_spin = ttk.Spinbox(proc_tab, from_=256, to=4096, increment=64,
                                textvariable=self.var_crop_size, width=10)
        crop_spin.grid(row=0, column=1, sticky="w", padx=5, pady=3)
        ToolTip(crop_spin, self.t("tip_crop_size"))

        ttk.Label(proc_tab, text=self.t("fov")).grid(row=0, column=2, sticky="w", padx=5)
        fov_spin = ttk.Spinbox(proc_tab, from_=60, to=120, increment=5,
                               textvariable=self.var_fov_deg, width=10)
        fov_spin.grid(row=0, column=3, sticky="w", padx=5, pady=3)
        ToolTip(fov_spin, self.t("tip_fov"))

        ttk.Label(proc_tab, text=self.t("max_images")).grid(row=1, column=0, sticky="w", padx=5)
        max_spin = ttk.Spinbox(proc_tab, from_=1, to=100000, increment=100,
                               textvariable=self.var_max_images, width=10)
        max_spin.grid(row=1, column=1, sticky="w", padx=5, pady=3)
        ToolTip(max_spin, self.t("tip_max_images"))

        ttk.Label(proc_tab, text=self.t("workers")).grid(row=1, column=2, sticky="w", padx=5)
        worker_spin = ttk.Spinbox(proc_tab, from_=1, to=32, increment=1,
                                  textvariable=self.var_num_workers, width=10)
        worker_spin.grid(row=1, column=3, sticky="w", padx=5, pady=3)
        ToolTip(worker_spin, self.t("tip_workers"))

        range_check = ttk.Checkbutton(proc_tab, text=self.t("range"),
                                      variable=self.var_range_enabled,
                                      command=self.toggle_range)
        range_check.grid(row=2, column=0, sticky="w", padx=5, pady=3)

        self.range_frame = ttk.Frame(proc_tab)
        self.range_frame.grid(row=2, column=1, columnspan=3, sticky="w", padx=5)

        ttk.Label(self.range_frame, text=self.t("range_start")).pack(side="left")
        self.range_start_spin = ttk.Spinbox(self.range_frame, from_=0, to=100000,
                                            textvariable=self.var_range_start, width=8)
        self.range_start_spin.pack(side="left", padx=2)

        ttk.Label(self.range_frame, text=self.t("range_end")).pack(side="left", padx=(10, 0))
        self.range_end_spin = ttk.Spinbox(self.range_frame, from_=1, to=100000,
                                          textvariable=self.var_range_end, width=8)
        self.range_end_spin.pack(side="left", padx=2)
        ToolTip(self.range_frame, self.t("tip_range"))
        self.toggle_range()

        ttk.Label(proc_tab, text=self.t("yaw_offset")).grid(row=3, column=0, sticky="w", padx=5)
        yaw_spin = ttk.Spinbox(proc_tab, from_=-180, to=180, increment=5,
                               textvariable=self.var_yaw_offset, width=10)
        yaw_spin.grid(row=3, column=1, sticky="w", padx=5, pady=3)
        ToolTip(yaw_spin, self.t("tip_yaw_offset"))

        ttk.Label(proc_tab, text=self.t("output_format")).grid(row=4, column=0, sticky="w", padx=5)
        format_combo = ttk.Combobox(proc_tab, textvariable=self.var_output_format,
                                     values=["auto", "jpg", "png", "tiff", "webp"], width=15, state="readonly")
        format_combo.grid(row=4, column=1, sticky="w", padx=5, pady=3)
        ToolTip(format_combo, self.t("tip_output_format"))

        # Skip directions tab
        directions_inner = ttk.Frame(skip_tab)
        directions_inner.pack(fill="x")

        for i, direction in enumerate(VALID_DIRECTIONS):
            cb = ttk.Checkbutton(directions_inner, text=direction.capitalize(),
                                 variable=self.var_skip_directions[direction])
            cb.grid(row=0, column=i, padx=10, pady=3)

        ttk.Label(skip_tab, text=self.t("skip_note"),
                  foreground="gray").pack(anchor="w", pady=(5, 0))

        # Mask generation tab
        ttk.Checkbutton(mask_tab, text=self.t("enable_yolo"),
                        variable=self.var_generate_masks,
                        command=self.toggle_mask_options).grid(row=0, column=0, columnspan=2, sticky="w")

        self.yolo_frame = ttk.Frame(mask_tab)
        self.yolo_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=5)

        ttk.Label(self.yolo_frame, text=self.t("yolo_classes")).grid(row=0, column=0, sticky="w", padx=5)
        yolo_entry = ttk.Entry(self.yolo_frame, textvariable=self.var_yolo_classes, width=20)
        yolo_entry.grid(row=0, column=1, sticky="w", padx=5)
        ToolTip(yolo_entry, self.t("tip_yolo_classes"))

        ttk.Label(self.yolo_frame, text=self.t("yolo_conf")).grid(row=0, column=2, sticky="w", padx=5)
        conf_spin = ttk.Spinbox(self.yolo_frame, from_=0.0, to=1.0, increment=0.05,
                    textvariable=self.var_yolo_conf, width=8)
        conf_spin.grid(row=0, column=3, sticky="w", padx=5)
        ToolTip(conf_spin, self.t("tip_yolo_conf"))

        ttk.Label(self.yolo_frame, text=self.t("yolo_model")).grid(row=1, column=0, sticky="w", padx=5)
        model_combo = ttk.Combobox(
            self.yolo_frame,
            textvariable=self.var_yolo_model,
            values=YOLO_SEG_MODELS,
            state="readonly",
            width=24,
        )
        model_combo.grid(row=1, column=1, columnspan=3, sticky="w", padx=5)

        ttk.Checkbutton(self.yolo_frame, text=self.t("invert_mask"),
                variable=self.var_invert_mask).grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=3)

        ttk.Separator(mask_tab, orient="horizontal").grid(row=2, column=0, columnspan=4, sticky="ew", pady=10)

        ttk.Checkbutton(mask_tab, text=self.t("enable_overexp"),
                        variable=self.var_mask_overexposure,
                        command=self.toggle_overexposure_options).grid(row=3, column=0, columnspan=2, sticky="w")

        self.overexposure_frame = ttk.Frame(mask_tab)
        self.overexposure_frame.grid(row=4, column=0, columnspan=4, sticky="ew", pady=5)

        ttk.Label(self.overexposure_frame, text=self.t("threshold")).grid(row=0, column=0, sticky="w", padx=5)
        thresh_spin = ttk.Spinbox(self.overexposure_frame, from_=200, to=255, increment=1,
                                  textvariable=self.var_overexposure_threshold, width=8)
        thresh_spin.grid(row=0, column=1, sticky="w", padx=5)
        ToolTip(thresh_spin, self.t("tip_overexp_threshold"))

        ttk.Label(self.overexposure_frame, text=self.t("dilate")).grid(row=0, column=2, sticky="w", padx=5)
        dilate_spin = ttk.Spinbox(self.overexposure_frame, from_=0, to=50, increment=1,
                                  textvariable=self.var_overexposure_dilate, width=8)
        dilate_spin.grid(row=0, column=3, sticky="w", padx=5)
        ToolTip(dilate_spin, self.t("tip_overexp_dilate"))

        self.toggle_mask_options()
        self.toggle_overexposure_options()
        tab_count = self.options_notebook.index("end")
        if tab_count > 0:
            self.options_notebook.select(min(old_options_tab_idx, tab_count - 1))
        
        # ===== Advanced Options Section (Accordion) =====
        adv_container = ttk.LabelFrame(scrollable_frame, text=self.t("advanced_section"), padding=10)
        adv_container.pack(fill="x", padx=padx, pady=pady)

        self.adv_toggle_btn = ttk.Button(
            adv_container,
            text="",
            command=self.toggle_advanced_section,
        )
        self.adv_toggle_btn.pack(anchor="w", padx=5, pady=(0, 5))

        self.adv_content_frame = ttk.Frame(adv_container)

        flip_cb = ttk.Checkbutton(
            self.adv_content_frame,
            text=self.t("flip_vertical"),
            variable=self.var_flip_vertical,
        )
        flip_cb.grid(row=0, column=0, sticky="w", padx=5)
        ToolTip(flip_cb, self.t("tip_flip_vertical"))

        ttk.Checkbutton(
            self.adv_content_frame,
            text=self.t("rotate_z180"),
            variable=self.var_rotate_z180,
        ).grid(row=0, column=1, sticky="w", padx=5)

        ttk.Checkbutton(
            self.adv_content_frame,
            text=self.t("apply_component"),
            variable=self.var_apply_component,
        ).grid(row=1, column=0, sticky="w", padx=5)

        ttk.Checkbutton(
            self.adv_content_frame,
            text=self.t("quiet"),
            variable=self.var_quiet,
        ).grid(row=1, column=1, sticky="w", padx=5)

        self.update_advanced_section_visibility()
        
        # ===== Action Buttons =====
        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.pack(fill="x", padx=padx, pady=10)
        
        ttk.Button(btn_frame, text=self.t("save_cfg"), command=self.save_config).pack(side="left", padx=5)
        ttk.Button(btn_frame, text=self.t("load_cfg"), command=self.load_config).pack(side="left", padx=5)
        ttk.Button(btn_frame, text=self.t("reset_defaults"), command=self.reset_defaults).pack(side="left", padx=5)
        
        self.run_btn = ttk.Button(btn_frame, text=self.t("run"), command=self.run_conversion, style="Accent.TButton")
        self.run_btn.pack(side="right", padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text=self.t("stop"), command=self.stop_conversion, state="disabled")
        self.stop_btn.pack(side="right", padx=5)
        
        # ===== Output Console =====
        console_frame = ttk.LabelFrame(scrollable_frame, text=self.t("console_section"), padding=10)
        console_frame.pack(fill="both", expand=True, padx=padx, pady=pady)
        
        self.console = scrolledtext.ScrolledText(console_frame, height=12, wrap=tk.WORD,
                                                  font=("Consolas", 9))
        self.console.pack(fill="both", expand=True)
        
        # Clear console button
        ttk.Button(console_frame, text=self.t("clear_log"),
                  command=lambda: self.console.delete(1.0, tk.END)).pack(anchor="e", pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(scrollable_frame, mode="indeterminate")
        self.progress.pack(fill="x", padx=padx, pady=5)

        if old_log:
            self.console.insert(tk.END, old_log)
            self.console.see(tk.END)
    
    def add_path_entry(self, parent, label, variable, is_folder=True, row=0, 
                       filetypes=None, tooltip=None):
        """Add a path entry with browse button."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=3)
        
        entry = ttk.Entry(parent, textvariable=variable, width=50)
        entry.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
        parent.columnconfigure(1, weight=1)
        
        if tooltip:
            ToolTip(entry, tooltip)
        
        cmd = lambda: self.browse_folder(variable) if is_folder else self.browse_file(variable, filetypes)
        ttk.Button(parent, text=self.t("browse"), command=cmd, width=8).grid(row=row, column=2, padx=5, pady=3)
    
    def browse_folder(self, variable):
        """Open folder browser dialog."""
        path = filedialog.askdirectory(initialdir=variable.get() or ".")
        if path:
            variable.set(path)
    
    def browse_file(self, variable, filetypes=None):
        """Open file browser dialog."""
        filetypes = filetypes or [("All files", "*.*")]
        path = filedialog.askopenfilename(initialdir=os.path.dirname(variable.get()) or ".",
                                          filetypes=filetypes)
        if path:
            variable.set(path)
    
    def toggle_range(self):
        """Toggle range input fields state."""
        state = "normal" if self.var_range_enabled.get() else "disabled"
        self.range_start_spin.config(state=state)
        self.range_end_spin.config(state=state)
    
    def toggle_mask_options(self):
        """Toggle YOLO mask options state."""
        state = "normal" if self.var_generate_masks.get() else "disabled"
        for child in self.yolo_frame.winfo_children():
            try:
                child.config(state=state)
            except:
                pass
    
    def toggle_overexposure_options(self):
        """Toggle overexposure mask options state."""
        state = "normal" if self.var_mask_overexposure.get() else "disabled"
        for child in self.overexposure_frame.winfo_children():
            try:
                child.config(state=state)
            except:
                pass

    def update_advanced_section_visibility(self):
        """Update accordion state for advanced options section."""
        expanded = self.var_advanced_expanded.get()
        if expanded:
            self.adv_content_frame.pack(fill="x", padx=0, pady=0)
            self.adv_toggle_btn.config(text=f"▼ {self.t('advanced_hide')}")
        else:
            self.adv_content_frame.pack_forget()
            self.adv_toggle_btn.config(text=f"▶ {self.t('advanced_show')}")

    def toggle_advanced_section(self):
        """Toggle expanded/collapsed state for advanced options."""
        self.var_advanced_expanded.set(not self.var_advanced_expanded.get())
        self.update_advanced_section_visibility()
    
    def reset_defaults(self):
        """Reset all settings to defaults."""
        self.var_images.set(DEFAULTS["images"])
        self.var_xml.set(DEFAULTS["xml"])
        self.var_ply.set(DEFAULTS["ply"])
        self.var_output.set(DEFAULTS["output"])
        self.var_crop_size.set(DEFAULTS["crop_size"])
        self.var_fov_deg.set(DEFAULTS["fov_deg"])
        self.var_max_images.set(DEFAULTS["max_images"])
        self.var_range_enabled.set(False)
        self.var_range_start.set(DEFAULTS["range_start"])
        self.var_range_end.set(DEFAULTS["range_end"])
        self.var_num_workers.set(DEFAULTS["num_workers"])
        self.var_yaw_offset.set(DEFAULTS["yaw_offset"])
        self.var_output_format.set(DEFAULTS["output_format"])
        self.var_generate_masks.set(DEFAULTS["generate_masks"])
        self.var_invert_mask.set(DEFAULTS["invert_mask"])
        self.var_yolo_classes.set(DEFAULTS["yolo_classes"])
        self.var_yolo_conf.set(DEFAULTS["yolo_conf"])
        self.var_yolo_model.set(DEFAULTS["yolo_model"])
        self.var_mask_overexposure.set(DEFAULTS["mask_overexposure"])
        self.var_overexposure_threshold.set(DEFAULTS["overexposure_threshold"])
        self.var_overexposure_dilate.set(DEFAULTS["overexposure_dilate"])
        self.var_flip_vertical.set(DEFAULTS["flip_vertical"])
        self.var_rotate_z180.set(DEFAULTS["rotate_z180"])
        self.var_apply_component.set(DEFAULTS["apply_component_transform"])
        self.var_quiet.set(DEFAULTS["quiet"])
        
        for d in VALID_DIRECTIONS:
            self.var_skip_directions[d].set(False)
        
        self.toggle_range()
        self.toggle_mask_options()
        self.toggle_overexposure_options()
        
        self.log(f"{self.t('reset_done')}\n")
    
    def build_command(self):
        """Build the command line arguments."""
        # In development we launch the Python script directly.
        # In frozen mode we launch a companion CLI executable from the same folder.
        if getattr(sys, "frozen", False):
            cli_exe = self.get_app_base_dir() / "metashape_360_to_colmap.exe"
            if not cli_exe.exists():
                raise FileNotFoundError(
                    f"Required converter executable was not found: {cli_exe}"
                )
            cmd = [str(cli_exe)]
        else:
            script_path = self.get_app_base_dir() / "metashape_360_to_colmap.py"
            # Use unbuffered mode so stdout/stderr is streamed to the GUI log in real time.
            cmd = [sys.executable, "-u", str(script_path)]
        
        # Required paths
        cmd.extend(["--images", self.var_images.get()])
        cmd.extend(["--xml", self.var_xml.get()])
        cmd.extend(["--output", self.var_output.get()])
        
        if self.var_ply.get().strip():
            cmd.extend(["--ply", self.var_ply.get()])
        
        # Processing options
        cmd.extend(["--crop-size", str(self.var_crop_size.get())])
        cmd.extend(["--fov-deg", str(self.var_fov_deg.get())])
        cmd.extend(["--max-images", str(self.var_max_images.get())])
        cmd.extend(["--num-workers", str(self.var_num_workers.get())])
        cmd.extend(["--yaw-offset", str(self.var_yaw_offset.get())])
        cmd.extend(["--output-format", self.var_output_format.get()])
        
        # Range
        if self.var_range_enabled.get():
            cmd.extend(["--range-images", f"{self.var_range_start.get()}-{self.var_range_end.get()}"])
        
        # Skip directions
        skip_dirs = [d for d in VALID_DIRECTIONS if self.var_skip_directions[d].get()]
        if skip_dirs:
            cmd.extend(["--skip-directions", ",".join(skip_dirs)])
        
        # Mask options
        if self.var_generate_masks.get():
            cmd.append("--generate-masks")
            cmd.extend(["--yolo-classes", self.var_yolo_classes.get()])
            cmd.extend(["--yolo-conf", str(self.var_yolo_conf.get())])
            cmd.extend(["--yolo-model", self.var_yolo_model.get()])
            if self.var_invert_mask.get():
                cmd.append("--invert-mask")
        
        # Overexposure mask
        if self.var_mask_overexposure.get():
            cmd.append("--mask-overexposure")
            cmd.extend(["--overexposure-threshold", str(self.var_overexposure_threshold.get())])
            cmd.extend(["--overexposure-dilate", str(self.var_overexposure_dilate.get())])
        
        # Advanced options
        if self.var_flip_vertical.get():
            cmd.append("--flip-vertical")
        else:
            cmd.append("--no-flip-vertical")
        
        if self.var_rotate_z180.get():
            cmd.append("--rotate-z180")
        else:
            cmd.append("--no-rotate-z180")
        
        if self.var_apply_component.get():
            cmd.append("--apply-component-transform-for-ply")
        
        if self.var_quiet.get():
            cmd.append("--quiet")
        
        return cmd

    def get_app_base_dir(self):
        """Return directory that contains app runtime files.

        - dev: folder containing this .py file
        - frozen: folder containing the .exe
        """
        if getattr(sys, "frozen", False):
            return Path(sys.executable).parent
        return Path(__file__).parent
    
    def validate_inputs(self):
        """Validate required inputs before running."""
        errors = []
        
        if not self.var_images.get().strip():
            errors.append(self.t("err_images_required"))
        elif not Path(self.var_images.get()).exists():
            errors.append(self.t("err_images_missing", path=self.var_images.get()))
        
        if not self.var_xml.get().strip():
            errors.append(self.t("err_xml_required"))
        elif not Path(self.var_xml.get()).exists():
            errors.append(self.t("err_xml_missing", path=self.var_xml.get()))
        
        if not self.var_output.get().strip():
            errors.append(self.t("err_output_required"))
        
        if self.var_ply.get().strip() and not Path(self.var_ply.get()).exists():
            errors.append(self.t("err_ply_missing", path=self.var_ply.get()))
        
        if errors:
            messagebox.showerror(self.t("err_title"), "\n".join(errors))
            return False
        return True
    
    def run_conversion(self):
        """Run the conversion process."""
        if not self.validate_inputs():
            return

        try:
            cmd = self.build_command()
        except Exception as e:
            messagebox.showerror(self.t("err_title"), str(e))
            self.log(f"{self.t('error', error=e)}\n")
            return

        self.log(f"{self.t('cmd', cmd=' '.join(cmd))}\n\n")
        
        # Disable run button, enable stop
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress.start()
        
        # Run in background thread
        def run_process():
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    cwd=str(self.get_app_base_dir()),
                    env=self.get_subprocess_env(),
                    creationflags=self.get_subprocess_creationflags(),
                )

                # Read one character at a time so we can handle both '\n' and
                # '\r' terminated progress output from the CLI in real time.
                line_buf = ""
                while True:
                    ch = self.process.stdout.read(1)
                    if ch == "":
                        if line_buf:
                            self.output_queue.put(line_buf + "\n")
                            line_buf = ""
                        if self.process.poll() is not None:
                            break
                        continue

                    if ch in ("\n", "\r"):
                        if line_buf:
                            self.output_queue.put(line_buf + "\n")
                            line_buf = ""
                        continue

                    line_buf += ch
                
                self.process.wait()
                rc = self.process.returncode
                self.output_queue.put(f"\n{'='*50}\n")
                if rc == 0:
                    self.output_queue.put(f"{self.t('done_ok')}\n")
                else:
                    self.output_queue.put(f"{self.t('done_ng', code=rc)}\n")
            except Exception as e:
                self.output_queue.put(f"{self.t('error', error=e)}\n")
            finally:
                self.output_queue.put("__DONE__")
        
        threading.Thread(target=run_process, daemon=True).start()

    def get_subprocess_env(self):
        """Return environment for subprocess execution.

        Force unbuffered stdio so CLI logs are streamed into the GUI promptly,
        especially when the GUI itself is running as a windowed executable.
        """
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return env

    def get_subprocess_creationflags(self):
        """Return subprocess creation flags.

        On Windows, suppress creating an extra console window when the GUI
        launches the CLI executable.
        """
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            return subprocess.CREATE_NO_WINDOW
        return 0
    
    def stop_conversion(self):
        """Stop the running conversion process."""
        if self.process:
            self.stop_process(force=False)

            # Restore UI state immediately so the user can run again
            # even if the worker thread takes time to observe process exit.
            self.run_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.progress.stop()

            self.log(f"\n{self.t('stopped')}\n")
            self.process = None

    def stop_process(self, force=False):
        """Stop the current CLI process.

        If force=True on Windows, kill the full process tree to ensure the
        bundled CLI process is terminated immediately.
        """
        if not self.process:
            return

        proc = self.process
        try:
            if force and os.name == "nt":
                # taskkill /T /F kills the target process and its child tree.
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    creationflags=self.get_subprocess_creationflags(),
                )
            else:
                proc.terminate()
        except Exception:
            pass

        try:
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def on_close_requested(self):
        """Handle window close with confirmation while conversion is running."""
        if self.process and self.process.poll() is None:
            should_exit = messagebox.askyesno(
                self.t("exit_confirm_title"),
                self.t("exit_confirm_running"),
            )
            if not should_exit:
                return
            self.stop_process(force=True)

        self.is_closing = True
        self.root.destroy()
    
    def poll_output(self):
        """Poll the output queue for process output."""
        if self.is_closing:
            return

        try:
            while True:
                msg = self.output_queue.get_nowait()
                if msg == "__DONE__":
                    self.run_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    self.progress.stop()
                    self.process = None
                else:
                    self.log(msg)
        except queue.Empty:
            pass
        try:
            self.root.after(100, self.poll_output)
        except tk.TclError:
            # Window is already closing/closed.
            return
    
    def log(self, message):
        """Log message to console."""
        self.console.insert(tk.END, message)
        self.console.see(tk.END)
    
    def save_config(self):
        """Save current settings to config.txt."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Config files", "*.txt"), ("All files", "*.*")],
            initialfile="config.txt"
        )
        if not filepath:
            return
        
        lines = [
            "# Configuration file for metashape_360_to_colmap.py",
            "# Generated by GUI application",
            "",
            f"images={self.var_images.get()}",
            f"xml={self.var_xml.get()}",
            f"ply={self.var_ply.get()}",
            f"output={self.var_output.get()}",
            "",
            f"crop-size={self.var_crop_size.get()}",
            f"fov-deg={self.var_fov_deg.get()}",
            f"max-images={self.var_max_images.get()}",
        ]
        
        if self.var_range_enabled.get():
            lines.append(f"range-images={self.var_range_start.get()}-{self.var_range_end.get()}")
        
        lines.extend([
            f"num-workers={self.var_num_workers.get()}",
            f"yaw-offset={self.var_yaw_offset.get()}",
            f"output-format={self.var_output_format.get()}",
            "",
        ])
        
        skip_dirs = [d for d in VALID_DIRECTIONS if self.var_skip_directions[d].get()]
        lines.append(f"skip-directions={','.join(skip_dirs)}")
        
        lines.extend([
            "",
            f"generate-masks={self.var_generate_masks.get()}",
            f"invert-mask={self.var_invert_mask.get()}",
            f"yolo-classes={self.var_yolo_classes.get()}",
            f"yolo-conf={self.var_yolo_conf.get()}",
            f"yolo-model={self.var_yolo_model.get()}",
            "",
            f"mask-overexposure={self.var_mask_overexposure.get()}",
            f"overexposure-threshold={self.var_overexposure_threshold.get()}",
            f"overexposure-dilate={self.var_overexposure_dilate.get()}",
            "",
            f"flip-vertical={self.var_flip_vertical.get()}",
            f"rotate-z180={self.var_rotate_z180.get()}",
            f"apply-component-transform-for-ply={self.var_apply_component.get()}",
            f"quiet={self.var_quiet.get()}",
            f"language={self.var_language.get()}",
        ])
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        self.log(f"{self.t('saved', path=filepath)}\n")
    
    def load_config(self):
        """Load settings from config.txt."""
        filepath = filedialog.askopenfilename(
            filetypes=[("Config files", "*.txt"), ("All files", "*.*")]
        )
        if not filepath:
            return
        self.load_config_from_path(filepath)
    
    def load_config_file(self):
        """Load config.txt if it exists in the script directory."""
        config_path = self.get_app_base_dir() / "config.txt"
        if config_path.exists():
            self.load_config_from_path(str(config_path))
            self.log(f"{self.t('auto_loaded')}\n")
    
    def load_config_from_path(self, filepath):
        """Load configuration from specified path."""
        config = {}
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        config[key] = value
        except Exception as e:
            self.log(f"{self.t('load_error', error=e)}\n")
            return
        
        # Apply loaded config
        if "images" in config:
            self.var_images.set(config["images"])
        if "xml" in config:
            self.var_xml.set(config["xml"])
        if "ply" in config:
            self.var_ply.set(config["ply"])
        if "output" in config:
            self.var_output.set(config["output"])
        
        if "crop-size" in config:
            self.var_crop_size.set(int(config["crop-size"]))
        if "fov-deg" in config:
            self.var_fov_deg.set(float(config["fov-deg"]))
        if "max-images" in config:
            self.var_max_images.set(int(config["max-images"]))
        if "num-workers" in config:
            self.var_num_workers.set(int(config["num-workers"]))
        if "yaw-offset" in config:
            self.var_yaw_offset.set(float(config["yaw-offset"]))
        if "output-format" in config:
            self.var_output_format.set(config["output-format"])
        
        if "range-images" in config and config["range-images"]:
            parts = config["range-images"].split("-")
            if len(parts) == 2:
                self.var_range_enabled.set(True)
                self.var_range_start.set(int(parts[0]))
                self.var_range_end.set(int(parts[1]))
                self.toggle_range()
        
        if "skip-directions" in config and config["skip-directions"]:
            skip_list = [d.strip().lower() for d in config["skip-directions"].split(",") if d.strip()]
            for d in VALID_DIRECTIONS:
                self.var_skip_directions[d].set(d in skip_list)
        
        # Boolean values
        def parse_bool(val):
            return val.lower() in ("true", "1", "yes")
        
        if "generate-masks" in config:
            self.var_generate_masks.set(parse_bool(config["generate-masks"]))
        if "invert-mask" in config:
            self.var_invert_mask.set(parse_bool(config["invert-mask"]))
        if "yolo-classes" in config:
            self.var_yolo_classes.set(config["yolo-classes"])
        if "yolo-conf" in config:
            self.var_yolo_conf.set(float(config["yolo-conf"]))
        if "yolo-model" in config:
            self.var_yolo_model.set(config["yolo-model"])
        
        if "mask-overexposure" in config:
            self.var_mask_overexposure.set(parse_bool(config["mask-overexposure"]))
        if "overexposure-threshold" in config:
            self.var_overexposure_threshold.set(int(config["overexposure-threshold"]))
        if "overexposure-dilate" in config:
            self.var_overexposure_dilate.set(int(config["overexposure-dilate"]))
        
        if "flip-vertical" in config:
            self.var_flip_vertical.set(parse_bool(config["flip-vertical"]))
        if "rotate-z180" in config:
            self.var_rotate_z180.set(parse_bool(config["rotate-z180"]))
        if "apply-component-transform-for-ply" in config:
            self.var_apply_component.set(parse_bool(config["apply-component-transform-for-ply"]))
        if "quiet" in config:
            self.var_quiet.set(parse_bool(config["quiet"]))
        if "language" in config and config["language"] in UI_TEXT:
            self.var_language.set(config["language"])
        
        self.toggle_mask_options()
        self.toggle_overexposure_options()
        
        self.log(f"{self.t('loaded', path=filepath)}\n")

        if "language" in config and config["language"] in UI_TEXT:
            self.setup_ui()


def main():
    multiprocessing.freeze_support()

    root = tk.Tk()
    
    # Set DPI awareness on Windows
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    
    # Configure ttk style
    style = ttk.Style()
    if "vista" in style.theme_names():
        style.theme_use("vista")
    elif "clam" in style.theme_names():
        style.theme_use("clam")
    
    app = Metashape360GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
