"""
GUI Interface for Image Analysis System using Tkinter
Provides an easy-to-use interface for single image and batch processing
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import threading
import time
import os
from pathlib import Path
from PIL import Image, ImageTk
import cv2

from image_analyzer import ImageAnalyzer
from batch_processor import BatchProcessor
from video_analyzer import VideoAnalyzer

class ImageAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vision AI - Computer Vision Analysis")
        self.root.geometry("1400x900")
        self.root.configure(bg='#fafafa')
        
        # Make window resizable and handle fullscreen properly
        self.root.minsize(1200, 800)
        self.root.state('zoomed')  # Start maximized on Windows
        
        # Modern minimal color palette
        self.colors = {
            'bg_primary': '#fafafa',      # Light background
            'bg_secondary': '#ffffff',    # Pure white
            'bg_tertiary': '#f5f5f5',     # Light gray
            'accent_primary': '#0ea5e9',  # Modern blue (sky-500)
            'accent_secondary': '#ff6b35', # Orange accent
            'accent_success': '#10b981',  # Emerald green
            'accent_warning': '#f59e0b',  # Amber
            'accent_danger': '#ef4444',   # Red
            'accent_hover': '#0284c7',    # Darker blue for hover
            'accent_light': '#e0f2fe',    # Light blue background
            'text_primary': '#0f172a',    # Dark slate
            'text_secondary': '#64748b',  # Slate gray
            'text_muted': '#94a3b8',      # Light slate
            'border': '#e2e8f0',          # Light border
            'shadow': 'rgba(0, 0, 0, 0.1)', # Soft shadow
            'success': '#10b981',         # Emerald
            'warning': '#f59e0b',         # Amber
            'danger': '#ef4444'           # Red
        }
        
        # Initialize analyzer
        self.analyzer = None
        self.batch_processor = None
        self.video_analyzer = None
        self.current_image_path = None
        self.current_results = None
        self.video_thread = None
        self.is_video_running = False
        
        self.setup_ui()
        self.initialize_analyzer()
    
    def setup_ui(self):
        """Create the main UI components with modern minimal design"""
        # Configure ttk styles for modern theme
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container with sidebar layout
        main_container = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_container.pack(fill='both', expand=True)
        
        # Left sidebar navigation
        self.setup_sidebar(main_container)
        
        # Main content area
        self.content_frame = tk.Frame(main_container, bg=self.colors['bg_primary'])
        self.content_frame.pack(side='right', fill='both', expand=True)
        
        # Initialize with single image analysis view
        self.current_view = "single"
        self.setup_single_image_view()
    
    def setup_sidebar(self, parent):
        """Create modern left sidebar navigation"""
        sidebar = tk.Frame(parent, bg=self.colors['bg_secondary'], width=280)
        sidebar.pack(side='left', fill='y', padx=(0, 1), pady=0)
        sidebar.pack_propagate(False)
        
        # Add subtle shadow effect with border
        shadow_frame = tk.Frame(parent, bg=self.colors['border'], width=1)
        shadow_frame.pack(side='left', fill='y')
        
        # App header in sidebar
        header_frame = tk.Frame(sidebar, bg=self.colors['bg_secondary'], height=80)
        header_frame.pack(fill='x', padx=24, pady=(24, 32))
        header_frame.pack_propagate(False)
        
        # App title
        title_label = tk.Label(
            header_frame,
            text="Vision AI",
            font=('Segoe UI', 20, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_secondary']
        )
        title_label.pack(anchor='w')
        
        # Subtitle
        subtitle_label = tk.Label(
            header_frame,
            text="Computer Vision Analysis",
            font=('Segoe UI', 11),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg_secondary']
        )
        subtitle_label.pack(anchor='w', pady=(4, 0))
        
        # Navigation buttons
        nav_frame = tk.Frame(sidebar, bg=self.colors['bg_secondary'])
        nav_frame.pack(fill='x', padx=16, pady=0)
        
        # Navigation items
        nav_items = [
            ("📷", "Single Image", "single"),
            ("📁", "Batch Processing", "batch"),
            ("🎥", "Live & Video", "video"),
            ("⚙️", "Settings", "settings")
        ]
        
        self.nav_buttons = {}
        for icon, text, view_id in nav_items:
            btn_frame = tk.Frame(nav_frame, bg=self.colors['bg_secondary'])
            btn_frame.pack(fill='x', pady=2)
            
            btn = tk.Button(
                btn_frame,
                text=f"  {icon}  {text}",
                font=('Segoe UI', 11, 'normal'),
                fg=self.colors['text_secondary'],
                bg=self.colors['bg_secondary'],
                activebackground=self.colors['accent_light'],
                activeforeground=self.colors['accent_primary'],
                relief='flat',
                anchor='w',
                padx=16,
                pady=12,
                cursor='hand2',
                command=lambda v=view_id: self.switch_view(v)
            )
            btn.pack(fill='x')
            
            # Hover effects
            def on_enter(e, button=btn, view=view_id):
                if self.current_view != view:
                    button.config(bg=self.colors['bg_tertiary'])
            
            def on_leave(e, button=btn, view=view_id):
                if self.current_view != view:
                    button.config(bg=self.colors['bg_secondary'])
            
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)
            
            self.nav_buttons[view_id] = btn
        
        # Set initial active state
        self.nav_buttons['single'].config(
            bg=self.colors['accent_light'],
            fg=self.colors['accent_primary']
        )
    
    def switch_view(self, view_id):
        """Switch between different views"""
        # Reset all nav buttons
        for btn_id, btn in self.nav_buttons.items():
            if btn_id == view_id:
                btn.config(bg=self.colors['accent_light'], fg=self.colors['accent_primary'])
            else:
                btn.config(bg=self.colors['bg_secondary'], fg=self.colors['text_secondary'])
        
        # Clear content frame
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Load appropriate view
        self.current_view = view_id
        if view_id == "single":
            self.setup_single_image_view()
        elif view_id == "batch":
            self.setup_batch_processing_view()
        elif view_id == "video":
            self.setup_video_analysis_view()
        elif view_id == "settings":
            self.setup_settings_view()
    
    def setup_single_image_view(self):
        """Setup the single image analysis view"""
        # Main container
        container = tk.Frame(self.content_frame, bg=self.colors['bg_primary'])
        container.pack(fill='both', expand=True, padx=32, pady=24)
        
        # Header
        header = tk.Frame(container, bg=self.colors['bg_primary'])
        header.pack(fill='x', pady=(0, 24))
        
        title = tk.Label(
            header,
            text="Single Image Analysis",
            font=('Segoe UI', 24, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_primary']
        )
        title.pack(anchor='w')
        
        subtitle = tk.Label(
            header,
            text="Analyze individual images for people, vehicles, and traffic lights",
            font=('Segoe UI', 12),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg_primary']
        )
        subtitle.pack(anchor='w', pady=(4, 0))
        
        # Main content area
        content = tk.Frame(container, bg=self.colors['bg_primary'])
        content.pack(fill='both', expand=True)
        
        # Left panel for image
        left_panel = tk.Frame(content, bg=self.colors['bg_secondary'], relief='solid', bd=1)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 16))
        
        # Image display area
        self.image_label = tk.Label(
            left_panel,
            text="📷 Drop an image here or use the controls\n\nSupported formats: JPG, PNG, BMP, TIFF, WebP",
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_muted'],
            font=('Segoe UI', 14),
            justify='center'
        )
        self.image_label.pack(expand=True, padx=32, pady=32)
        
        # Right panel for controls
        right_panel = tk.Frame(content, bg=self.colors['bg_primary'], width=400)
        right_panel.pack(side='right', fill='y')
        right_panel.pack_propagate(False)
        
        # File selection card
        file_card = tk.Frame(right_panel, bg=self.colors['bg_secondary'], relief='solid', bd=1)
        file_card.pack(fill='x', pady=(0, 16))
        
        file_header = tk.Label(
            file_card,
            text="📁 Select Image",
            font=('Segoe UI', 14, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_secondary']
        )
        file_header.pack(anchor='w', padx=20, pady=(20, 8))
        
        # Buttons
        btn_frame = tk.Frame(file_card, bg=self.colors['bg_secondary'])
        btn_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        self.select_btn = tk.Button(
            btn_frame,
            text="Browse Files",
            command=self.select_image,
            bg=self.colors['accent_primary'],
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief='flat',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.select_btn.pack(fill='x', pady=(0, 8))
        
        self.add_photo_btn = tk.Button(
            btn_frame,
            text="Add Photo",
            command=self.add_photo_from_files,
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_primary'],
            font=('Segoe UI', 10),
            relief='flat',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.add_photo_btn.pack(fill='x')
        
        # File path display
        self.file_path_label = tk.Label(
            file_card,
            text="No file selected",
            font=('Segoe UI', 9),
            fg=self.colors['text_muted'],
            bg=self.colors['bg_secondary'],
            wraplength=350,
            justify='left'
        )
        self.file_path_label.pack(anchor='w', padx=20, pady=(0, 20))
        
        # Analysis card
        analysis_card = tk.Frame(right_panel, bg=self.colors['bg_secondary'], relief='solid', bd=1)
        analysis_card.pack(fill='x', pady=(0, 16))
        
        analysis_header = tk.Label(
            analysis_card,
            text="🔍 Analysis",
            font=('Segoe UI', 14, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_secondary']
        )
        analysis_header.pack(anchor='w', padx=20, pady=(20, 8))
        
        self.analyze_btn = tk.Button(
            analysis_card,
            text="Analyze Image",
            command=self.analyze_single_image,
            bg=self.colors['success'],
            fg='white',
            font=('Segoe UI', 12, 'bold'),
            relief='flat',
            padx=20,
            pady=12,
            cursor='hand2',
            state='disabled'
        )
        self.analyze_btn.pack(fill='x', padx=20, pady=(0, 20))
        
        # Progress bar
        self.progress_single = ttk.Progressbar(analysis_card, mode='indeterminate')
        self.progress_single.pack(fill='x', padx=20, pady=(0, 20))
        
        # Results card
        results_card = tk.Frame(right_panel, bg=self.colors['bg_secondary'], relief='solid', bd=1)
        results_card.pack(fill='both', expand=True)
        
        results_header = tk.Label(
            results_card,
            text="📊 Results",
            font=('Segoe UI', 14, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_secondary']
        )
        results_header.pack(anchor='w', padx=20, pady=(20, 8))
        
        self.results_text = scrolledtext.ScrolledText(
            results_card,
            font=('Consolas', 9),
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_primary'],
            relief='flat',
            wrap='word'
        )
        self.results_text.pack(fill='both', expand=True, padx=20, pady=(0, 16))
        
        # Save button
        self.save_single_btn = tk.Button(
            results_card,
            text="💾 Save Results",
            command=self.save_single_results,
            bg=self.colors['warning'],
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief='flat',
            padx=20,
            pady=10,
            cursor='hand2',
            state='disabled'
        )
        self.save_single_btn.pack(fill='x', padx=20, pady=(0, 20))
    
    def setup_batch_processing_view(self):
        """Setup the batch processing view"""
        container = tk.Frame(self.content_frame, bg=self.colors['bg_primary'])
        container.pack(fill='both', expand=True, padx=32, pady=24)
        
        # Header
        header = tk.Frame(container, bg=self.colors['bg_primary'])
        header.pack(fill='x', pady=(0, 24))
        
        title = tk.Label(
            header,
            text="Batch Processing",
            font=('Segoe UI', 24, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_primary']
        )
        title.pack(anchor='w')
        
        subtitle = tk.Label(
            header,
            text="Process multiple images from a directory automatically",
            font=('Segoe UI', 12),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg_primary']
        )
        subtitle.pack(anchor='w', pady=(4, 0))
        
        # Main content
        content = tk.Frame(container, bg=self.colors['bg_primary'])
        content.pack(fill='both', expand=True)
        
        # Left panel for controls
        left_panel = tk.Frame(content, bg=self.colors['bg_primary'], width=400)
        left_panel.pack(side='left', fill='y', padx=(0, 16))
        left_panel.pack_propagate(False)
        
        # Directory selection card
        dir_card = tk.Frame(left_panel, bg=self.colors['bg_secondary'], relief='solid', bd=1)
        dir_card.pack(fill='x', pady=(0, 16))
        
        dir_header = tk.Label(
            dir_card,
            text="📂 Select Directory",
            font=('Segoe UI', 14, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_secondary']
        )
        dir_header.pack(anchor='w', padx=20, pady=(20, 8))
        
        self.select_dir_btn = tk.Button(
            dir_card,
            text="Browse Directory",
            command=self.select_directory,
            bg=self.colors['accent_primary'],
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief='flat',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.select_dir_btn.pack(fill='x', padx=20, pady=(0, 8))
        
        self.dir_path_label = tk.Label(
            dir_card,
            text="No directory selected",
            font=('Segoe UI', 9),
            fg=self.colors['text_muted'],
            bg=self.colors['bg_secondary'],
            wraplength=350,
            justify='left'
        )
        self.dir_path_label.pack(anchor='w', padx=20, pady=(0, 20))
        
        # Processing controls card
        process_card = tk.Frame(left_panel, bg=self.colors['bg_secondary'], relief='solid', bd=1)
        process_card.pack(fill='x', pady=(0, 16))
        
        process_header = tk.Label(
            process_card,
            text="⚡ Processing Options",
            font=('Segoe UI', 14, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_secondary']
        )
        process_header.pack(anchor='w', padx=20, pady=(20, 8))
        
        # Parallel processing checkbox
        self.parallel_var = tk.BooleanVar(value=True)
        parallel_check = tk.Checkbutton(
            process_card,
            text="Enable parallel processing (faster)",
            variable=self.parallel_var,
            font=('Segoe UI', 10),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            activebackground=self.colors['bg_secondary']
        )
        parallel_check.pack(anchor='w', padx=20, pady=(0, 8))
        
        self.process_btn = tk.Button(
            process_card,
            text="Start Batch Processing",
            command=self.start_batch_processing,
            bg=self.colors['success'],
            fg='white',
            font=('Segoe UI', 12, 'bold'),
            relief='flat',
            padx=20,
            pady=12,
            cursor='hand2',
            state='disabled'
        )
        self.process_btn.pack(fill='x', padx=20, pady=(0, 16))
        
        # Progress
        self.progress_batch = ttk.Progressbar(process_card, mode='determinate')
        self.progress_batch.pack(fill='x', padx=20, pady=(0, 8))
        
        self.progress_label = tk.Label(
            process_card,
            text="Ready to process",
            font=('Segoe UI', 9),
            fg=self.colors['text_muted'],
            bg=self.colors['bg_secondary']
        )
        self.progress_label.pack(padx=20, pady=(0, 20))
        
        # Right panel for results
        right_panel = tk.Frame(content, bg=self.colors['bg_secondary'], relief='solid', bd=1)
        right_panel.pack(side='right', fill='both', expand=True)
        
        results_header = tk.Label(
            right_panel,
            text="📊 Batch Results",
            font=('Segoe UI', 14, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_secondary']
        )
        results_header.pack(anchor='w', padx=20, pady=(20, 8))
        
        self.batch_results_text = scrolledtext.ScrolledText(
            right_panel,
            font=('Consolas', 9),
            bg=self.colors['bg_tertiary'],
            fg=self.colors['text_primary'],
            relief='flat',
            wrap='word'
        )
        self.batch_results_text.pack(fill='both', expand=True, padx=20, pady=(0, 16))
        
        # Save results button
        self.save_batch_btn = tk.Button(
            right_panel,
            text="💾 Save Batch Results",
            command=self.save_batch_results,
            bg=self.colors['warning'],
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief='flat',
            padx=20,
            pady=10,
            cursor='hand2',
            state='disabled'
        )
        self.save_batch_btn.pack(fill='x', padx=20, pady=(0, 20))
    
    def setup_video_analysis_view(self):
        """Setup the video analysis view"""
        # Placeholder for video analysis view
        container = tk.Frame(self.content_frame, bg=self.colors['bg_primary'])
        container.pack(fill='both', expand=True, padx=32, pady=24)
        
        title = tk.Label(
            container,
            text="Live & Video Analysis",
            font=('Segoe UI', 24, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_primary']
        )
        title.pack(anchor='w')
        
        subtitle = tk.Label(
            container,
            text="Analyze live video feeds and video files",
            font=('Segoe UI', 12),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg_primary']
        )
        subtitle.pack(anchor='w', pady=(4, 24))
        
        # Main content area
        content = tk.Frame(container, bg=self.colors['bg_primary'])
        content.pack(fill='both', expand=True)
        
        # Left panel for video display
        left_panel = tk.Frame(content, bg=self.colors['bg_secondary'], relief='solid', bd=1)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 16))
        
        # Video display area
        self.video_label = tk.Label(
            left_panel,
            text="🎥 LIVE VIDEO FEED\n\nSelect a video source to start analysis",
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_secondary'],
            font=('Segoe UI', 14, 'bold'),
            justify='center'
        )
        self.video_label.pack(expand=True)
        
        # Right panel for controls
        right_panel = tk.Frame(content, bg=self.colors['bg_primary'], width=400)
        right_panel.pack(side='right', fill='y')
        right_panel.pack_propagate(False)
        
        # Video source selection
        source_frame = tk.LabelFrame(
            right_panel,
            text="🎬 VIDEO SOURCE",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['accent_secondary'],
            bd=2,
            relief='solid'
        )
        source_frame.pack(fill='x', pady=8)
        
        # Webcam button
        self.webcam_btn = tk.Button(
            source_frame,
            text="📹 START WEBCAM",
            command=self.start_webcam_analysis,
            bg=self.colors['accent_success'],
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            relief='flat',
            padx=20,
            pady=8,
            cursor='hand2'
        )
        self.webcam_btn.pack(pady=8)
        
        # Video file button
        self.video_file_btn = tk.Button(
            source_frame,
            text="📁 SELECT VIDEO FILE",
            command=self.select_video_file,
            bg=self.colors['accent_primary'],
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            relief='flat',
            padx=20,
            pady=8,
            cursor='hand2'
        )
        self.video_file_btn.pack(pady=8)
        
        # RTSP stream input
        rtsp_frame = tk.Frame(source_frame, bg=self.colors['bg_primary'])
        rtsp_frame.pack(fill='x', pady=8)
        
        tk.Label(rtsp_frame, text="RTSP URL:", bg=self.colors['bg_primary'], fg=self.colors['text_secondary'], font=('Segoe UI', 9)).pack(anchor='w')
        
        self.rtsp_entry = tk.Entry(
            rtsp_frame,
            font=('Segoe UI', 9),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            insertbackground=self.colors['accent_primary']
        )
        self.rtsp_entry.pack(fill='x', pady=2)
        self.rtsp_entry.insert(0, "rtsp://example.com/stream")
        
        self.rtsp_btn = tk.Button(
            rtsp_frame,
            text="📡 CONNECT RTSP",
            command=self.start_rtsp_analysis,
            bg=self.colors['accent_primary'],
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief='flat',
            padx=15,
            pady=6,
            cursor='hand2'
        )
        self.rtsp_btn.pack(pady=5)
        
        # Control buttons
        control_frame = tk.LabelFrame(
            right_panel,
            text="🎮 CONTROLS",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['accent_warning'],
            bd=2,
            relief='solid'
        )
        control_frame.pack(fill='x', pady=8)
        
        self.stop_btn = tk.Button(
            control_frame,
            text="⏹️ STOP ANALYSIS",
            command=self.stop_video_analysis,
            bg=self.colors['accent_danger'],
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            relief='flat',
            padx=20,
            pady=8,
            cursor='hand2',
            state='disabled'
        )
        self.stop_btn.pack(pady=8)
        
        # Performance settings
        perf_frame = tk.Frame(control_frame, bg=self.colors['bg_primary'])
        perf_frame.pack(fill='x', pady=5)
        
        tk.Label(perf_frame, text="Processing FPS:", bg=self.colors['bg_primary'], fg=self.colors['text_secondary'], font=('Segoe UI', 9)).pack(anchor='w')
        
        self.fps_var = tk.IntVar(value=5)
        fps_scale = tk.Scale(
            perf_frame,
            from_=1,
            to=15,
            orient='horizontal',
            variable=self.fps_var,
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary'],
            highlightthickness=0,
            length=200
        )
        fps_scale.pack(fill='x')
        
        # Live results display
        results_frame = tk.LabelFrame(
            right_panel,
            text="📊 LIVE RESULTS",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['accent_success'],
            bd=2,
            relief='solid'
        )
        results_frame.pack(fill='both', expand=True, pady=8)
        
        self.video_results_text = scrolledtext.ScrolledText(
            results_frame,
            height=12,
            font=('Consolas', 9),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            insertbackground=self.colors['accent_primary'],
            selectbackground=self.colors['accent_primary']
        )
        self.video_results_text.pack(fill='both', expand=True, padx=8, pady=8)
        
        # Save video results
        self.save_video_btn = tk.Button(
            right_panel,
            text="💾 SAVE VIDEO RESULTS",
            command=self.save_video_results,
            bg=self.colors['accent_warning'],
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            relief='flat',
            padx=20,
            pady=8,
            cursor='hand2',
            state='disabled'
        )
        self.save_video_btn.pack(pady=8)
    
    def setup_settings_view(self):
        """Setup the settings view"""
        # Placeholder for settings view
        container = tk.Frame(self.content_frame, bg=self.colors['bg_primary'])
        container.pack(fill='both', expand=True, padx=32, pady=24)
        
        title = tk.Label(
            container,
            text="Settings",
            font=('Segoe UI', 24, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_primary']
        )
        title.pack(anchor='w')
        
        subtitle = tk.Label(
            container,
            text="Configure analysis parameters and system settings",
            font=('Segoe UI', 12),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg_primary']
        )
        subtitle.pack(anchor='w', pady=(4, 24))
        
        # Settings content
        settings_frame = tk.Frame(container, bg=self.colors['bg_secondary'], relief='solid', bd=1)
        settings_frame.pack(fill='x', pady=(0, 16))
        
        # Confidence threshold
        conf_header = tk.Label(
            settings_frame,
            text="🎯 Detection Settings",
            font=('Segoe UI', 14, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg_secondary']
        )
        conf_header.pack(anchor='w', padx=20, pady=(20, 8))
        
        tk.Label(
            settings_frame,
            text="Confidence Threshold:",
            font=('Segoe UI', 10),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg_secondary']
        ).pack(anchor='w', padx=20, pady=(0, 4))
        
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = tk.Scale(
            settings_frame,
            from_=0.1,
            to=0.9,
            resolution=0.1,
            orient='horizontal',
            variable=self.confidence_var,
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            highlightthickness=0,
            length=300
        )
        confidence_scale.pack(anchor='w', padx=20, pady=(0, 20))
        
        # Apply button
        apply_btn = tk.Button(
            container,
            text="Apply Settings",
            command=self.apply_settings,
            bg=self.colors['success'],
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            relief='flat',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        apply_btn.pack(anchor='w', pady=16)
    
    def initialize_analyzer(self):
        """Initialize the analyzer in a separate thread"""
        def init():
            try:
                self.analyzer = ImageAnalyzer(confidence_threshold=0.5)
                self.batch_processor = BatchProcessor(confidence_threshold=0.5, max_workers=4)
                self.video_analyzer = VideoAnalyzer(confidence_threshold=0.5, fps_limit=5)
                print("✅ All analyzers initialized successfully!")
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Failed to initialize analyzer: {msg}"))
        
        threading.Thread(target=init, daemon=True).start()
    
    def select_image(self):
        """Select an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.file_path_label.config(text=f"Selected: {os.path.basename(file_path)}")
            self.analyze_btn.config(state='normal')
            self.display_image(file_path)
    
    def add_photo_from_files(self):
        """Add photo from anywhere on the computer"""
        file_path = filedialog.askopenfilename(
            title="Add Photo from Files",
            initialdir=os.path.expanduser("~"),
            filetypes=[
                ("JPEG Images", "*.jpg *.jpeg"),
                ("PNG Images", "*.png"),
                ("BMP Images", "*.bmp"),
                ("TIFF Images", "*.tiff *.tif"),
                ("WebP Images", "*.webp"),
                ("All Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            try:
                test_image = Image.open(file_path)
                test_image.close()
                
                self.current_image_path = file_path
                self.file_path_label.config(
                    text=f"📷 Photo Added: {os.path.basename(file_path)}", 
                    fg=self.colors['accent_primary']
                )
                self.analyze_btn.config(state='normal')
                self.display_image(file_path)
                
                messagebox.showinfo(
                    "Photo Added", 
                    f"Successfully added photo:\n{os.path.basename(file_path)}\n\nReady for analysis!"
                )
                
            except Exception as e:
                error_msg = str(e)
                messagebox.showerror(
                    "Invalid Image", 
                    f"Could not load the selected file as an image.\n\nError: {error_msg}\n\nPlease select a valid image file."
                )
    
    def display_image(self, image_path):
        """Display the selected image"""
        try:
            image = Image.open(image_path)
            display_size = (500, 400)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            
        except Exception as e:
            error_msg = str(e)
            messagebox.showerror("Error", f"Could not display image: {error_msg}")
    
    def analyze_single_image(self):
        """Analyze the selected image"""
        if not self.current_image_path or not self.analyzer:
            return
        
        def analyze():
            try:
                self.root.after(0, lambda: self.progress_single.start())
                self.root.after(0, lambda: self.analyze_btn.config(state='disabled'))
                
                result = self.analyzer.analyze_image(self.current_image_path)
                self.current_results = result
                
                results_text = f"""Analysis Results for: {os.path.basename(result.image_path)}
{'='*50}

👥 PEOPLE DETECTED: {result.people_count}
🚗 VEHICLES DETECTED: {result.vehicle_count}

🚦 TRAFFIC LIGHTS:
   Total: {result.traffic_lights['total']}
   Red: {result.traffic_lights['red']}
   Green: {result.traffic_lights['green']}
   Yellow: {result.traffic_lights['yellow']}

📊 CONFIDENCE SCORES:
   People: {result.confidence_scores['people']:.3f}
   Vehicles: {result.confidence_scores['vehicles']:.3f}
   Traffic Lights: {result.confidence_scores['traffic_lights']:.3f}

⏱️ Processing Time: {result.processing_time:.3f} seconds
📅 Timestamp: {result.timestamp}

JSON Output:
{json.dumps({
    'people_count': result.people_count,
    'vehicle_count': result.vehicle_count,
    'traffic_lights': result.traffic_lights,
    'confidence_scores': result.confidence_scores,
    'processing_time': result.processing_time,
    'image_path': result.image_path,
    'timestamp': result.timestamp
}, indent=2)}"""
                
                self.root.after(0, lambda text=results_text: self.update_single_results(text))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Analysis failed: {msg}"))
            finally:
                self.root.after(0, lambda: self.progress_single.stop())
                self.root.after(0, lambda: self.analyze_btn.config(state='normal'))
        
        threading.Thread(target=analyze, daemon=True).start()
    
    def update_single_results(self, results_text):
        """Update the results display"""
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert('1.0', results_text)
        self.save_single_btn.config(state='normal')
    
    def save_single_results(self):
        """Save single image results to file"""
        if not self.current_results:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                result_dict = {
                    "people_count": self.current_results.people_count,
                    "vehicle_count": self.current_results.vehicle_count,
                    "traffic_lights": self.current_results.traffic_lights,
                    "confidence_scores": self.current_results.confidence_scores,
                    "processing_time": self.current_results.processing_time,
                    "image_path": self.current_results.image_path,
                    "timestamp": self.current_results.timestamp
                }
                
                with open(file_path, 'w') as f:
                    json.dump(result_dict, f, indent=2)
                
                messagebox.showinfo("Success", f"Results saved to {file_path}")
                
            except Exception as e:
                error_msg = str(e)
                messagebox.showerror("Error", f"Could not save results: {error_msg}")
    
    def select_directory(self):
        """Select directory for batch processing"""
        dir_path = filedialog.askdirectory(title="Select Directory with Images")
        
        if dir_path:
            self.current_dir_path = dir_path
            self.dir_path_label.config(text=f"Selected: {dir_path}")
            self.process_btn.config(state='normal')
    
    def start_batch_processing(self):
        """Start batch processing"""
        if not hasattr(self, 'current_dir_path') or not self.batch_processor:
            return
        
        def process():
            try:
                self.root.after(0, lambda: self.process_btn.config(state='disabled'))
                self.root.after(0, lambda: self.progress_label.config(text="Finding images..."))
                
                # Find images
                image_paths = self.batch_processor.find_images(self.current_dir_path)
                
                if not image_paths:
                    self.root.after(0, lambda: messagebox.showinfo("Info", "No images found in selected directory"))
                    return
                
                self.root.after(0, lambda count=len(image_paths): self.progress_batch.config(maximum=count))
                self.root.after(0, lambda count=len(image_paths): self.progress_label.config(text=f"Processing {count} images..."))
                
                # Process images
                results = self.batch_processor.process_directory(
                    input_dir=self.current_dir_path,
                    parallel=self.parallel_var.get()
                )
                
                # Format results
                total_people = sum(r.people_count for r in results)
                total_vehicles = sum(r.vehicle_count for r in results)
                total_lights = sum(r.traffic_lights['total'] for r in results)
                avg_time = sum(r.processing_time for r in results) / len(results) if results else 0
                
                summary = f"""Batch Processing Results
{'='*50}

📊 SUMMARY:
   Images Processed: {len(results)}
   Total People: {total_people}
   Total Vehicles: {total_vehicles}
   Total Traffic Lights: {total_lights}
   Average Processing Time: {avg_time:.3f}s

📋 DETAILED RESULTS:
"""
                
                for i, result in enumerate(results, 1):
                    summary += f"\n{i}. {os.path.basename(result.image_path)}:\n"
                    summary += f"   People: {result.people_count}, Vehicles: {result.vehicle_count}, "
                    summary += f"Traffic Lights: {result.traffic_lights['total']}\n"
                
                self.batch_results = results
                self.root.after(0, lambda text=summary: self.update_batch_results(text))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Batch processing failed: {msg}"))
            finally:
                self.root.after(0, lambda: self.process_btn.config(state='normal'))
                self.root.after(0, lambda: self.progress_label.config(text="Processing complete"))
        
        threading.Thread(target=process, daemon=True).start()
    
    def update_batch_results(self, results_text):
        """Update batch results display"""
        self.batch_results_text.delete('1.0', tk.END)
        self.batch_results_text.insert('1.0', results_text)
        self.save_batch_btn.config(state='normal')
    
    def save_batch_results(self):
        """Save batch results to file"""
        if not hasattr(self, 'batch_results'):
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Batch Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.batch_processor.save_results_with_summary(self.batch_results, file_path)
                messagebox.showinfo("Success", f"Batch results saved to {file_path}")
            except Exception as e:
                error_msg = str(e)
                messagebox.showerror("Error", f"Could not save results: {error_msg}")
    
    def start_webcam_analysis(self):
        """Start webcam analysis"""
        if self.is_video_running:
            return
        
        def webcam_thread():
            try:
                if not self.video_analyzer:
                    self.video_analyzer = VideoAnalyzer(
                        confidence_threshold=0.5,
                        fps_limit=self.fps_var.get()
                    )
                
                self.video_analyzer.set_callbacks(
                    frame_callback=self.update_video_display,
                    results_callback=self.update_video_results
                )
                
                self.is_video_running = True
                self.root.after(0, lambda: self.stop_btn.config(state='normal'))
                self.root.after(0, lambda: self.webcam_btn.config(state='disabled'))
                self.root.after(0, lambda: self.save_video_btn.config(state='normal'))
                
                self.video_analyzer.analyze_webcam(camera_index=0, display_window=False)
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Webcam Error", f"Could not start webcam: {msg}"))
            finally:
                self.is_video_running = False
                self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
                self.root.after(0, lambda: self.webcam_btn.config(state='normal'))
        
        self.video_thread = threading.Thread(target=webcam_thread, daemon=True)
        self.video_thread.start()
    
    def select_video_file(self):
        """Select and analyze video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            def video_thread():
                try:
                    if not self.video_analyzer:
                        self.video_analyzer = VideoAnalyzer(
                            confidence_threshold=0.5,
                            fps_limit=self.fps_var.get()
                        )
                    
                    self.video_analyzer.set_callbacks(
                        frame_callback=self.update_video_display,
                        results_callback=self.update_video_results
                    )
                    
                    self.is_video_running = True
                    self.root.after(0, lambda: self.stop_btn.config(state='normal'))
                    self.root.after(0, lambda: self.video_file_btn.config(state='disabled'))
                    self.root.after(0, lambda: self.save_video_btn.config(state='normal'))
                    
                    self.video_analyzer.analyze_video_file(
                        video_path=file_path,
                        display_window=False
                    )
                    
                except Exception as e:
                    error_msg = str(e)
                    self.root.after(0, lambda msg=error_msg: messagebox.showerror("Video Error", f"Could not analyze video: {msg}"))
                finally:
                    self.is_video_running = False
                    self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
                    self.root.after(0, lambda: self.video_file_btn.config(state='normal'))
            
            self.video_thread = threading.Thread(target=video_thread, daemon=True)
            self.video_thread.start()
    
    def start_rtsp_analysis(self):
        """Start RTSP stream analysis"""
        if self.is_video_running:
            return
        
        rtsp_url = self.rtsp_entry.get().strip()
        if not rtsp_url or rtsp_url == "rtsp://example.com/stream":
            messagebox.showwarning("RTSP URL", "Please enter a valid RTSP URL")
            return
        
        def rtsp_thread():
            try:
                if not self.video_analyzer:
                    self.video_analyzer = VideoAnalyzer(
                        confidence_threshold=0.5,
                        fps_limit=self.fps_var.get()
                    )
                
                self.video_analyzer.set_callbacks(
                    frame_callback=self.update_video_display,
                    results_callback=self.update_video_results
                )
                
                self.is_video_running = True
                self.root.after(0, lambda: self.stop_btn.config(state='normal'))
                self.root.after(0, lambda: self.rtsp_btn.config(state='disabled'))
                self.root.after(0, lambda: self.save_video_btn.config(state='normal'))
                
                self.video_analyzer.analyze_rtsp_stream(rtsp_url, display_window=False)
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("RTSP Error", f"Could not connect to RTSP stream: {msg}"))
            finally:
                self.is_video_running = False
                self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
                self.root.after(0, lambda: self.rtsp_btn.config(state='normal'))
        
        self.video_thread = threading.Thread(target=rtsp_thread, daemon=True)
        self.video_thread.start()
    
    def stop_video_analysis(self):
        """Stop video analysis"""
        if self.video_analyzer:
            self.video_analyzer.stop_processing()
        
        self.is_video_running = False
        self.stop_btn.config(state='disabled')
        self.webcam_btn.config(state='normal')
        self.video_file_btn.config(state='normal')
        self.rtsp_btn.config(state='normal')
    
    def update_video_display(self, frame):
        """Update video display with current frame"""
        def update_display():
            try:
                display_height = 400
                height, width = frame.shape[:2]
                aspect_ratio = width / height
                display_width = int(display_height * aspect_ratio)
                
                resized_frame = cv2.resize(frame, (display_width, display_height))
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(pil_image)
                
                self.video_label.config(image=photo, text="")
                self.video_label.image = photo
                
            except Exception as e:
                print(f"Error updating video display: {str(e)}")
        
        self.root.after(0, update_display)
    
    def update_video_results(self, result):
        """Update video results display in real-time"""
        def update_ui():
            result_text = f"""Frame {result.frame_number} - {time.strftime('%H:%M:%S')}
👥 People: {result.people_count}
🚗 Vehicles: {result.vehicle_count}
🚦 Traffic Lights: {result.traffic_lights['total']} (R:{result.traffic_lights['red']} G:{result.traffic_lights['green']} Y:{result.traffic_lights['yellow']})
⚡ Processing: {result.processing_time:.3f}s
📊 Confidence: P:{result.confidence_scores['people']:.2f} V:{result.confidence_scores['vehicles']:.2f} T:{result.confidence_scores['traffic_lights']:.2f}

"""
            
            self.video_results_text.insert('1.0', result_text)
            
            content = self.video_results_text.get('1.0', tk.END)
            lines = content.split('\n')
            if len(lines) > 300:
                truncated = '\n'.join(lines[:300])
                self.video_results_text.delete('1.0', tk.END)
                self.video_results_text.insert('1.0', truncated)
        
        self.root.after(0, update_ui)
    
    def save_video_results(self):
        """Save video analysis results"""
        if not self.video_analyzer or not self.video_analyzer.results_history:
            messagebox.showwarning("No Results", "No video analysis results to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Video Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.video_analyzer.save_results_to_file(file_path)
                
                stats = self.video_analyzer.get_statistics()
                stats_msg = f"""Video Analysis Results Saved!

Statistics:
• Frames Processed: {stats.get('total_frames_processed', 0)}
• Average People: {stats.get('average_people', 0):.1f}
• Max People: {stats.get('max_people', 0)}
• Average Vehicles: {stats.get('average_vehicles', 0):.1f}
• Max Vehicles: {stats.get('max_vehicles', 0)}
• Average FPS: {stats.get('average_fps', 0):.1f}"""
                
                messagebox.showinfo("Results Saved", stats_msg)
                
            except Exception as e:
                error_msg = str(e)
                messagebox.showerror("Save Error", f"Could not save results: {error_msg}")
    
    def apply_settings(self):
        """Apply new settings"""
        try:
            confidence = self.confidence_var.get()
            
            self.analyzer = ImageAnalyzer(confidence_threshold=confidence)
            self.batch_processor = BatchProcessor(confidence_threshold=confidence, max_workers=4)
            if self.video_analyzer:
                self.video_analyzer = VideoAnalyzer(confidence_threshold=confidence, fps_limit=5)
            
            messagebox.showinfo("Success", f"Settings applied:\nConfidence: {confidence}")
            
        except Exception as e:
            error_msg = str(e)
            messagebox.showerror("Error", f"Could not apply settings: {error_msg}")

def main():
    """Launch the GUI application"""
    root = tk.Tk()
    app = ImageAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()elf):
        """Setup the single image analysis tab"""
        # Left panel for image
        left_panel = tk.Frame(self.single_frame, bg='white', relief='sunken', bd=2)
        left_panel.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Image display area
        self.image_label = tk.Label(
            left_panel, 
            text="Click 'Browse Files' or 'Add Photo' to load an image", 
            bg='white',
            font=('Arial', 12),
            fg='gray'
        )
        self.image_label.pack(expand=True)
        
        # Right panel for controls and results
        right_panel = tk.Frame(self.single_frame, width=350)
        right_panel.pack(side='right', fill='y', padx=5, pady=5)
        right_panel.pack_propagate(False)
        
        # File selection
        file_frame = tk.LabelFrame(right_panel, text="Image Selection", font=('Arial', 10, 'bold'))
        file_frame.pack(fill='x', pady=5)
        
        # Button container for better layout
        button_container = tk.Frame(file_frame)
        button_container.pack(pady=10)
        
        self.select_btn = tk.Button(
            button_container, 
            text="📁 Browse Files", 
            command=self.select_image,
            bg='#3498db', 
            fg='white', 
            font=('Arial', 10, 'bold'),
            relief='flat',
            padx=15,
            pady=5
        )
        self.select_btn.pack(side='left', padx=5)
        
        self.add_photo_btn = tk.Button(
            button_container, 
            text="📷 Add Photo", 
            command=self.add_photo_from_files,
            bg='#9b59b6', 
            fg='white', 
            font=('Arial', 10, 'bold'),
            relief='flat',
            padx=15,
            pady=5
        )
        self.add_photo_btn.pack(side='left', padx=5)
        
        self.file_path_label = tk.Label(file_frame, text="No file selected", wraplength=300, justify='left')
        self.file_path_label.pack(pady=5)
        
        # Analysis button
        self.analyze_btn = tk.Button(
            right_panel,
            text="🔍 Analyze Image",
            command=self.analyze_single_image,
            bg='#27ae60',
            fg='white',
            font=('Arial', 12, 'bold'),
            relief='flat',
            padx=20,
            pady=10,
            state='disabled'
        )
        self.analyze_btn.pack(pady=10)
        
        # Progress bar
        self.progress_single = ttk.Progressbar(right_panel, mode='indeterminate')
        self.progress_single.pack(fill='x', pady=5)
        
        # Results display
        results_frame = tk.LabelFrame(right_panel, text="Analysis Results", font=('Arial', 10, 'bold'))
        results_frame.pack(fill='both', expand=True, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(
            results_frame, 
            height=15, 
            font=('Consolas', 9),
            bg='#f8f9fa'
        )
        self.results_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Save results button
        self.save_single_btn = tk.Button(
            right_panel,
            text="💾 Save Results",
            command=self.save_single_results,
            bg='#f39c12',
            fg='white',
            font=('Arial', 10),
            relief='flat',
            state='disabled'
        )
        self.save_single_btn.pack(pady=5)
    
    def setup_batch_processing_tab(self):
        """Setup the batch processing tab"""
        # Top controls
        controls_frame = tk.Frame(self.batch_frame)
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        # Directory selection
        dir_frame = tk.LabelFrame(controls_frame, text="Directory Selection", font=('Arial', 10, 'bold'))
        dir_frame.pack(fill='x', pady=5)
        
        dir_btn_frame = tk.Frame(dir_frame)
        dir_btn_frame.pack(fill='x', padx=5, pady=5)
        
        self.select_dir_btn = tk.Button(
            dir_btn_frame,
            text="📂 Select Directory",
            command=self.select_directory,
            bg='#3498db',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief='flat'
        )
        self.select_dir_btn.pack(side='left', padx=5)
        
        self.dir_path_label = tk.Label(dir_btn_frame, text="No directory selected")
        self.dir_path_label.pack(side='left', padx=10)
        
        # Processing controls
        process_frame = tk.Frame(dir_frame)
        process_frame.pack(fill='x', padx=5, pady=5)
        
        self.process_btn = tk.Button(
            process_frame,
            text="⚡ Start Batch Processing",
            command=self.start_batch_processing,
            bg='#27ae60',
            fg='white',
            font=('Arial', 12, 'bold'),
            relief='flat',
            state='disabled'
        )
        self.process_btn.pack(side='left', padx=5)
        
        # Parallel processing checkbox
        self.parallel_var = tk.BooleanVar(value=True)
        parallel_check = tk.Checkbutton(
            process_frame,
            text="Parallel Processing",
            variable=self.parallel_var,
            font=('Arial', 9)
        )
        parallel_check.pack(side='left', padx=20)
        
        # Progress section
        progress_frame = tk.LabelFrame(controls_frame, text="Progress", font=('Arial', 10, 'bold'))
        progress_frame.pack(fill='x', pady=5)
        
        self.progress_batch = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_batch.pack(fill='x', padx=5, pady=5)
        
        self.progress_label = tk.Label(progress_frame, text="Ready to process")
        self.progress_label.pack(pady=2)
        
        # Results area
        results_frame = tk.LabelFrame(self.batch_frame, text="Batch Results", font=('Arial', 10, 'bold'))
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.batch_results_text = scrolledtext.ScrolledText(
            results_frame,
            font=('Consolas', 9),
            bg='#f8f9fa'
        )
        self.batch_results_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Save batch results
        self.save_batch_btn = tk.Button(
            self.batch_frame,
            text="💾 Save Batch Results",
            command=self.save_batch_results,
            bg='#f39c12',
            fg='white',
            font=('Arial', 10),
            relief='flat',
            state='disabled'
        )
        self.save_batch_btn.pack(pady=5)
    
    def setup_video_analysis_tab(self):
        """Setup the video analysis tab"""
        # Main container with light theme
        main_container = tk.Frame(self.video_frame, bg=self.colors['bg_primary'])
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel for video display
        left_panel = tk.Frame(main_container, bg=self.colors['bg_secondary'], relief='solid', bd=2)
        left_panel.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Video display area
        self.video_label = tk.Label(
            left_panel,
            text="🎥 LIVE VIDEO FEED\n\nSelect a video source to start analysis",
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_secondary'],
            font=('Segoe UI', 14, 'bold'),
            justify='center'
        )
        self.video_label.pack(expand=True)
        
        # Right panel for controls
        right_panel = tk.Frame(main_container, width=380, bg=self.colors['bg_primary'])
        right_panel.pack(side='right', fill='y', padx=5, pady=5)
        right_panel.pack_propagate(False)
        
        # Video source selection
        source_frame = tk.LabelFrame(
            right_panel,
            text="🎬 VIDEO SOURCE",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['accent_secondary'],
            bd=2,
            relief='solid'
        )
        source_frame.pack(fill='x', pady=8)
        
        # Webcam button
        self.webcam_btn = tk.Button(
            source_frame,
            text="📹 START WEBCAM",
            command=self.start_webcam_analysis,
            bg='#4caf50',
            fg='white',
            font=('Arial', 11, 'bold'),
            relief='flat',
            padx=20,
            pady=8,
            cursor='hand2'
        )
        self.webcam_btn.pack(pady=8)
        
        # Video file button
        self.video_file_btn = tk.Button(
            source_frame,
            text="📁 SELECT VIDEO FILE",
            command=self.select_video_file,
            bg='#2196f3',
            fg='white',
            font=('Arial', 11, 'bold'),
            relief='flat',
            padx=20,
            pady=8,
            cursor='hand2'
        )
        self.video_file_btn.pack(pady=8)
        
        # RTSP stream input
        rtsp_frame = tk.Frame(source_frame, bg=self.colors['bg_primary'])
        rtsp_frame.pack(fill='x', pady=8)
        
        tk.Label(rtsp_frame, text="RTSP URL:", bg=self.colors['bg_primary'], fg=self.colors['text_secondary'], font=('Segoe UI', 9)).pack(anchor='w')
        
        self.rtsp_entry = tk.Entry(
            rtsp_frame,
            font=('Segoe UI', 9),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            insertbackground=self.colors['accent_primary']
        )
        self.rtsp_entry.pack(fill='x', pady=2)
        self.rtsp_entry.insert(0, "rtsp://example.com/stream")
        
        self.rtsp_btn = tk.Button(
            rtsp_frame,
            text="📡 CONNECT RTSP",
            command=self.start_rtsp_analysis,
            bg=self.colors['accent_primary'],
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief='flat',
            padx=15,
            pady=6,
            cursor='hand2'
        )
        self.rtsp_btn.pack(pady=5)
        
        # Control buttons
        control_frame = tk.LabelFrame(
            right_panel,
            text="🎮 CONTROLS",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['accent_warning'],
            bd=2,
            relief='solid'
        )
        control_frame.pack(fill='x', pady=8)
        
        self.stop_btn = tk.Button(
            control_frame,
            text="⏹️ STOP ANALYSIS",
            command=self.stop_video_analysis,
            bg='#f44336',
            fg='white',
            font=('Arial', 11, 'bold'),
            relief='flat',
            padx=20,
            pady=8,
            cursor='hand2',
            state='disabled'
        )
        self.stop_btn.pack(pady=8)
        
        # Performance settings
        perf_frame = tk.Frame(control_frame, bg=self.colors['bg_primary'])
        perf_frame.pack(fill='x', pady=5)
        
        tk.Label(perf_frame, text="Processing FPS:", bg=self.colors['bg_primary'], fg=self.colors['text_secondary'], font=('Segoe UI', 9)).pack(anchor='w')
        
        self.fps_var = tk.IntVar(value=5)
        fps_scale = tk.Scale(
            perf_frame,
            from_=1,
            to=15,
            orient='horizontal',
            variable=self.fps_var,
            bg=self.colors['bg_primary'],
            fg=self.colors['text_primary'],
            highlightthickness=0,
            length=200
        )
        fps_scale.pack(fill='x')
        
        # Live results display
        results_frame = tk.LabelFrame(
            right_panel,
            text="📊 LIVE RESULTS",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['bg_primary'],
            fg=self.colors['accent_success'],
            bd=2,
            relief='solid'
        )
        results_frame.pack(fill='both', expand=True, pady=8)
        
        self.video_results_text = scrolledtext.ScrolledText(
            results_frame,
            height=12,
            font=('Consolas', 9),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            insertbackground=self.colors['accent_primary'],
            selectbackground=self.colors['accent_primary']
        )
        self.video_results_text.pack(fill='both', expand=True, padx=8, pady=8)
        
        # Save video results
        self.save_video_btn = tk.Button(
            right_panel,
            text="💾 SAVE VIDEO RESULTS",
            command=self.save_video_results,
            bg=self.colors['accent_warning'],
            fg=self.colors['text_primary'],
            font=('Segoe UI', 11, 'bold'),
            relief='flat',
            padx=20,
            pady=8,
            cursor='hand2',
            state='disabled'
        )
        self.save_video_btn.pack(pady=8)
    
    def setup_settings_tab(self):
        """Setup the settings tab"""
        settings_main = tk.Frame(self.settings_frame)
        settings_main.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Confidence threshold
        conf_frame = tk.LabelFrame(settings_main, text="Detection Settings", font=('Arial', 10, 'bold'))
        conf_frame.pack(fill='x', pady=10)
        
        tk.Label(conf_frame, text="Confidence Threshold:", font=('Arial', 10)).pack(anchor='w', padx=10, pady=5)
        
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = tk.Scale(
            conf_frame,
            from_=0.1,
            to=0.9,
            resolution=0.1,
            orient='horizontal',
            variable=self.confidence_var,
            length=300
        )
        confidence_scale.pack(padx=10, pady=5)
        
        # Workers setting
        workers_frame = tk.LabelFrame(settings_main, text="Performance Settings", font=('Arial', 10, 'bold'))
        workers_frame.pack(fill='x', pady=10)
        
        tk.Label(workers_frame, text="Number of Workers (for batch processing):", font=('Arial', 10)).pack(anchor='w', padx=10, pady=5)
        
        self.workers_var = tk.IntVar(value=4)
        workers_scale = tk.Scale(
            workers_frame,
            from_=1,
            to=8,
            orient='horizontal',
            variable=self.workers_var,
            length=300
        )
        workers_scale.pack(padx=10, pady=5)
        
        # Apply settings button
        apply_btn = tk.Button(
            settings_main,
            text="✅ Apply Settings",
            command=self.apply_settings,
            bg='#27ae60',
            fg='white',
            font=('Arial', 11, 'bold'),
            relief='flat',
            padx=20,
            pady=10
        )
        apply_btn.pack(pady=20)
        
        # System info
        info_frame = tk.LabelFrame(settings_main, text="System Information", font=('Arial', 10, 'bold'))
        info_frame.pack(fill='x', pady=10)
        
        info_text = scrolledtext.ScrolledText(info_frame, height=8, font=('Consolas', 9))
        info_text.pack(fill='x', padx=10, pady=10)
        
        # Add system info
        import torch
        info_content = f"""System Information:
- Python Version: {os.sys.version.split()[0]}
- PyTorch Version: {torch.__version__}
- CUDA Available: {torch.cuda.is_available()}
- Device: {'GPU' if torch.cuda.is_available() else 'CPU'}
- Working Directory: {os.getcwd()}
"""
        info_text.insert('1.0', info_content)
        info_text.config(state='disabled')
    
    def initialize_analyzer(self):
        """Initialize the analyzer in a separate thread"""
        def init():
            try:
                self.analyzer = ImageAnalyzer(confidence_threshold=0.5)
                self.batch_processor = BatchProcessor(confidence_threshold=0.5, max_workers=4)
                self.root.after(0, lambda: messagebox.showinfo("Success", "Image analyzer initialized successfully!"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to initialize analyzer: {str(e)}"))
        
        threading.Thread(target=init, daemon=True).start()
    
    def select_image(self):
        """Select an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.file_path_label.config(text=f"Selected: {os.path.basename(file_path)}")
            self.analyze_btn.config(state='normal')
            self.display_image(file_path)
    
    def add_photo_from_files(self):
        """Add photo from anywhere on the computer with enhanced file browser"""
        file_path = filedialog.askopenfilename(
            title="Add Photo from Files",
            initialdir=os.path.expanduser("~"),  # Start from user's home directory
            filetypes=[
                ("JPEG Images", "*.jpg *.jpeg"),
                ("PNG Images", "*.png"),
                ("BMP Images", "*.bmp"),
                ("TIFF Images", "*.tiff *.tif"),
                ("WebP Images", "*.webp"),
                ("All Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            # Validate that it's actually an image file
            try:
                # Try to open the image to validate it
                test_image = Image.open(file_path)
                test_image.close()
                
                # If successful, set as current image
                self.current_image_path = file_path
                self.file_path_label.config(
                    text=f"📷 Photo Added: {os.path.basename(file_path)}", 
                    fg='#9b59b6'
                )
                self.analyze_btn.config(state='normal')
                self.display_image(file_path)
                
                # Show success message
                messagebox.showinfo(
                    "Photo Added", 
                    f"Successfully added photo:\n{os.path.basename(file_path)}\n\nReady for analysis!"
                )
                
            except Exception as e:
                messagebox.showerror(
                    "Invalid Image", 
                    f"Could not load the selected file as an image.\n\nError: {str(e)}\n\nPlease select a valid image file."
                )
    
    def display_image(self, image_path):
        """Display the selected image"""
        try:
            # Load and resize image for display
            image = Image.open(image_path)
            
            # Calculate display size (max 400x400)
            display_size = (400, 400)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not display image: {str(e)}")
    
    def analyze_single_image(self):
        """Analyze the selected image"""
        if not self.current_image_path or not self.analyzer:
            return
        
        def analyze():
            try:
                self.root.after(0, lambda: self.progress_single.start())
                self.root.after(0, lambda: self.analyze_btn.config(state='disabled'))
                
                result = self.analyzer.analyze_image(self.current_image_path)
                self.current_results = result
                
                # Format results for display
                results_text = f"""Analysis Results for: {os.path.basename(result.image_path)}
{'='*50}

👥 PEOPLE DETECTED: {result.people_count}
🚗 VEHICLES DETECTED: {result.vehicle_count}

🚦 TRAFFIC LIGHTS:
   Total: {result.traffic_lights['total']}
   Red: {result.traffic_lights['red']}
   Green: {result.traffic_lights['green']}
   Yellow: {result.traffic_lights['yellow']}

📊 CONFIDENCE SCORES:
   People: {result.confidence_scores['people']:.3f}
   Vehicles: {result.confidence_scores['vehicles']:.3f}
   Traffic Lights: {result.confidence_scores['traffic_lights']:.3f}

⏱️ Processing Time: {result.processing_time:.3f} seconds
📅 Timestamp: {result.timestamp}

JSON Output:
{json.dumps({
    'people_count': result.people_count,
    'vehicle_count': result.vehicle_count,
    'traffic_lights': result.traffic_lights,
    'confidence_scores': result.confidence_scores,
    'processing_time': result.processing_time,
    'image_path': result.image_path,
    'timestamp': result.timestamp
}, indent=2)}"""
                
                self.root.after(0, lambda: self.update_single_results(results_text))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
            finally:
                self.root.after(0, lambda: self.progress_single.stop())
                self.root.after(0, lambda: self.analyze_btn.config(state='normal'))
        
        threading.Thread(target=analyze, daemon=True).start()
    
    def update_single_results(self, results_text):
        """Update the results display"""
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert('1.0', results_text)
        self.save_single_btn.config(state='normal')
    
    def save_single_results(self):
        """Save single image results to file"""
        if not self.current_results:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                result_dict = {
                    "people_count": self.current_results.people_count,
                    "vehicle_count": self.current_results.vehicle_count,
                    "traffic_lights": self.current_results.traffic_lights,
                    "confidence_scores": self.current_results.confidence_scores,
                    "processing_time": self.current_results.processing_time,
                    "image_path": self.current_results.image_path,
                    "timestamp": self.current_results.timestamp
                }
                
                with open(file_path, 'w') as f:
                    json.dump(result_dict, f, indent=2)
                
                messagebox.showinfo("Success", f"Results saved to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not save results: {str(e)}")
    
    def select_directory(self):
        """Select directory for batch processing"""
        dir_path = filedialog.askdirectory(title="Select Directory with Images")
        
        if dir_path:
            self.current_dir_path = dir_path
            self.dir_path_label.config(text=f"Selected: {dir_path}")
            self.process_btn.config(state='normal')
    
    def start_batch_processing(self):
        """Start batch processing"""
        if not hasattr(self, 'current_dir_path') or not self.batch_processor:
            return
        
        def process():
            try:
                self.root.after(0, lambda: self.process_btn.config(state='disabled'))
                self.root.after(0, lambda: self.progress_label.config(text="Finding images..."))
                
                # Find images
                image_paths = self.batch_processor.find_images(self.current_dir_path)
                
                if not image_paths:
                    self.root.after(0, lambda: messagebox.showinfo("Info", "No images found in selected directory"))
                    return
                
                self.root.after(0, lambda: self.progress_batch.config(maximum=len(image_paths)))
                self.root.after(0, lambda: self.progress_label.config(text=f"Processing {len(image_paths)} images..."))
                
                # Process images
                results = self.batch_processor.process_directory(
                    input_dir=self.current_dir_path,
                    parallel=self.parallel_var.get()
                )
                
                # Format results
                total_people = sum(r.people_count for r in results)
                total_vehicles = sum(r.vehicle_count for r in results)
                total_lights = sum(r.traffic_lights['total'] for r in results)
                avg_time = sum(r.processing_time for r in results) / len(results) if results else 0
                
                summary = f"""Batch Processing Results
{'='*50}

📊 SUMMARY:
   Images Processed: {len(results)}
   Total People: {total_people}
   Total Vehicles: {total_vehicles}
   Total Traffic Lights: {total_lights}
   Average Processing Time: {avg_time:.3f}s

📋 DETAILED RESULTS:
"""
                
                for i, result in enumerate(results, 1):
                    summary += f"\n{i}. {os.path.basename(result.image_path)}:\n"
                    summary += f"   People: {result.people_count}, Vehicles: {result.vehicle_count}, "
                    summary += f"Traffic Lights: {result.traffic_lights['total']}\n"
                
                self.batch_results = results
                self.root.after(0, lambda: self.update_batch_results(summary))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Batch processing failed: {str(e)}"))
            finally:
                self.root.after(0, lambda: self.process_btn.config(state='normal'))
                self.root.after(0, lambda: self.progress_label.config(text="Processing complete"))
        
        threading.Thread(target=process, daemon=True).start()
    
    def update_batch_results(self, results_text):
        """Update batch results display"""
        self.batch_results_text.delete('1.0', tk.END)
        self.batch_results_text.insert('1.0', results_text)
        self.save_batch_btn.config(state='normal')
    
    def save_batch_results(self):
        """Save batch results to file"""
        if not hasattr(self, 'batch_results'):
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Batch Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.batch_processor.save_results_with_summary(self.batch_results, file_path)
                messagebox.showinfo("Success", f"Batch results saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save results: {str(e)}")
    
    def apply_settings(self):
        """Apply new settings"""
        try:
            confidence = self.confidence_var.get()
            workers = self.workers_var.get()
            
            # Reinitialize with new settings
            self.analyzer = ImageAnalyzer(confidence_threshold=confidence)
            self.batch_processor = BatchProcessor(confidence_threshold=confidence, max_workers=workers)
            
            messagebox.showinfo("Success", f"Settings applied:\nConfidence: {confidence}\nWorkers: {workers}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not apply settings: {str(e)}")
    
    def start_webcam_analysis(self):
        """Start webcam analysis"""
        if self.is_video_running:
            return
        
        def webcam_thread():
            try:
                self.video_analyzer = VideoAnalyzer(
                    confidence_threshold=self.confidence_var.get(),
                    fps_limit=self.fps_var.get()
                )
                
                # Set up callbacks for real-time updates
                self.video_analyzer.set_callbacks(
                    frame_callback=self.update_video_display,
                    results_callback=self.update_video_results
                )
                
                self.is_video_running = True
                self.root.after(0, lambda: self.stop_btn.config(state='normal'))
                self.root.after(0, lambda: self.webcam_btn.config(state='disabled'))
                self.root.after(0, lambda: self.save_video_btn.config(state='normal'))
                
                # Start webcam analysis
                self.video_analyzer.analyze_webcam(camera_index=0, display_window=False)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Webcam Error", f"Could not start webcam: {str(e)}"))
            finally:
                self.is_video_running = False
                self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
                self.root.after(0, lambda: self.webcam_btn.config(state='normal'))
        
        self.video_thread = threading.Thread(target=webcam_thread, daemon=True)
        self.video_thread.start()
    
    def select_video_file(self):
        """Select and analyze video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            def video_thread():
                try:
                    self.video_analyzer = VideoAnalyzer(
                        confidence_threshold=self.confidence_var.get(),
                        fps_limit=self.fps_var.get()
                    )
                    
                    self.video_analyzer.set_callbacks(
                        frame_callback=self.update_video_display,
                        results_callback=self.update_video_results
                    )
                    
                    self.is_video_running = True
                    self.root.after(0, lambda: self.stop_btn.config(state='normal'))
                    self.root.after(0, lambda: self.video_file_btn.config(state='disabled'))
                    self.root.after(0, lambda: self.save_video_btn.config(state='normal'))
                    
                    # Analyze video file
                    self.video_analyzer.analyze_video_file(
                        video_path=file_path,
                        display_window=False
                    )
                    
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Video Error", f"Could not analyze video: {str(e)}"))
                finally:
                    self.is_video_running = False
                    self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
                    self.root.after(0, lambda: self.video_file_btn.config(state='normal'))
            
            self.video_thread = threading.Thread(target=video_thread, daemon=True)
            self.video_thread.start()
    
    def start_rtsp_analysis(self):
        """Start RTSP stream analysis"""
        if self.is_video_running:
            return
        
        rtsp_url = self.rtsp_entry.get().strip()
        if not rtsp_url or rtsp_url == "rtsp://example.com/stream":
            messagebox.showwarning("RTSP URL", "Please enter a valid RTSP URL")
            return
        
        def rtsp_thread():
            try:
                self.video_analyzer = VideoAnalyzer(
                    confidence_threshold=self.confidence_var.get(),
                    fps_limit=self.fps_var.get()
                )
                
                self.video_analyzer.set_callbacks(
                    results_callback=self.update_video_results
                )
                
                self.is_video_running = True
                self.root.after(0, lambda: self.stop_btn.config(state='normal'))
                self.root.after(0, lambda: self.rtsp_btn.config(state='disabled'))
                self.root.after(0, lambda: self.save_video_btn.config(state='normal'))
                
                # Start RTSP analysis
                self.video_analyzer.analyze_rtsp_stream(rtsp_url, display_window=False)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("RTSP Error", f"Could not connect to RTSP stream: {str(e)}"))
            finally:
                self.is_video_running = False
                self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
                self.root.after(0, lambda: self.rtsp_btn.config(state='normal'))
        
        self.video_thread = threading.Thread(target=rtsp_thread, daemon=True)
        self.video_thread.start()
    
    def stop_video_analysis(self):
        """Stop video analysis"""
        if self.video_analyzer:
            self.video_analyzer.stop_processing()
        
        self.is_video_running = False
        self.stop_btn.config(state='disabled')
        self.webcam_btn.config(state='normal')
        self.video_file_btn.config(state='normal')
        self.rtsp_btn.config(state='normal')
    
    def update_video_results(self, result):
        """Update video results display in real-time"""
        def update_ui():
            # Format the result for display
            result_text = f"""Frame {result.frame_number} - {time.strftime('%H:%M:%S')}
👥 People: {result.people_count}
🚗 Vehicles: {result.vehicle_count}
🚦 Traffic Lights: {result.traffic_lights['total']} (R:{result.traffic_lights['red']} G:{result.traffic_lights['green']} Y:{result.traffic_lights['yellow']})
⚡ Processing: {result.processing_time:.3f}s
📊 Confidence: P:{result.confidence_scores['people']:.2f} V:{result.confidence_scores['vehicles']:.2f} T:{result.confidence_scores['traffic_lights']:.2f}

"""
            
            # Insert at the beginning and limit text length
            self.video_results_text.insert('1.0', result_text)
            
            # Keep only last 50 results to prevent memory issues
            content = self.video_results_text.get('1.0', tk.END)
            lines = content.split('\n')
            if len(lines) > 300:  # ~50 results * 6 lines each
                truncated = '\n'.join(lines[:300])
                self.video_results_text.delete('1.0', tk.END)
                self.video_results_text.insert('1.0', truncated)
        
        # Schedule UI update in main thread
        self.root.after(0, update_ui)
    
    def update_video_display(self, frame):
        """Update video display with current frame"""
        def update_display():
            try:
                # Resize frame for display
                display_height = 400
                height, width = frame.shape[:2]
                aspect_ratio = width / height
                display_width = int(display_height * aspect_ratio)
                
                # Resize frame
                resized_frame = cv2.resize(frame, (display_width, display_height))
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(rgb_frame)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update video label
                self.video_label.config(image=photo, text="")
                self.video_label.image = photo  # Keep reference
                
            except Exception as e:
                print(f"Error updating video display: {str(e)}")
        
        # Schedule display update in main thread
        self.root.after(0, update_display)
    
    def save_video_results(self):
        """Save video analysis results"""
        if not self.video_analyzer or not self.video_analyzer.results_history:
            messagebox.showwarning("No Results", "No video analysis results to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Video Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.video_analyzer.save_results_to_file(file_path)
                
                # Show statistics
                stats = self.video_analyzer.get_statistics()
                stats_msg = f"""Video Analysis Results Saved!

Statistics:
• Frames Processed: {stats.get('total_frames_processed', 0)}
• Average People: {stats.get('average_people', 0):.1f}
• Max People: {stats.get('max_people', 0)}
• Average Vehicles: {stats.get('average_vehicles', 0):.1f}
• Max Vehicles: {stats.get('max_vehicles', 0)}
• Average FPS: {stats.get('average_fps', 0):.1f}"""
                
                messagebox.showinfo("Results Saved", stats_msg)
                
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save results: {str(e)}")

def main():
    """Launch the GUI application"""
    root = tk.Tk()
    app = ImageAnalysisGUI(root)
    root.mainloop()

    def initialize_analyzer(self):
        """Initialize the analyzer in a separate thread"""
        def init():
            try:
                self.analyzer = ImageAnalyzer(confidence_threshold=0.5)
                self.batch_processor = BatchProcessor(confidence_threshold=0.5, max_workers=4)
                self.video_analyzer = VideoAnalyzer(confidence_threshold=0.5, fps_limit=5)
                print("✅ All analyzers initialized successfully!")
            except Exception as e:
                print(f"❌ Error initializing analyzers: {str(e)}")
        
        threading.Thread(target=init, daemon=True).start()
    
    def select_image(self):
        """Select an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.file_path_label.config(text=f"Selected: {os.path.basename(file_path)}")
            self.analyze_btn.config(state='normal')
            self.display_image(file_path)
    
    def add_photo_from_files(self):
        """Add photo from anywhere on the computer with enhanced file browser"""
        file_path = filedialog.askopenfilename(
            title="Add Photo from Files",
            initialdir=os.path.expanduser("~"),
            filetypes=[
                ("JPEG Images", "*.jpg *.jpeg"),
                ("PNG Images", "*.png"),
                ("BMP Images", "*.bmp"),
                ("TIFF Images", "*.tiff *.tif"),
                ("WebP Images", "*.webp"),
                ("All Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            try:
                test_image = Image.open(file_path)
                test_image.close()
                
                self.current_image_path = file_path
                self.file_path_label.config(
                    text=f"📷 Photo Added: {os.path.basename(file_path)}", 
                    fg=self.colors['accent_primary']
                )
                self.analyze_btn.config(state='normal')
                self.display_image(file_path)
                
                messagebox.showinfo(
                    "Photo Added", 
                    f"Successfully added photo:\n{os.path.basename(file_path)}\n\nReady for analysis!"
                )
                
            except Exception as e:
                messagebox.showerror(
                    "Invalid Image", 
                    f"Could not load the selected file as an image.\n\nError: {str(e)}\n\nPlease select a valid image file."
                )
    
    def display_image(self, image_path):
        """Display the selected image"""
        try:
            image = Image.open(image_path)
            display_size = (500, 400)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not display image: {str(e)}")
    
    def analyze_single_image(self):
        """Analyze the selected image"""
        if not self.current_image_path or not self.analyzer:
            return
        
        def analyze():
            try:
                self.root.after(0, lambda: self.progress_single.start())
                self.root.after(0, lambda: self.analyze_btn.config(state='disabled'))
                
                result = self.analyzer.analyze_image(self.current_image_path)
                self.current_results = result
                
                results_text = f"""Analysis Results for: {os.path.basename(result.image_path)}
{'='*50}

👥 PEOPLE DETECTED: {result.people_count}
🚗 VEHICLES DETECTED: {result.vehicle_count}

🚦 TRAFFIC LIGHTS:
   Total: {result.traffic_lights['total']}
   Red: {result.traffic_lights['red']}
   Green: {result.traffic_lights['green']}
   Yellow: {result.traffic_lights['yellow']}

📊 CONFIDENCE SCORES:
   People: {result.confidence_scores['people']:.3f}
   Vehicles: {result.confidence_scores['vehicles']:.3f}
   Traffic Lights: {result.confidence_scores['traffic_lights']:.3f}

⏱️ Processing Time: {result.processing_time:.3f} seconds
📅 Timestamp: {result.timestamp}

JSON Output:
{json.dumps({
    'people_count': result.people_count,
    'vehicle_count': result.vehicle_count,
    'traffic_lights': result.traffic_lights,
    'confidence_scores': result.confidence_scores,
    'processing_time': result.processing_time,
    'image_path': result.image_path,
    'timestamp': result.timestamp
}, indent=2)}"""
                
                self.root.after(0, lambda: self.update_single_results(results_text))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
            finally:
                self.root.after(0, lambda: self.progress_single.stop())
                self.root.after(0, lambda: self.analyze_btn.config(state='normal'))
        
        threading.Thread(target=analyze, daemon=True).start()
    
    def update_single_results(self, results_text):
        """Update the results display"""
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert('1.0', results_text)
        self.save_single_btn.config(state='normal')
    
    def save_single_results(self):
        """Save single image results to file"""
        if not self.current_results:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                result_dict = {
                    "people_count": self.current_results.people_count,
                    "vehicle_count": self.current_results.vehicle_count,
                    "traffic_lights": self.current_results.traffic_lights,
                    "confidence_scores": self.current_results.confidence_scores,
                    "processing_time": self.current_results.processing_time,
                    "image_path": self.current_results.image_path,
                    "timestamp": self.current_results.timestamp
                }
                
                with open(file_path, 'w') as f:
                    json.dump(result_dict, f, indent=2)
                
                messagebox.showinfo("Success", f"Results saved to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not save results: {str(e)}")
    
    def start_webcam_analysis(self):
        """Start webcam analysis"""
        if self.is_video_running:
            return
        
        def webcam_thread():
            try:
                if not self.video_analyzer:
                    self.video_analyzer = VideoAnalyzer(
                        confidence_threshold=0.5,
                        fps_limit=self.fps_var.get()
                    )
                
                self.video_analyzer.set_callbacks(
                    frame_callback=self.update_video_display,
                    results_callback=self.update_video_results
                )
                
                self.is_video_running = True
                self.root.after(0, lambda: self.stop_btn.config(state='normal'))
                self.root.after(0, lambda: self.webcam_btn.config(state='disabled'))
                self.root.after(0, lambda: self.save_video_btn.config(state='normal'))
                
                self.video_analyzer.analyze_webcam(camera_index=0, display_window=False)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Webcam Error", f"Could not start webcam: {str(e)}"))
            finally:
                self.is_video_running = False
                self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
                self.root.after(0, lambda: self.webcam_btn.config(state='normal'))
        
        self.video_thread = threading.Thread(target=webcam_thread, daemon=True)
        self.video_thread.start()
    
    def select_video_file(self):
        """Select and analyze video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            def video_thread():
                try:
                    if not self.video_analyzer:
                        self.video_analyzer = VideoAnalyzer(
                            confidence_threshold=0.5,
                            fps_limit=self.fps_var.get()
                        )
                    
                    self.video_analyzer.set_callbacks(
                        frame_callback=self.update_video_display,
                        results_callback=self.update_video_results
                    )
                    
                    self.is_video_running = True
                    self.root.after(0, lambda: self.stop_btn.config(state='normal'))
                    self.root.after(0, lambda: self.video_file_btn.config(state='disabled'))
                    self.root.after(0, lambda: self.save_video_btn.config(state='normal'))
                    
                    self.video_analyzer.analyze_video_file(
                        video_path=file_path,
                        display_window=False
                    )
                    
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Video Error", f"Could not analyze video: {str(e)}"))
                finally:
                    self.is_video_running = False
                    self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
                    self.root.after(0, lambda: self.video_file_btn.config(state='normal'))
            
            self.video_thread = threading.Thread(target=video_thread, daemon=True)
            self.video_thread.start()
    
    def start_rtsp_analysis(self):
        """Start RTSP stream analysis"""
        if self.is_video_running:
            return
        
        rtsp_url = self.rtsp_entry.get().strip()
        if not rtsp_url or rtsp_url == "rtsp://example.com/stream":
            messagebox.showwarning("RTSP URL", "Please enter a valid RTSP URL")
            return
        
        def rtsp_thread():
            try:
                if not self.video_analyzer:
                    self.video_analyzer = VideoAnalyzer(
                        confidence_threshold=0.5,
                        fps_limit=self.fps_var.get()
                    )
                
                self.video_analyzer.set_callbacks(
                    frame_callback=self.update_video_display,
                    results_callback=self.update_video_results
                )
                
                self.is_video_running = True
                self.root.after(0, lambda: self.stop_btn.config(state='normal'))
                self.root.after(0, lambda: self.rtsp_btn.config(state='disabled'))
                self.root.after(0, lambda: self.save_video_btn.config(state='normal'))
                
                self.video_analyzer.analyze_rtsp_stream(rtsp_url, display_window=False)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("RTSP Error", f"Could not connect to RTSP stream: {str(e)}"))
            finally:
                self.is_video_running = False
                self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
                self.root.after(0, lambda: self.rtsp_btn.config(state='normal'))
        
        self.video_thread = threading.Thread(target=rtsp_thread, daemon=True)
        self.video_thread.start()
    
    def stop_video_analysis(self):
        """Stop video analysis"""
        if self.video_analyzer:
            self.video_analyzer.stop_processing()
        
        self.is_video_running = False
        self.stop_btn.config(state='disabled')
        self.webcam_btn.config(state='normal')
        self.video_file_btn.config(state='normal')
        self.rtsp_btn.config(state='normal')
    
    def update_video_display(self, frame):
        """Update video display with current frame"""
        def update_display():
            try:
                display_height = 400
                height, width = frame.shape[:2]
                aspect_ratio = width / height
                display_width = int(display_height * aspect_ratio)
                
                resized_frame = cv2.resize(frame, (display_width, display_height))
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(pil_image)
                
                self.video_label.config(image=photo, text="")
                self.video_label.image = photo
                
            except Exception as e:
                print(f"Error updating video display: {str(e)}")
        
        self.root.after(0, update_display)
    
    def update_video_results(self, result):
        """Update video results display in real-time"""
        def update_ui():
            result_text = f"""Frame {result.frame_number} - {time.strftime('%H:%M:%S')}
👥 People: {result.people_count}
🚗 Vehicles: {result.vehicle_count}
🚦 Traffic Lights: {result.traffic_lights['total']} (R:{result.traffic_lights['red']} G:{result.traffic_lights['green']} Y:{result.traffic_lights['yellow']})
⚡ Processing: {result.processing_time:.3f}s
📊 Confidence: P:{result.confidence_scores['people']:.2f} V:{result.confidence_scores['vehicles']:.2f} T:{result.confidence_scores['traffic_lights']:.2f}

"""
            
            self.video_results_text.insert('1.0', result_text)
            
            content = self.video_results_text.get('1.0', tk.END)
            lines = content.split('\n')
            if len(lines) > 300:
                truncated = '\n'.join(lines[:300])
                self.video_results_text.delete('1.0', tk.END)
                self.video_results_text.insert('1.0', truncated)
        
        self.root.after(0, update_ui)
    
    def save_video_results(self):
        """Save video analysis results"""
        if not self.video_analyzer or not self.video_analyzer.results_history:
            messagebox.showwarning("No Results", "No video analysis results to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Video Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.video_analyzer.save_results_to_file(file_path)
                
                stats = self.video_analyzer.get_statistics()
                stats_msg = f"""Video Analysis Results Saved!

Statistics:
• Frames Processed: {stats.get('total_frames_processed', 0)}
• Average People: {stats.get('average_people', 0):.1f}
• Max People: {stats.get('max_people', 0)}
• Average Vehicles: {stats.get('average_vehicles', 0):.1f}
• Max Vehicles: {stats.get('max_vehicles', 0)}
• Average FPS: {stats.get('average_fps', 0):.1f}"""
                
                messagebox.showinfo("Results Saved", stats_msg)
                
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save results: {str(e)}")
    
    def apply_settings(self):
        """Apply new settings"""
        try:
            confidence = self.confidence_var.get()
            
            self.analyzer = ImageAnalyzer(confidence_threshold=confidence)
            self.batch_processor = BatchProcessor(confidence_threshold=confidence, max_workers=4)
            if self.video_analyzer:
                self.video_analyzer = VideoAnalyzer(confidence_threshold=confidence, fps_limit=5)
            
            messagebox.showinfo("Success", f"Settings applied:\nConfidence: {confidence}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not apply settings: {str(e)}")

def main():
    """Launch the GUI application"""
    root = tk.Tk()
    app = ImageAnalysisGUI(root)
    root.mainloop()

    def initialize_analyzer(self):
        """Initialize the analyzer in a separate thread"""
        def init():
            try:
                self.analyzer = ImageAnalyzer(confidence_threshold=0.5)
                self.batch_processor = BatchProcessor(confidence_threshold=0.5, max_workers=4)
                self.video_analyzer = VideoAnalyzer(confidence_threshold=0.5, fps_limit=5)
                print("✅ All analyzers initialized successfully!")
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Failed to initialize analyzer: {msg}"))
        
        threading.Thread(target=init, daemon=True).start()
    
    def select_image(self):
        """Select an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.file_path_label.config(text=f"Selected: {os.path.basename(file_path)}")
            self.analyze_btn.config(state='normal')
            self.display_image(file_path)
    
    def add_photo_from_files(self):
        """Add photo from anywhere on the computer"""
        file_path = filedialog.askopenfilename(
            title="Add Photo from Files",
            initialdir=os.path.expanduser("~"),
            filetypes=[
                ("JPEG Images", "*.jpg *.jpeg"),
                ("PNG Images", "*.png"),
                ("BMP Images", "*.bmp"),
                ("TIFF Images", "*.tiff *.tif"),
                ("WebP Images", "*.webp"),
                ("All Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            try:
                test_image = Image.open(file_path)
                test_image.close()
                
                self.current_image_path = file_path
                self.file_path_label.config(
                    text=f"📷 Photo Added: {os.path.basename(file_path)}", 
                    fg=self.colors['accent_primary']
                )
                self.analyze_btn.config(state='normal')
                self.display_image(file_path)
                
                messagebox.showinfo(
                    "Photo Added", 
                    f"Successfully added photo:\n{os.path.basename(file_path)}\n\nReady for analysis!"
                )
                
            except Exception as e:
                error_msg = str(e)
                messagebox.showerror(
                    "Invalid Image", 
                    f"Could not load the selected file as an image.\n\nError: {error_msg}\n\nPlease select a valid image file."
                )
    
    def display_image(self, image_path):
        """Display the selected image"""
        try:
            image = Image.open(image_path)
            display_size = (500, 400)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            
        except Exception as e:
            error_msg = str(e)
            messagebox.showerror("Error", f"Could not display image: {error_msg}")
    
    def analyze_single_image(self):
        """Analyze the selected image"""
        if not self.current_image_path or not self.analyzer:
            return
        
        def analyze():
            try:
                self.root.after(0, lambda: self.progress_single.start())
                self.root.after(0, lambda: self.analyze_btn.config(state='disabled'))
                
                result = self.analyzer.analyze_image(self.current_image_path)
                self.current_results = result
                
                results_text = f"""Analysis Results for: {os.path.basename(result.image_path)}
{'='*50}

👥 PEOPLE DETECTED: {result.people_count}
🚗 VEHICLES DETECTED: {result.vehicle_count}

🚦 TRAFFIC LIGHTS:
   Total: {result.traffic_lights['total']}
   Red: {result.traffic_lights['red']}
   Green: {result.traffic_lights['green']}
   Yellow: {result.traffic_lights['yellow']}

📊 CONFIDENCE SCORES:
   People: {result.confidence_scores['people']:.3f}
   Vehicles: {result.confidence_scores['vehicles']:.3f}
   Traffic Lights: {result.confidence_scores['traffic_lights']:.3f}

⏱️ Processing Time: {result.processing_time:.3f} seconds
📅 Timestamp: {result.timestamp}

JSON Output:
{json.dumps({
    'people_count': result.people_count,
    'vehicle_count': result.vehicle_count,
    'traffic_lights': result.traffic_lights,
    'confidence_scores': result.confidence_scores,
    'processing_time': result.processing_time,
    'image_path': result.image_path,
    'timestamp': result.timestamp
}, indent=2)}"""
                
                self.root.after(0, lambda text=results_text: self.update_single_results(text))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Analysis failed: {msg}"))
            finally:
                self.root.after(0, lambda: self.progress_single.stop())
                self.root.after(0, lambda: self.analyze_btn.config(state='normal'))
        
        threading.Thread(target=analyze, daemon=True).start()
    
    def update_single_results(self, results_text):
        """Update the results display"""
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert('1.0', results_text)
        self.save_single_btn.config(state='normal')
    
    def save_single_results(self):
        """Save single image results to file"""
        if not self.current_results:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                result_dict = {
                    "people_count": self.current_results.people_count,
                    "vehicle_count": self.current_results.vehicle_count,
                    "traffic_lights": self.current_results.traffic_lights,
                    "confidence_scores": self.current_results.confidence_scores,
                    "processing_time": self.current_results.processing_time,
                    "image_path": self.current_results.image_path,
                    "timestamp": self.current_results.timestamp
                }
                
                with open(file_path, 'w') as f:
                    json.dump(result_dict, f, indent=2)
                
                messagebox.showinfo("Success", f"Results saved to {file_path}")
                
            except Exception as e:
                error_msg = str(e)
                messagebox.showerror("Error", f"Could not save results: {error_msg}")
    
    def select_directory(self):
        """Select directory for batch processing"""
        dir_path = filedialog.askdirectory(title="Select Directory with Images")
        
        if dir_path:
            self.current_dir_path = dir_path
            self.dir_path_label.config(text=f"Selected: {dir_path}")
            self.process_btn.config(state='normal')
    
    def start_batch_processing(self):
        """Start batch processing"""
        if not hasattr(self, 'current_dir_path') or not self.batch_processor:
            return
        
        def process():
            try:
                self.root.after(0, lambda: self.process_btn.config(state='disabled'))
                self.root.after(0, lambda: self.progress_label.config(text="Finding images..."))
                
                # Find images
                image_paths = self.batch_processor.find_images(self.current_dir_path)
                
                if not image_paths:
                    self.root.after(0, lambda: messagebox.showinfo("Info", "No images found in selected directory"))
                    return
                
                self.root.after(0, lambda count=len(image_paths): self.progress_batch.config(maximum=count))
                self.root.after(0, lambda count=len(image_paths): self.progress_label.config(text=f"Processing {count} images..."))
                
                # Process images
                results = self.batch_processor.process_directory(
                    input_dir=self.current_dir_path,
                    parallel=self.parallel_var.get()
                )
                
                # Format results
                total_people = sum(r.people_count for r in results)
                total_vehicles = sum(r.vehicle_count for r in results)
                total_lights = sum(r.traffic_lights['total'] for r in results)
                avg_time = sum(r.processing_time for r in results) / len(results) if results else 0
                
                summary = f"""Batch Processing Results
{'='*50}

📊 SUMMARY:
   Images Processed: {len(results)}
   Total People: {total_people}
   Total Vehicles: {total_vehicles}
   Total Traffic Lights: {total_lights}
   Average Processing Time: {avg_time:.3f}s

📋 DETAILED RESULTS:
"""
                
                for i, result in enumerate(results, 1):
                    summary += f"\n{i}. {os.path.basename(result.image_path)}:\n"
                    summary += f"   People: {result.people_count}, Vehicles: {result.vehicle_count}, "
                    summary += f"Traffic Lights: {result.traffic_lights['total']}\n"
                
                self.batch_results = results
                self.root.after(0, lambda text=summary: self.update_batch_results(text))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Batch processing failed: {msg}"))
            finally:
                self.root.after(0, lambda: self.process_btn.config(state='normal'))
                self.root.after(0, lambda: self.progress_label.config(text="Processing complete"))
        
        threading.Thread(target=process, daemon=True).start()
    
    def update_batch_results(self, results_text):
        """Update batch results display"""
        self.batch_results_text.delete('1.0', tk.END)
        self.batch_results_text.insert('1.0', results_text)
        self.save_batch_btn.config(state='normal')
    
    def save_batch_results(self):
        """Save batch results to file"""
        if not hasattr(self, 'batch_results'):
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Batch Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.batch_processor.save_results_with_summary(self.batch_results, file_path)
                messagebox.showinfo("Success", f"Batch results saved to {file_path}")
            except Exception as e:
                error_msg = str(e)
                messagebox.showerror("Error", f"Could not save results: {error_msg}")
    
    def start_webcam_analysis(self):
        """Start webcam analysis"""
        if self.is_video_running:
            return
        
        def webcam_thread():
            try:
                if not self.video_analyzer:
                    self.video_analyzer = VideoAnalyzer(
                        confidence_threshold=0.5,
                        fps_limit=self.fps_var.get()
                    )
                
                self.video_analyzer.set_callbacks(
                    frame_callback=self.update_video_display,
                    results_callback=self.update_video_results
                )
                
                self.is_video_running = True
                self.root.after(0, lambda: self.stop_btn.config(state='normal'))
                self.root.after(0, lambda: self.webcam_btn.config(state='disabled'))
                self.root.after(0, lambda: self.save_video_btn.config(state='normal'))
                
                self.video_analyzer.analyze_webcam(camera_index=0, display_window=False)
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Webcam Error", f"Could not start webcam: {msg}"))
            finally:
                self.is_video_running = False
                self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
                self.root.after(0, lambda: self.webcam_btn.config(state='normal'))
        
        self.video_thread = threading.Thread(target=webcam_thread, daemon=True)
        self.video_thread.start()
    
    def select_video_file(self):
        """Select and analyze video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            def video_thread():
                try:
                    if not self.video_analyzer:
                        self.video_analyzer = VideoAnalyzer(
                            confidence_threshold=0.5,
                            fps_limit=self.fps_var.get()
                        )
                    
                    self.video_analyzer.set_callbacks(
                        frame_callback=self.update_video_display,
                        results_callback=self.update_video_results
                    )
                    
                    self.is_video_running = True
                    self.root.after(0, lambda: self.stop_btn.config(state='normal'))
                    self.root.after(0, lambda: self.video_file_btn.config(state='disabled'))
                    self.root.after(0, lambda: self.save_video_btn.config(state='normal'))
                    
                    self.video_analyzer.analyze_video_file(
                        video_path=file_path,
                        display_window=False
                    )
                    
                except Exception as e:
                    error_msg = str(e)
                    self.root.after(0, lambda msg=error_msg: messagebox.showerror("Video Error", f"Could not analyze video: {msg}"))
                finally:
                    self.is_video_running = False
                    self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
                    self.root.after(0, lambda: self.video_file_btn.config(state='normal'))
            
            self.video_thread = threading.Thread(target=video_thread, daemon=True)
            self.video_thread.start()
    
    def start_rtsp_analysis(self):
        """Start RTSP stream analysis"""
        if self.is_video_running:
            return
        
        rtsp_url = self.rtsp_entry.get().strip()
        if not rtsp_url or rtsp_url == "rtsp://example.com/stream":
            messagebox.showwarning("RTSP URL", "Please enter a valid RTSP URL")
            return
        
        def rtsp_thread():
            try:
                if not self.video_analyzer:
                    self.video_analyzer = VideoAnalyzer(
                        confidence_threshold=0.5,
                        fps_limit=self.fps_var.get()
                    )
                
                self.video_analyzer.set_callbacks(
                    frame_callback=self.update_video_display,
                    results_callback=self.update_video_results
                )
                
                self.is_video_running = True
                self.root.after(0, lambda: self.stop_btn.config(state='normal'))
                self.root.after(0, lambda: self.rtsp_btn.config(state='disabled'))
                self.root.after(0, lambda: self.save_video_btn.config(state='normal'))
                
                self.video_analyzer.analyze_rtsp_stream(rtsp_url, display_window=False)
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("RTSP Error", f"Could not connect to RTSP stream: {msg}"))
            finally:
                self.is_video_running = False
                self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
                self.root.after(0, lambda: self.rtsp_btn.config(state='normal'))
        
        self.video_thread = threading.Thread(target=rtsp_thread, daemon=True)
        self.video_thread.start()
    
    def stop_video_analysis(self):
        """Stop video analysis"""
        if self.video_analyzer:
            self.video_analyzer.stop_processing()
        
        self.is_video_running = False
        self.stop_btn.config(state='disabled')
        self.webcam_btn.config(state='normal')
        self.video_file_btn.config(state='normal')
        self.rtsp_btn.config(state='normal')
    
    def update_video_display(self, frame):
        """Update video display with current frame"""
        def update_display():
            try:
                display_height = 400
                height, width = frame.shape[:2]
                aspect_ratio = width / height
                display_width = int(display_height * aspect_ratio)
                
                resized_frame = cv2.resize(frame, (display_width, display_height))
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(pil_image)
                
                self.video_label.config(image=photo, text="")
                self.video_label.image = photo
                
            except Exception as e:
                print(f"Error updating video display: {str(e)}")
        
        self.root.after(0, update_display)
    
    def update_video_results(self, result):
        """Update video results display in real-time"""
        def update_ui():
            result_text = f"""Frame {result.frame_number} - {time.strftime('%H:%M:%S')}
👥 People: {result.people_count}
🚗 Vehicles: {result.vehicle_count}
🚦 Traffic Lights: {result.traffic_lights['total']} (R:{result.traffic_lights['red']} G:{result.traffic_lights['green']} Y:{result.traffic_lights['yellow']})
⚡ Processing: {result.processing_time:.3f}s
📊 Confidence: P:{result.confidence_scores['people']:.2f} V:{result.confidence_scores['vehicles']:.2f} T:{result.traffic_lights['total']:.2f}

"""
            
            self.video_results_text.insert('1.0', result_text)
            
            content = self.video_results_text.get('1.0', tk.END)
            lines = content.split('\n')
            if len(lines) > 300:
                truncated = '\n'.join(lines[:300])
                self.video_results_text.delete('1.0', tk.END)
                self.video_results_text.insert('1.0', truncated)
        
        self.root.after(0, update_ui)
    
    def save_video_results(self):
        """Save video analysis results"""
        if not self.video_analyzer or not self.video_analyzer.results_history:
            messagebox.showwarning("No Results", "No video analysis results to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Video Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.video_analyzer.save_results_to_file(file_path)
                
                stats = self.video_analyzer.get_statistics()
                stats_msg = f"""Video Analysis Results Saved!

Statistics:
• Frames Processed: {stats.get('total_frames_processed', 0)}
• Average People: {stats.get('average_people', 0):.1f}
• Max People: {stats.get('max_people', 0)}
• Average Vehicles: {stats.get('average_vehicles', 0):.1f}
• Max Vehicles: {stats.get('max_vehicles', 0)}
• Average FPS: {stats.get('average_fps', 0):.1f}"""
                
                messagebox.showinfo("Results Saved", stats_msg)
                
            except Exception as e:
                error_msg = str(e)
                messagebox.showerror("Save Error", f"Could not save results: {error_msg}")
    
    def apply_settings(self):
        """Apply new settings"""
        try:
            confidence = self.confidence_var.get()
            
            self.analyzer = ImageAnalyzer(confidence_threshold=confidence)
            self.batch_processor = BatchProcessor(confidence_threshold=confidence, max_workers=4)
            if self.video_analyzer:
                self.video_analyzer = VideoAnalyzer(confidence_threshold=confidence, fps_limit=5)
            
            messagebox.showinfo("Success", f"Settings applied:\nConfidence: {confidence}")
            
        except Exception as e:
            error_msg = str(e)
            messagebox.showerror("Error", f"Could not apply settings: {error_msg}")

def main():
    """Launch the GUI application"""
    root = tk.Tk()
    app = ImageAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
        """Initialize the analyzer in a separate thread"""
        def init():
            try:
                self.analyzer = ImageAnalyzer(confidence_threshold=0.5)
                self.batch_processor = BatchProcessor(confidence_threshold=0.5, max_workers=4)
                self.video_analyzer = VideoAnalyzer(confidence_threshold=0.5, fps_limit=5)
                print("✅ All analyzers initialized successfully!")
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Failed to initialize analyzer: {msg}"))
        
        threading.Thread(target=init, daemon=True).start()
    
    def select_image(self):
        """Select an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.file_path_label.config(text=f"Selected: {os.path.basename(file_path)}")
            self.analyze_btn.config(state='normal')
            self.display_image(file_path)
    
    def add_photo_from_files(self):
        """Add photo from anywhere on the computer"""
        file_path = filedialog.askopenfilename(
            title="Add Photo from Files",
            initialdir=os.path.expanduser("~"),
            filetypes=[
                ("JPEG Images", "*.jpg *.jpeg"),
                ("PNG Images", "*.png"),
                ("BMP Images", "*.bmp"),
                ("TIFF Images", "*.tiff *.tif"),
                ("WebP Images", "*.webp"),
                ("All Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            try:
                test_image = Image.open(file_path)
                test_image.close()
                
                self.current_image_path = file_path
                self.file_path_label.config(
                    text=f"📷 Photo Added: {os.path.basename(file_path)}", 
                    fg=self.colors['accent_primary']
                )
                self.analyze_btn.config(state='normal')
                self.display_image(file_path)
                
                messagebox.showinfo(
                    "Photo Added", 
                    f"Successfully added photo:\n{os.path.basename(file_path)}\n\nReady for analysis!"
                )
                
            except Exception as e:
                error_msg = str(e)
                messagebox.showerror(
                    "Invalid Image", 
                    f"Could not load the selected file as an image.\n\nError: {error_msg}\n\nPlease select a valid image file."
                )
    
    def display_image(self, image_path):
        """Display the selected image"""
        try:
            image = Image.open(image_path)
            display_size = (500, 400)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            
        except Exception as e:
            error_msg = str(e)
            messagebox.showerror("Error", f"Could not display image: {error_msg}")
    
    def analyze_single_image(self):
        """Analyze the selected image"""
        if not self.current_image_path or not self.analyzer:
            return
        
        def analyze():
            try:
                self.root.after(0, lambda: self.progress_single.start())
                self.root.after(0, lambda: self.analyze_btn.config(state='disabled'))
                
                result = self.analyzer.analyze_image(self.current_image_path)
                self.current_results = result
                
                results_text = f"""Analysis Results for: {os.path.basename(result.image_path)}
{'='*50}

👥 PEOPLE DETECTED: {result.people_count}
🚗 VEHICLES DETECTED: {result.vehicle_count}

🚦 TRAFFIC LIGHTS:
   Total: {result.traffic_lights['total']}
   Red: {result.traffic_lights['red']}
   Green: {result.traffic_lights['green']}
   Yellow: {result.traffic_lights['yellow']}

📊 CONFIDENCE SCORES:
   People: {result.confidence_scores['people']:.3f}
   Vehicles: {result.confidence_scores['vehicles']:.3f}
   Traffic Lights: {result.confidence_scores['traffic_lights']:.3f}

⏱️ Processing Time: {result.processing_time:.3f} seconds
📅 Timestamp: {result.timestamp}

JSON Output:
{json.dumps({
    'people_count': result.people_count,
    'vehicle_count': result.vehicle_count,
    'traffic_lights': result.traffic_lights,
    'confidence_scores': result.confidence_scores,
    'processing_time': result.processing_time,
    'image_path': result.image_path,
    'timestamp': result.timestamp
}, indent=2)}"""
                
                self.root.after(0, lambda text=results_text: self.update_single_results(text))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Analysis failed: {msg}"))
            finally:
                self.root.after(0, lambda: self.progress_single.stop())
                self.root.after(0, lambda: self.analyze_btn.config(state='normal'))
        
        threading.Thread(target=analyze, daemon=True).start()
    
    def update_single_results(self, results_text):
        """Update the results display"""
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert('1.0', results_text)
        self.save_single_btn.config(state='normal')
    
    def save_single_results(self):
        """Save single image results to file"""
        if not self.current_results:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                result_dict = {
                    "people_count": self.current_results.people_count,
                    "vehicle_count": self.current_results.vehicle_count,
                    "traffic_lights": self.current_results.traffic_lights,
                    "confidence_scores": self.current_results.confidence_scores,
                    "processing_time": self.current_results.processing_time,
                    "image_path": self.current_results.image_path,
                    "timestamp": self.current_results.timestamp
                }
                
                with open(file_path, 'w') as f:
                    json.dump(result_dict, f, indent=2)
                
                messagebox.showinfo("Success", f"Results saved to {file_path}")
                
            except Exception as e:
                error_msg = str(e)
                messagebox.showerror("Error", f"Could not save results: {error_msg}")
    
    def select_directory(self):
        """Select directory for batch processing"""
        dir_path = filedialog.askdirectory(title="Select Directory with Images")
        
        if dir_path:
            self.current_dir_path = dir_path
            self.dir_path_label.config(text=f"Selected: {dir_path}")
            self.process_btn.config(state='normal')
    
    def start_batch_processing(self):
        """Start batch processing"""
        if not hasattr(self, 'current_dir_path') or not self.batch_processor:
            return
        
        def process():
            try:
                self.root.after(0, lambda: self.process_btn.config(state='disabled'))
                self.root.after(0, lambda: self.progress_label.config(text="Finding images..."))
                
                # Find images
                image_paths = self.batch_processor.find_images(self.current_dir_path)
                
                if not image_paths:
                    self.root.after(0, lambda: messagebox.showinfo("Info", "No images found in selected directory"))
                    return
                
                self.root.after(0, lambda count=len(image_paths): self.progress_batch.config(maximum=count))
                self.root.after(0, lambda count=len(image_paths): self.progress_label.config(text=f"Processing {count} images..."))
                
                # Process images
                results = self.batch_processor.process_directory(
                    input_dir=self.current_dir_path,
                    parallel=self.parallel_var.get()
                )
                
                # Format results
                total_people = sum(r.people_count for r in results)
                total_vehicles = sum(r.vehicle_count for r in results)
                total_lights = sum(r.traffic_lights['total'] for r in results)
                avg_time = sum(r.processing_time for r in results) / len(results) if results else 0
                
                summary = f"""Batch Processing Results
{'='*50}

📊 SUMMARY:
   Images Processed: {len(results)}
   Total People: {total_people}
   Total Vehicles: {total_vehicles}
   Total Traffic Lights: {total_lights}
   Average Processing Time: {avg_time:.3f}s

📋 DETAILED RESULTS:
"""
                
                for i, result in enumerate(results, 1):
                    summary += f"\n{i}. {os.path.basename(result.image_path)}:\n"
                    summary += f"   People: {result.people_count}, Vehicles: {result.vehicle_count}, "
                    summary += f"Traffic Lights: {result.traffic_lights['total']}\n"
                
                self.batch_results = results
                self.root.after(0, lambda text=summary: self.update_batch_results(text))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Batch processing failed: {msg}"))
            finally:
                self.root.after(0, lambda: self.process_btn.config(state='normal'))
                self.root.after(0, lambda: self.progress_label.config(text="Processing complete"))
        
        threading.Thread(target=process, daemon=True).start()
    
    def update_batch_results(self, results_text):
        """Update batch results display"""
        self.batch_results_text.delete('1.0', tk.END)
        self.batch_results_text.insert('1.0', results_text)
        self.save_batch_btn.config(state='normal')
    
    def save_batch_results(self):
        """Save batch results to file"""
        if not hasattr(self, 'batch_results'):
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Batch Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.batch_processor.save_results_with_summary(self.batch_results, file_path)
                messagebox.showinfo("Success", f"Batch results saved to {file_path}")
            except Exception as e:
                error_msg = str(e)
                messagebox.showerror("Error", f"Could not save results: {error_msg}")
    
    def start_webcam_analysis(self):
        """Start webcam analysis"""
        if self.is_video_running:
            return
        
        def webcam_thread():
            try:
                if not self.video_analyzer:
                    self.video_analyzer = VideoAnalyzer(
                        confidence_threshold=0.5,
                        fps_limit=self.fps_var.get()
                    )
                
                self.video_analyzer.set_callbacks(
                    frame_callback=self.update_video_display,
                    results_callback=self.update_video_results
                )
                
                self.is_video_running = True
                self.root.after(0, lambda: self.stop_btn.config(state='normal'))
                self.root.after(0, lambda: self.webcam_btn.config(state='disabled'))
                self.root.after(0, lambda: self.save_video_btn.config(state='normal'))
                
                self.video_analyzer.analyze_webcam(camera_index=0, display_window=False)
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Webcam Error", f"Could not start webcam: {msg}"))
            finally:
                self.is_video_running = False
                self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
                self.root.after(0, lambda: self.webcam_btn.config(state='normal'))
        
        self.video_thread = threading.Thread(target=webcam_thread, daemon=True)
        self.video_thread.start()
    
    def select_video_file(self):
        """Select and analyze video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            def video_thread():
                try:
                    if not self.video_analyzer:
                        self.video_analyzer = VideoAnalyzer(
                            confidence_threshold=0.5,
                            fps_limit=self.fps_var.get()
                        )
                    
                    self.video_analyzer.set_callbacks(
                        frame_callback=self.update_video_display,
                        results_callback=self.update_video_results
                    )
                    
                    self.is_video_running = True
                    self.root.after(0, lambda: self.stop_btn.config(state='normal'))
                    self.root.after(0, lambda: self.video_file_btn.config(state='disabled'))
                    self.root.after(0, lambda: self.save_video_btn.config(state='normal'))
                    
                    self.video_analyzer.analyze_video_file(
                        video_path=file_path,
                        display_window=False
                    )
                    
                except Exception as e:
                    error_msg = str(e)
                    self.root.after(0, lambda msg=error_msg: messagebox.showerror("Video Error", f"Could not analyze video: {msg}"))
                finally:
                    self.is_video_running = False
                    self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
                    self.root.after(0, lambda: self.video_file_btn.config(state='normal'))
            
            self.video_thread = threading.Thread(target=video_thread, daemon=True)
            self.video_thread.start()
    
    def start_rtsp_analysis(self):
        """Start RTSP stream analysis"""
        if self.is_video_running:
            return
        
        rtsp_url = self.rtsp_entry.get().strip()
        if not rtsp_url or rtsp_url == "rtsp://example.com/stream":
            messagebox.showwarning("RTSP URL", "Please enter a valid RTSP URL")
            return
        
        def rtsp_thread():
            try:
                if not self.video_analyzer:
                    self.video_analyzer = VideoAnalyzer(
                        confidence_threshold=0.5,
                        fps_limit=self.fps_var.get()
                    )
                
                self.video_analyzer.set_callbacks(
                    frame_callback=self.update_video_display,
                    results_callback=self.update_video_results
                )
                
                self.is_video_running = True
                self.root.after(0, lambda: self.stop_btn.config(state='normal'))
                self.root.after(0, lambda: self.rtsp_btn.config(state='disabled'))
                self.root.after(0, lambda: self.save_video_btn.config(state='normal'))
                
                self.video_analyzer.analyze_rtsp_stream(rtsp_url, display_window=False)
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("RTSP Error", f"Could not connect to RTSP stream: {msg}"))
            finally:
                self.is_video_running = False
                self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
                self.root.after(0, lambda: self.rtsp_btn.config(state='normal'))
        
        self.video_thread = threading.Thread(target=rtsp_thread, daemon=True)
        self.video_thread.start()
    
    def stop_video_analysis(self):
        """Stop video analysis"""
        if self.video_analyzer:
            self.video_analyzer.stop_processing()
        
        self.is_video_running = False
        self.stop_btn.config(state='disabled')
        self.webcam_btn.config(state='normal')
        self.video_file_btn.config(state='normal')
        self.rtsp_btn.config(state='normal')
    
    def update_video_display(self, frame):
        """Update video display with current frame"""
        def update_display():
            try:
                display_height = 400
                height, width = frame.shape[:2]
                aspect_ratio = width / height
                display_width = int(display_height * aspect_ratio)
                
                resized_frame = cv2.resize(frame, (display_width, display_height))
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(pil_image)
                
                self.video_label.config(image=photo, text="")
                self.video_label.image = photo
                
            except Exception as e:
                print(f"Error updating video display: {str(e)}")
        
        self.root.after(0, update_display)
    
    def update_video_results(self, result):
        """Update video results display in real-time"""
        def update_ui():
            result_text = f"""Frame {result.frame_number} - {time.strftime('%H:%M:%S')}
👥 People: {result.people_count}
🚗 Vehicles: {result.vehicle_count}
🚦 Traffic Lights: {result.traffic_lights['total']} (R:{result.traffic_lights['red']} G:{result.traffic_lights['green']} Y:{result.traffic_lights['yellow']})
⚡ Processing: {result.processing_time:.3f}s
📊 Confidence: P:{result.confidence_scores['people']:.2f} V:{result.confidence_scores['vehicles']:.2f} T:{result.confidence_scores['traffic_lights']:.2f}

"""
            
            self.video_results_text.insert('1.0', result_text)
            
            content = self.video_results_text.get('1.0', tk.END)
            lines = content.split('\n')
            if len(lines) > 300:
                truncated = '\n'.join(lines[:300])
                self.video_results_text.delete('1.0', tk.END)
                self.video_results_text.insert('1.0', truncated)
        
        self.root.after(0, update_ui)
    
    def save_video_results(self):
        """Save video analysis results"""
        if not self.video_analyzer or not self.video_analyzer.results_history:
            messagebox.showwarning("No Results", "No video analysis results to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Video Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.video_analyzer.save_results_to_file(file_path)
                
                stats = self.video_analyzer.get_statistics()
                stats_msg = f"""Video Analysis Results Saved!

Statistics:
• Frames Processed: {stats.get('total_frames_processed', 0)}
• Average People: {stats.get('average_people', 0):.1f}
• Max People: {stats.get('max_people', 0)}
• Average Vehicles: {stats.get('average_vehicles', 0):.1f}
• Max Vehicles: {stats.get('max_vehicles', 0)}
• Average FPS: {stats.get('average_fps', 0):.1f}"""
                
                messagebox.showinfo("Results Saved", stats_msg)
                
            except Exception as e:
                error_msg = str(e)
                messagebox.showerror("Save Error", f"Could not save results: {error_msg}")
    
    def apply_settings(self):
        """Apply new settings"""
        try:
            confidence = self.confidence_var.get()
            
            self.analyzer = ImageAnalyzer(confidence_threshold=confidence)
            self.batch_processor = BatchProcessor(confidence_threshold=confidence, max_workers=4)
            if self.video_analyzer:
                self.video_analyzer = VideoAnalyzer(confidence_threshold=confidence, fps_limit=5)
            
            messagebox.showinfo("Success", f"Settings applied:\nConfidence: {confidence}")
            
        except Exception as e:
            error_msg = str(e)
            messagebox.showerror("Error", f"Could not apply settings: {error_msg}")

def main():
    """Launch the GUI application"""
    root = tk.Tk()
    app = ImageAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()