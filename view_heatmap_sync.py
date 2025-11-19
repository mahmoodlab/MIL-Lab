"""
Synchronized Heatmap Viewer for OpenSlide

This script allows you to view the original WSI and its attention heatmap
side-by-side in sync, so you can explore the same regions simultaneously.

Usage:
    python view_heatmap_sync.py --slide path/to/slide.svs --heatmap path/to/heatmap.png

Requirements:
    - openslide-python
    - Pillow
    - matplotlib (for visualization)
"""

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    print("Error: openslide-python not installed")
    print("Install with: pip install openslide-python")
    exit(1)


class SynchronizedViewer:
    """Interactive viewer for WSI and heatmap overlay"""

    def __init__(self, slide_path, heatmap_path=None, initial_level=2):
        """
        Initialize viewer

        Args:
            slide_path: Path to WSI file
            heatmap_path: Path to heatmap image (optional)
            initial_level: Initial pyramid level to display
        """
        # Load slide
        self.slide = openslide.open_slide(slide_path)
        self.slide_name = os.path.basename(slide_path)

        # Get slide properties
        self.level_count = self.slide.level_count
        self.level_dimensions = self.slide.level_dimensions
        self.level_downsamples = self.slide.level_downsamples

        print(f"\nSlide: {self.slide_name}")
        print(f"Dimensions (level 0): {self.level_dimensions[0]}")
        print(f"Number of levels: {self.level_count}")
        print(f"Level dimensions: {self.level_dimensions}")

        # Load heatmap if provided
        self.heatmap = None
        if heatmap_path is not None and os.path.exists(heatmap_path):
            self.heatmap = Image.open(heatmap_path).convert('RGB')
            print(f"Heatmap: {os.path.basename(heatmap_path)}")
            print(f"Heatmap size: {self.heatmap.size}")

        # Initialize viewing parameters
        self.level = min(initial_level, self.level_count - 1)
        self.x = 0
        self.y = 0
        self.view_size = 1024  # Size of viewport in pixels

        # Setup figure
        self.setup_figure()

    def setup_figure(self):
        """Setup matplotlib figure and controls"""
        if self.heatmap is not None:
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
            self.ax1.set_title('Original WSI')
            self.ax2.set_title('Attention Heatmap')
        else:
            self.fig, self.ax1 = plt.subplots(1, 1, figsize=(10, 8))
            self.ax1.set_title('Original WSI')
            self.ax2 = None

        self.ax1.axis('off')
        if self.ax2 is not None:
            self.ax2.axis('off')

        # Add control panel
        self.add_controls()

        # Initial render
        self.update_view()

    def add_controls(self):
        """Add interactive controls"""
        # Adjust subplot spacing to make room for controls
        plt.subplots_adjust(bottom=0.25)

        # Level slider
        ax_level = plt.axes([0.2, 0.15, 0.6, 0.03])
        self.slider_level = Slider(
            ax_level, 'Level',
            0, self.level_count - 1,
            valinit=self.level,
            valstep=1
        )
        self.slider_level.on_changed(self.on_level_change)

        # X position slider
        ax_x = plt.axes([0.2, 0.10, 0.6, 0.03])
        max_x = max(0, self.level_dimensions[self.level][0] - self.view_size)
        self.slider_x = Slider(
            ax_x, 'X',
            0, max_x,
            valinit=self.x,
            valstep=self.view_size // 10
        )
        self.slider_x.on_changed(self.on_position_change)

        # Y position slider
        ax_y = plt.axes([0.2, 0.05, 0.6, 0.03])
        max_y = max(0, self.level_dimensions[self.level][1] - self.view_size)
        self.slider_y = Slider(
            ax_y, 'Y',
            0, max_y,
            valinit=self.y,
            valstep=self.view_size // 10
        )
        self.slider_y.on_changed(self.on_position_change)

        # Navigation buttons
        ax_left = plt.axes([0.1, 0.10, 0.05, 0.03])
        ax_right = plt.axes([0.85, 0.10, 0.05, 0.03])
        ax_up = plt.axes([0.475, 0.12, 0.05, 0.03])
        ax_down = plt.axes([0.475, 0.08, 0.05, 0.03])

        self.btn_left = Button(ax_left, '←')
        self.btn_right = Button(ax_right, '→')
        self.btn_up = Button(ax_up, '↑')
        self.btn_down = Button(ax_down, '↓')

        self.btn_left.on_clicked(lambda x: self.move(-self.view_size // 2, 0))
        self.btn_right.on_clicked(lambda x: self.move(self.view_size // 2, 0))
        self.btn_up.on_clicked(lambda x: self.move(0, -self.view_size // 2))
        self.btn_down.on_clicked(lambda x: self.move(0, self.view_size // 2))

    def on_level_change(self, val):
        """Handle level slider change"""
        self.level = int(val)

        # Update position sliders ranges
        max_x = max(0, self.level_dimensions[self.level][0] - self.view_size)
        max_y = max(0, self.level_dimensions[self.level][1] - self.view_size)

        self.slider_x.valmax = max_x
        self.slider_y.valmax = max_y

        # Clamp current position
        self.x = min(self.x, max_x)
        self.y = min(self.y, max_y)

        self.slider_x.set_val(self.x)
        self.slider_y.set_val(self.y)

        self.update_view()

    def on_position_change(self, val):
        """Handle position slider change"""
        self.x = int(self.slider_x.val)
        self.y = int(self.slider_y.val)
        self.update_view()

    def move(self, dx, dy):
        """Move view by offset"""
        max_x = max(0, self.level_dimensions[self.level][0] - self.view_size)
        max_y = max(0, self.level_dimensions[self.level][1] - self.view_size)

        self.x = np.clip(self.x + dx, 0, max_x)
        self.y = np.clip(self.y + dy, 0, max_y)

        self.slider_x.set_val(self.x)
        self.slider_y.set_val(self.y)

    def update_view(self):
        """Update displayed images"""
        # Get downsample factor for current level
        downsample = self.level_downsamples[self.level]

        # Calculate position at level 0
        x0 = int(self.x * downsample)
        y0 = int(self.y * downsample)

        # Read region from slide
        size = (self.view_size, self.view_size)
        region = self.slide.read_region((x0, y0), self.level, size).convert('RGB')

        # Display slide
        self.ax1.clear()
        self.ax1.imshow(region)
        self.ax1.axis('off')
        self.ax1.set_title(f'Original WSI (Level {self.level}, Pos: {x0}, {y0})')

        # Display heatmap if available
        if self.heatmap is not None and self.ax2 is not None:
            # Calculate corresponding region in heatmap
            # Heatmap is typically at a fixed downsample level
            # We need to map from slide coordinates to heatmap coordinates

            slide_w, slide_h = self.level_dimensions[0]
            heatmap_w, heatmap_h = self.heatmap.size

            # Calculate scale factors
            scale_x = heatmap_w / slide_w
            scale_y = heatmap_h / slide_h

            # Calculate region in heatmap
            hm_x = int(x0 * scale_x)
            hm_y = int(y0 * scale_y)
            hm_w = int(self.view_size * downsample * scale_x)
            hm_h = int(self.view_size * downsample * scale_y)

            # Clamp to heatmap bounds
            hm_x = np.clip(hm_x, 0, heatmap_w - 1)
            hm_y = np.clip(hm_y, 0, heatmap_h - 1)
            hm_w = min(hm_w, heatmap_w - hm_x)
            hm_h = min(hm_h, heatmap_h - hm_y)

            # Crop heatmap region
            if hm_w > 0 and hm_h > 0:
                heatmap_region = self.heatmap.crop((hm_x, hm_y, hm_x + hm_w, hm_y + hm_h))
                heatmap_region = heatmap_region.resize(size, Image.LANCZOS)
            else:
                heatmap_region = Image.new('RGB', size, (0, 0, 0))

            self.ax2.clear()
            self.ax2.imshow(heatmap_region)
            self.ax2.axis('off')
            self.ax2.set_title(f'Attention Heatmap (Pos: {hm_x}, {hm_y})')

        plt.draw()

    def show(self):
        """Display the viewer"""
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='View WSI and attention heatmap side-by-side'
    )
    parser.add_argument('--slide', type=str, required=True,
                       help='Path to WSI file')
    parser.add_argument('--heatmap', type=str, default=None,
                       help='Path to heatmap image')
    parser.add_argument('--level', type=int, default=2,
                       help='Initial pyramid level (default: 2)')

    args = parser.parse_args()

    if not os.path.exists(args.slide):
        print(f"Error: Slide not found: {args.slide}")
        return

    if args.heatmap is not None and not os.path.exists(args.heatmap):
        print(f"Warning: Heatmap not found: {args.heatmap}")
        args.heatmap = None

    # Create viewer
    viewer = SynchronizedViewer(
        slide_path=args.slide,
        heatmap_path=args.heatmap,
        initial_level=args.level
    )

    print("\nControls:")
    print("  - Use sliders to navigate")
    print("  - Arrow buttons for quick navigation")
    print("  - Level slider to change magnification")
    print("\nClose window to exit.")

    viewer.show()


if __name__ == '__main__':
    main()
