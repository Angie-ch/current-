#!/usr/bin/env python3
"""
LT3P Typhoon Track Prediction - Interactive Demo
tkinter + matplotlib + cartopy desktop application

Launch -> auto run pipeline inference -> display Top10 typhoon tracks on a single map

Usage:
  # First run (auto inference, cache results)
  python typhoon_demo.py

  # Force re-inference
  python typhoon_demo.py --rebuild
"""

import argparse
import os
import pickle
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.lines as mlines

# cartopy optional
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(SCRIPT_DIR, "demo_cache_all.pkl")

# Distinct colors for typhoon buttons
TYPHOON_COLORS = [
    '#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231',
    '#911EB4', '#42D4F4', '#F032E6', '#BFEF45', '#FABED4',
]

# ============================================================
# Light theme palette (Modern Light – VS Code Light style)
# ============================================================
UI_COLORS = {
    'win_bg':       '#F3F3F3',   # window background – light grey
    'panel_bg':     '#FFFFFF',   # side-panel / frame background – pure white
    'card_bg':      '#FFFFFF',   # button / card resting background
    'card_hover':   '#E5F1FB',   # button hover – soft blue tint
    'accent':       '#0066CC',   # accent – clear brand blue
    'text':         '#1A1A1A',   # primary text – near black
    'text_dim':     '#595959',   # secondary / dimmed text
    'status_bg':    '#007ACC',   # status bar background – deep blue
    'status_fg':    '#FFFFFF',   # status bar text – white
    'separator':    '#E5E5E5',   # separator lines – light grey
    'fig_bg':       '#FFFFFF',   # matplotlib figure background
    'axes_bg':      '#F8F9FA',   # matplotlib axes background
}

UI_FONTS = {
    'title':   ('Segoe UI', 14, 'bold'),
    'heading': ('Segoe UI', 12, 'bold'),
    'body':    ('Segoe UI', 10),
    'small':   ('Segoe UI', 9),
    'status':  ('Consolas', 10),
    'btn':     ('Segoe UI', 10),
}

# ============================================================
# Default parameters (match your usual pipeline command)
# ============================================================
DEFAULTS = {
    'diffusion_code': r"C:\Users\fyp\Desktop\newtry",
    'diffusion_ckpt': r"C:\Users\fyp\Desktop\newtry\checkpoints\best.pt",
    'trajectory_ckpt': os.path.join(SCRIPT_DIR, "checkpoints_finetune_free", "best_finetune.pt"),
    'norm_stats': r"C:\Users\fyp\Desktop\newtry\norm_stats.pt",
    'data_root': r"C:\Users\fyp\Desktop\Typhoon_data_final",
    'track_csv': os.path.join(SCRIPT_DIR, "processed_typhoon_tracks.csv"),
    'num_typhoons': 999,
    'top_k': 10,
    'ddim_steps': 50,
}


# ============================================================
# Name mapping
# ============================================================

def build_name_map(track_csv: str) -> dict:
    """Build typhoon_id -> typhoon_name mapping from CSV"""
    name_map = {}
    try:
        import csv
        with open(track_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = row['typhoon_id']
                if tid not in name_map:
                    name_map[tid] = row['typhoon_name']
    except Exception as e:
        print(f"[Warning] Failed to read {track_csv}: {e}")
    return name_map


# ============================================================
# Pipeline invocation
# ============================================================

def run_pipeline(args) -> list:
    """Call predict_pipeline.run_end_to_end() for inference"""
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)

    from predict_pipeline import run_end_to_end

    class PipelineArgs:
        pass

    pa = PipelineArgs()
    pa.mode = "end_to_end"
    pa.diffusion_code = args.diffusion_code
    pa.diffusion_ckpt = args.diffusion_ckpt
    pa.trajectory_ckpt = args.trajectory_ckpt
    pa.norm_stats = args.norm_stats
    pa.data_root = args.data_root
    pa.track_csv = args.track_csv
    pa.num_typhoons = args.num_typhoons
    pa.top_k = args.top_k
    pa.ddim_steps = args.ddim_steps
    pa.output_dir = os.path.join(SCRIPT_DIR, "demo_outputs")
    pa.bias_path = getattr(args, 'bias_path', None)
    pa.preprocess_dir = getattr(args, 'preprocess_dir', None)
    pa.num_samples = 5
    pa.target_typhoon_ids = None  # run all test set typhoons

    return run_end_to_end(pa)


def load_or_run(args) -> tuple:
    """Load from cache if available, otherwise run pipeline.

    Cache stores ALL results (unsorted). Sorting + top_k slicing happens
    at the end so changing --top_k never requires re-inference.
    """
    if not args.rebuild and os.path.exists(CACHE_FILE):
        print(f"[Cache] Loading from {CACHE_FILE}...")
        with open(CACHE_FILE, 'rb') as f:
            data = pickle.load(f)
        all_results = data['results']
        name_map = data['name_map']
    else:
        print("[Inference] Running pipeline (first time may take minutes)...")
        results = run_pipeline(args)

        # Serialize numpy arrays
        all_results = []
        for r in results:
            sr = {}
            for k, v in r.items():
                sr[k] = np.array(v) if hasattr(v, '__array__') else v
            all_results.append(sr)

        name_map = build_name_map(args.track_csv)

        # Save ALL results to cache (no sorting, no slicing)
        cache_data = {'results': all_results, 'name_map': name_map}
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"[Cache] Saved {len(all_results)} typhoons to {CACHE_FILE}")

    # Sort by mean error and take top_k (runs every time, cheap)
    sorted_results = sorted(all_results, key=lambda r: float(np.mean(r['error_km'])))
    top_results = sorted_results[:args.top_k]
    return top_results, name_map


# ============================================================
# GUI Application
# ============================================================

class TyphoonDemoApp:
    # Fixed semantic colors
    COLOR_HISTORY = '#2196F3'   # Blue  = history
    COLOR_PREDICT = '#F44336'   # Red   = predicted
    COLOR_TRUTH   = '#4CAF50'   # Green = ground truth

    def __init__(self, root: tk.Tk, args):
        self.root = root
        self.args = args
        self.results = []
        self.name_map = {}
        self._loading = False
        # Track visibility per typhoon index (False = hidden initially)
        self._visible = {}
        # Store plot artists per typhoon index for toggle
        self._artists = {}
        # Animation timer id per typhoon index
        self._anim_timers = {}

        self.root.title("LT3P Typhoon Track Prediction Demo")
        self.root.geometry("1400x850")
        self.root.minsize(1100, 700)
        self.root.configure(bg=UI_COLORS['win_bg'])

        # Configure ttk theme
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background=UI_COLORS['panel_bg'])
        style.configure('TLabel', background=UI_COLORS['panel_bg'],
                        foreground=UI_COLORS['text'], font=UI_FONTS['body'])
        style.configure('TLabelframe', background=UI_COLORS['panel_bg'],
                        foreground=UI_COLORS['accent'], font=UI_FONTS['heading'],
                        bordercolor=UI_COLORS['separator'])
        style.configure('TLabelframe.Label', background=UI_COLORS['panel_bg'],
                        foreground=UI_COLORS['accent'], font=UI_FONTS['heading'])
        style.configure('TSeparator', background=UI_COLORS['separator'])
        style.configure('Vertical.TScrollbar',
                        background=UI_COLORS['panel_bg'],
                        troughcolor=UI_COLORS['win_bg'],
                        arrowcolor=UI_COLORS['text_dim'])

        self._build_ui()
        self.root.after(100, self._start_loading)

    def _build_ui(self):
        """Build the UI layout: map on left, typhoon list on right"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Left: matplotlib canvas
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(11, 7), dpi=100,
                              facecolor=UI_COLORS['fig_bg'])
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().configure(bg=UI_COLORS['fig_bg'])
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar
        toolbar_frame = ttk.Frame(left_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # Right: typhoon list panel
        right_frame = ttk.LabelFrame(main_frame,
                                     text="Typhoons (click to toggle)",
                                     padding=8)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))

        # Color legend at top of right panel
        legend_frame = ttk.Frame(right_frame)
        legend_frame.pack(fill=tk.X, pady=(0, 10))
        for color, label in [(self.COLOR_HISTORY, 'History 48 h'),
                             (self.COLOR_TRUTH, 'Ground Truth 72 h'),
                             (self.COLOR_PREDICT, 'Predicted 72 h')]:
            row = ttk.Frame(legend_frame)
            row.pack(fill=tk.X, pady=2)
            swatch = tk.Canvas(row, width=24, height=12,
                               highlightthickness=0,
                               bg=UI_COLORS['panel_bg'])
            swatch.create_rectangle(2, 1, 22, 11, fill=color, outline=color,
                                    width=0)
            # rounded-feel: overlay tiny arcs at corners
            swatch.create_oval(0, 0, 6, 12, fill=color, outline=color)
            swatch.create_oval(18, 0, 24, 12, fill=color, outline=color)
            swatch.pack(side=tk.LEFT, padx=(0, 6))
            ttk.Label(row, text=label, font=UI_FONTS['small']).pack(side=tk.LEFT)

        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # Scrollable button list
        list_canvas = tk.Canvas(right_frame, width=220, highlightthickness=0,
                                bg=UI_COLORS['panel_bg'])
        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL,
                                  command=list_canvas.yview)
        self.btn_frame = tk.Frame(list_canvas, bg=UI_COLORS['panel_bg'])
        self.btn_frame.bind('<Configure>',
                            lambda e: list_canvas.configure(
                                scrollregion=list_canvas.bbox('all')))
        list_canvas.create_window((0, 0), window=self.btn_frame, anchor='nw')
        list_canvas.configure(yscrollcommand=scrollbar.set)
        list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bottom status bar (tk.Label for direct bg/fg control)
        self.info_var = tk.StringVar(value="Loading data, please wait...")
        info_label = tk.Label(self.root, textvariable=self.info_var,
                              font=UI_FONTS['status'],
                              bg=UI_COLORS['status_bg'],
                              fg=UI_COLORS['status_fg'],
                              anchor='w', padx=12, pady=6)
        info_label.pack(fill=tk.X, padx=0, pady=0, side=tk.BOTTOM)

        # Initial loading message on canvas
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(UI_COLORS['axes_bg'])
        ax.text(0.5, 0.55, "LT3P",
                ha='center', va='center', fontsize=36,
                color=UI_COLORS['accent'], fontweight='bold',
                transform=ax.transAxes)
        ax.text(0.5, 0.40, "Loading data...",
                ha='center', va='center', fontsize=18,
                color=UI_COLORS['accent'], fontweight='bold',
                transform=ax.transAxes)
        ax.text(0.5, 0.32, "First run requires inference, please wait.",
                ha='center', va='center', fontsize=12,
                color=UI_COLORS['text_dim'],
                transform=ax.transAxes)
        ax.set_axis_off()
        self.canvas.draw()

    # ---- Background loading ----

    def _start_loading(self):
        self._loading = True
        self.info_var.set("Loading data (first run requires inference, please wait)...")

        def _worker():
            try:
                results, name_map = load_or_run(self.args)
                self.root.after(0, lambda: self._on_data_ready(results, name_map))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self._on_data_error(str(e)))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _on_data_ready(self, results, name_map):
        self._loading = False
        self.results = results
        self.name_map = name_map
        # All hidden initially — user clicks to show
        self._visible = {i: False for i in range(len(results))}

        self._setup_map()
        self._build_typhoon_buttons()

        n = len(results)
        mean_all = np.mean([float(np.mean(r['error_km'])) for r in results])
        self.info_var.set(
            f"Top {n} Typhoons | Overall Mean Error: {mean_all:.1f} km | "
            f"Click a typhoon name to show/hide its track"
        )

    def _on_data_error(self, error_msg):
        self._loading = False
        self.info_var.set(f"Load failed: {error_msg}")
        messagebox.showerror("Load Failed", f"Inference error:\n{error_msg}")

    # ---- Map setup (empty, no tracks) ----

    def _setup_map(self):
        """Draw the base map with no tracks"""
        self.fig.clear()
        self._artists.clear()
        self.fig.set_facecolor(UI_COLORS['fig_bg'])

        if HAS_CARTOPY:
            ax = self.fig.add_subplot(111, projection=ccrs.PlateCarree())
            ax.set_facecolor(UI_COLORS['axes_bg'])
            ax.set_extent([100, 180, 0, 60], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor='#F5F5F6', edgecolor='#B0BEC5',
                           linewidth=0.5)
            ax.add_feature(cfeature.OCEAN, facecolor='#E3F2FD')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#B0BEC5')
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle='--',
                           edgecolor='#CFD8DC')
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='#B0BEC5',
                              alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'fontsize': 10, 'color': UI_COLORS['text_dim']}
            gl.ylabel_style = {'fontsize': 10, 'color': UI_COLORS['text_dim']}
            self._transform = ccrs.PlateCarree()
        else:
            ax = self.fig.add_subplot(111)
            ax.set_facecolor(UI_COLORS['axes_bg'])
            ax.set_xlim(100, 180)
            ax.set_ylim(0, 60)
            ax.set_aspect('equal')
            ax.grid(True, linewidth=0.3, alpha=0.5, linestyle='--',
                    color='#B0BEC5')
            ax.tick_params(colors=UI_COLORS['text_dim'], labelsize=10)
            self._transform = None

        ax.set_xlabel('Longitude (°E)', fontsize=11,
                      color=UI_COLORS['text_dim'])
        ax.set_ylabel('Latitude (°N)', fontsize=11,
                      color=UI_COLORS['text_dim'])
        ax.set_title("LT3P Top Typhoon Track Predictions",
                     fontsize=15, fontweight='bold',
                     color=UI_COLORS['accent'], pad=12)

        self.ax = ax
        self.fig.tight_layout()
        self.canvas.draw()

    # ---- Right-panel typhoon buttons ----

    def _build_typhoon_buttons(self):
        """Create a toggle button for each typhoon in the right panel"""
        self._buttons = {}
        for i, r in enumerate(self.results):
            storm_id = r['storm_id']
            name = self.name_map.get(storm_id, storm_id)
            mean_err = float(np.mean(r['error_km']))
            tc = TYPHOON_COLORS[i % len(TYPHOON_COLORS)]

            # Container row for color bar + button
            row = tk.Frame(self.btn_frame, bg=UI_COLORS['panel_bg'])
            row.pack(fill=tk.X, pady=2)

            # Left color bar indicator
            bar = tk.Frame(row, width=4, bg=tc)
            bar.pack(side=tk.LEFT, fill=tk.Y)
            bar.pack_propagate(False)

            btn = tk.Button(
                row,
                text=f"#{i+1}  {name}\n     {mean_err:.0f} km avg error",
                font=UI_FONTS['btn'],
                width=24,
                relief=tk.FLAT,
                bg=UI_COLORS['card_bg'],
                fg=UI_COLORS['text'],
                activebackground=UI_COLORS['card_hover'],
                activeforeground=UI_COLORS['text'],
                anchor='w',
                padx=8,
                pady=4,
                cursor='hand2',
                highlightbackground=UI_COLORS['separator'],
                highlightthickness=1,
                command=lambda idx=i: self._toggle_typhoon(idx),
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

            # Hover effects
            def _on_enter(e, b=btn):
                if b.cget('bg') == UI_COLORS['card_bg']:
                    b.configure(bg=UI_COLORS['card_hover'])

            def _on_leave(e, b=btn):
                if b.cget('bg') == UI_COLORS['card_hover']:
                    b.configure(bg=UI_COLORS['card_bg'])

            btn.bind('<Enter>', _on_enter)
            btn.bind('<Leave>', _on_leave)

            self._buttons[i] = btn

    # ---- Toggle + animation ----

    def _toggle_typhoon(self, idx):
        """Toggle a typhoon's track on/off with animation"""
        # Cancel any running animation for this typhoon
        if idx in self._anim_timers:
            self.root.after_cancel(self._anim_timers.pop(idx))

        if self._visible[idx]:
            # Hide: remove artists
            self._remove_track(idx)
            self._visible[idx] = False
            self._buttons[idx].configure(relief=tk.FLAT,
                                         bg=UI_COLORS['card_bg'],
                                         fg=UI_COLORS['text'])
            r = self.results[idx]
            name = self.name_map.get(r['storm_id'], r['storm_id'])
            self.info_var.set(f"{name}: hidden")
        else:
            # Show: animate drawing
            self._visible[idx] = True
            tc = TYPHOON_COLORS[idx % len(TYPHOON_COLORS)]
            # Build a lighter tint of the typhoon color for active bg
            r_c = int(tc[1:3], 16)
            g_c = int(tc[3:5], 16)
            b_c = int(tc[5:7], 16)
            # Light tint: blend towards white for light theme
            active_bg = '#{:02X}{:02X}{:02X}'.format(
                min(255, r_c + (255 - r_c) * 3 // 4),
                min(255, g_c + (255 - g_c) * 3 // 4),
                min(255, b_c + (255 - b_c) * 3 // 4))
            self._buttons[idx].configure(relief=tk.FLAT,
                                         bg=active_bg,
                                         fg=UI_COLORS['text'])
            self._animate_track(idx)

    def _remove_track(self, idx):
        """Remove all artists for a typhoon"""
        for a in self._artists.pop(idx, []):
            a.remove()
        self.canvas.draw_idle()

    def _animate_track(self, idx):
        """Animate drawing of history (blue), truth (green), pred (red)"""
        r = self.results[idx]
        plot_kw = dict(transform=self._transform) if self._transform else {}

        hist_lon = np.array(r['history_lon'])
        hist_lat = np.array(r['history_lat'])
        gt_lon = np.array(r['gt_lon'])
        gt_lat = np.array(r['gt_lat'])
        pred_lon = np.array(r['pred_lon'])
        pred_lat = np.array(r['pred_lat'])

        # Build animation sequence: (lon_array, lat_array, color, marker, style, zorder)
        segments = [
            (hist_lon, hist_lat, self.COLOR_HISTORY, 'o', '-',  3),
            (gt_lon,   gt_lat,   self.COLOR_TRUTH,   's', '-',  4),
            (pred_lon, pred_lat, self.COLOR_PREDICT,  'D', '--', 4),
        ]

        self._artists[idx] = []
        # Flatten all points into a single animation queue
        queue = []  # list of (seg_idx, point_idx)
        for si, (lons, lats, *_) in enumerate(segments):
            for pi in range(len(lons)):
                queue.append((si, pi))

        # State: current line artists being built (one per segment)
        line_objs = [None, None, None]

        delay_ms = 30  # ms between points

        def _step(qi):
            if qi >= len(queue):
                # Animation done — show info
                self._show_typhoon_info(idx)
                return
            if not self._visible.get(idx, False):
                return  # cancelled

            si, pi = queue[qi]
            lons, lats, color, marker, style, zorder = segments[si]

            if line_objs[si] is None:
                # Create new line with first point
                line, = self.ax.plot(
                    [lons[pi]], [lats[pi]],
                    marker=marker, linestyle=style, color=color,
                    linewidth=1.8, markersize=3.5, alpha=0.85,
                    zorder=zorder, **plot_kw
                )
                line_objs[si] = line
                self._artists[idx].append(line)
            else:
                # Extend existing line
                line = line_objs[si]
                xdata = list(line.get_xdata()) + [lons[pi]]
                ydata = list(line.get_ydata()) + [lats[pi]]
                line.set_xdata(xdata)
                line.set_ydata(ydata)

            self.canvas.draw_idle()
            self._anim_timers[idx] = self.root.after(delay_ms, _step, qi + 1)

        # Start point marker (star at prediction start = end of history)
        star, = self.ax.plot(
            hist_lon[-1], hist_lat[-1], '*',
            color='#FF9800', markersize=12,
            markeredgecolor='black', markeredgewidth=0.5,
            zorder=6, **plot_kw
        )
        self._artists[idx] = [star]

        _step(0)

    def _show_typhoon_info(self, idx):
        """Update info bar with typhoon details after animation"""
        r = self.results[idx]
        name = self.name_map.get(r['storm_id'], r['storm_id'])
        error_km = np.array(r['error_km'])
        n_gt = len(error_km)
        self.info_var.set(
            f"{name} | Mean: {error_km.mean():.1f} km | "
            f"+24h: {error_km[min(7, n_gt-1)]:.1f} km | "
            f"+48h: {error_km[min(15, n_gt-1)]:.1f} km | "
            f"+72h: {error_km[-1]:.1f} km"
        )


# ============================================================
# Entry point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="LT3P Typhoon Track Prediction Demo")

    parser.add_argument('--diffusion_code', type=str,
                        default=DEFAULTS['diffusion_code'])
    parser.add_argument('--diffusion_ckpt', type=str,
                        default=DEFAULTS['diffusion_ckpt'])
    parser.add_argument('--trajectory_ckpt', type=str,
                        default=DEFAULTS['trajectory_ckpt'])
    parser.add_argument('--norm_stats', type=str,
                        default=DEFAULTS['norm_stats'])
    parser.add_argument('--data_root', type=str,
                        default=DEFAULTS['data_root'])
    parser.add_argument('--track_csv', type=str,
                        default=DEFAULTS['track_csv'])
    parser.add_argument('--num_typhoons', type=int,
                        default=DEFAULTS['num_typhoons'])
    parser.add_argument('--top_k', type=int,
                        default=DEFAULTS['top_k'])
    parser.add_argument('--ddim_steps', type=int,
                        default=DEFAULTS['ddim_steps'])
    parser.add_argument('--rebuild', action='store_true',
                        help='Force re-inference (ignore cache)')

    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    app = TyphoonDemoApp(root, args)
    root.mainloop()


if __name__ == '__main__':
    main()
