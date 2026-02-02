#!/usr/bin/env python3
"""
Refactored 2D Pipeline Results Plotter - Complete Version
Incorporating fixes from the original plot_pipeline_results.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import json
import csv
import warnings
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
from scipy.interpolate import griddata, interp1d

# Import from existing FWHM module
try:
    from fwhm_methods_addon import estimate_all_fwhm_methods
except ImportError:
    # Fallback if module not found
    def estimate_all_fwhm_methods(*args, **kwargs):
        methods = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
        return {m: 0.0 for m in methods}

# Suppress warnings
warnings.filterwarnings("ignore")

# Global plotting settings (matching original)
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 18
})

# =============================================================================
# HELPER FUNCTIONS FROM ORIGINAL CODE
# =============================================================================

def ensure_dir(path):
    """Garante que o diretório existe"""
    os.makedirs(path, exist_ok=True)

def classify_result(value, reference, is_axial=True):
    """
    Classifica resultado baseado em erro relativo
    Incorporated color scheme from plot_pipeline_results_FWHM.py
    """
    if reference is None or reference <= 0 or value is None or np.isnan(value):
        return '#7f8c8d', 'Unknown'  # Cinza
    
    error_pct = abs(value - reference) / reference * 100
    
    # Usando o esquema de cores do plot_pipeline_results_FWHM.py
    if is_axial:
        # Limites mais rígidos para axial (ajustados para 4 categorias)
        if error_pct < 10:
            return '#2ecc71', 'Good (<10% error)'       # Verde
        elif error_pct < 20:
            return '#f39c12', 'Moderate (10-20% error)' # Amarelo/Laranja
        elif error_pct < 40:
            return '#e67e22', 'Poor (20-40% error)'     # Laranja escuro
        else:
            return '#e74c3c', 'Very Poor (>40% error)'  # Vermelho
    else:
        # Limites para lateral usando o esquema FWHM
        if error_pct < 15:
            return '#2ecc71', 'Good (<15% error)'       # Verde
        elif error_pct < 50:
            return '#f39c12', 'Approximation (15-50% error)'  # Amarelo/Laranja
        else:
            return '#e74c3c', 'Poor (>50% error)'       # Vermelho

def create_legend_elements(include_theoretical=True, include_patterns=True):
    """
    Cria elementos de legenda consistentes com plot_pipeline_results_FWHM.py
    """
    legend_elements = []
    
    # Seção de Qualidade (baseado no classify_result)
    if include_theoretical:
        legend_elements.append(
            Line2D([0], [0], color='black', ls='--', lw=3, label='Theoretical Baseline')
        )
    
    # Seção de Metodologia (padrões)
    if include_patterns:
        legend_elements.extend([
            Patch(facecolor='gray', hatch='///', alpha=0.7, edgecolor='black', 
                  label='Digital Twin (Hatched)'),
            Patch(facecolor='gray', alpha=0.9, edgecolor='black', 
                  label='TFM (Solid)')
        ])
    
    # Seção de Categorias de Qualidade
    legend_elements.extend([
        Patch(facecolor='#2ecc71', label='Good (<15% error)'),
        Patch(facecolor='#f39c12', label='Approximation (15-50% error)'),
        Patch(facecolor='#e74c3c', label='Poor (>50% error)'),
        Patch(facecolor='#7f8c8d', label='Unknown/Invalid')
    ])
    
    return legend_elements

# =============================================================================
# DATA MODELS
# =============================================================================

class FieldData:
    """Data container for field data with the same orientation logic as original"""
    
    def __init__(self, x_vals: np.ndarray, z_vals: np.ndarray, field_matrix: np.ndarray,
                 source_type: str = 'dt'):
        """
        Initialize field data with the same orientation logic as original code
        
        Args:
            x_vals: Lateral coordinates (mm)
            z_vals: Depth coordinates (mm)
            field_matrix: 2D field data
            source_type: 'dt' for Digital Twin, 'tfm' for TFM
        """
        self.x_vals = np.array(x_vals)
        self.z_vals = np.array(z_vals)
        self.raw_matrix = np.array(field_matrix)
        self.source_type = source_type
        
        # Apply the same orientation logic as original code
        self.field_matrix = self._apply_original_orientation_logic()
        
        print(f"[FieldData {source_type.upper()}] Final shape: {self.field_matrix.shape} (z={len(self.z_vals)}, x={len(self.x_vals)})")
    
    def _apply_original_orientation_logic(self) -> np.ndarray:
        """Apply the exact orientation logic from the original code"""
        # Digital Twin logic from original load_data_2d()
        if self.source_type == 'dt':
            if self.raw_matrix.shape[0] == len(self.z_vals) and self.raw_matrix.shape[1] == len(self.x_vals):
                return self.raw_matrix  # Already in (z, x) format
            elif self.raw_matrix.shape[0] == len(self.x_vals) and self.raw_matrix.shape[1] == len(self.z_vals):
                return self.raw_matrix.T  # Transpose to (z, x)
            else:
                print(f"[WARNING] Unexpected DT shape: {self.raw_matrix.shape}")
                return self.raw_matrix
        
        # TFM logic from original load_data_2d()
        elif self.source_type == 'tfm':
            # FIX from original: Check shape and transpose if necessary
            if self.raw_matrix.shape[0] == len(self.x_vals) and self.raw_matrix.shape[1] == len(self.z_vals):
                # Already in (x, z) format as per original
                return self.raw_matrix
            elif self.raw_matrix.shape[0] == len(self.z_vals) and self.raw_matrix.shape[1] == len(self.x_vals):
                # Needs transpose to (x, z) as per original
                return self.raw_matrix.T
            else:
                print(f"[WARNING] Unexpected TFM shape: {self.raw_matrix.shape}")
                return self.raw_matrix
        
        return self.raw_matrix
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.field_matrix.shape
    
    def axial_profile(self, x_index: Optional[int] = None) -> np.ndarray:
        """Extract axial profile at given x index or center"""
        if x_index is None:
            x_index = np.argmin(np.abs(self.x_vals - 0.0))
        
        # For TFM, axial profile is at fixed x (x_index) across all z
        if self.source_type == 'tfm':
            # TFM data is in (x, z) format, so we need [x_index, :]
            return self.field_matrix[x_index, :]
        else:
            # DT data is in (z, x) format, so we need [:, x_index]
            return self.field_matrix[:, x_index]
    
    def lateral_profile_at_z(self, z_index: int) -> np.ndarray:
        """Extract lateral profile at given z index"""
        if self.source_type == 'tfm':
            # TFM data is in (x, z) format, so we need [:, z_index]
            return self.field_matrix[:, z_index]
        else:
            # DT data is in (z, x) format, so we need [z_index, :]
            return self.field_matrix[z_index, :]
    
    def normalized_matrix(self) -> np.ndarray:
        """Return field matrix normalized by global maximum"""
        max_val = np.max(self.field_matrix)
        return self.field_matrix / max_val if max_val > 0 else self.field_matrix
    
    def get_dB_matrix(self, vmin: float = -40, vmax: float = 0) -> np.ndarray:
        """Convert to dB scale"""
        normalized = self.normalized_matrix()
        return 20 * np.log10(normalized + 1e-6)
    
    def get_corrected_for_plotting(self) -> np.ndarray:
        """
        Apply the SMART LOGIC from original plot_field_maps_2d for TFM
        """
        if self.source_type == 'tfm':
            # SMART LOGIC: Auto-Correct Orientation for plotting
            # Check if dimensions match (Lateral, Depth). If so, transpose to (Depth, Lateral).
            # This aligns the data so Rows=Depth, Cols=Lateral.
            if len(self.x_vals) != len(self.z_vals):
                return self.field_matrix.T
        return self.field_matrix


class PipelineParameters:
    """Configuration parameters with validation and defaults"""
    
    def __init__(self, **kwargs):
        self.dt0 = float(kwargs.get('Dt0', 59.2))
        self.target_z = float(kwargs.get('DF', 23.6))
        self.theoretical_fwhm = float(kwargs.get('theoretical_fwhm_mm', 0.0))
        self.raw_params = kwargs  # Keep original for reporting
    
    @classmethod
    def from_json_file(cls, filepath: Path) -> 'PipelineParameters':
        """Load parameters from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                params = json.load(f)
            return cls(**params)
        except Exception as e:
            print(f"[WARNING] Failed to load parameters from {filepath}: {e}")
            return cls()


class PipelineDataset:
    """Container for both Digital Twin and TFM data with original fixes"""
    
    def __init__(self, dt_data: FieldData, tfm_data: FieldData):
        self.dt_data = dt_data
        self.tfm_data = tfm_data
        
        # Apply FIX 1 from original plot_profiles_comparison: Find actual peak in TFM
        self.tfm_peak_z, self.tfm_peak_idx = self._find_tfm_peak_with_original_fix()
        
        print(f"[Dataset] TFM Peak at z={self.tfm_peak_z:.1f}mm (index={self.tfm_peak_idx})")
    
    def _find_tfm_peak_with_original_fix(self) -> Tuple[float, int]:
        """Find actual peak depth in TFM data using original fix"""
        # This replicates the FIX from plot_profiles_comparison in original code
        
        # First, find peak in the region after the interface
        dt0 = 59.2  # Default, will be updated from params
        mask_post_interface = self.tfm_data.z_vals > dt0 + 5.0
        
        if np.any(mask_post_interface):
            # Get indices where mask is True
            valid_indices = np.where(mask_post_interface)[0]
            # Find peak among valid indices
            tfm_axial = self.tfm_data.axial_profile()
            peak_value = -np.inf
            peak_idx = 0
            for idx in valid_indices:
                if tfm_axial[idx] > peak_value:
                    peak_value = tfm_axial[idx]
                    peak_idx = idx
            
            actual_peak_z = self.tfm_data.z_vals[peak_idx]
            print(f"[INFO] TFM peak at depth: {actual_peak_z:.1f} mm")
            return actual_peak_z, peak_idx
        else:
            # Fallback: use center
            center_idx = len(self.tfm_data.z_vals) // 2
            actual_peak_z = self.tfm_data.z_vals[center_idx]
            print(f"[WARNING] Using center depth: {actual_peak_z:.1f} mm")
            return actual_peak_z, center_idx
    
    def get_tfm_lateral_profile(self) -> np.ndarray:
        """Get lateral profile at peak depth using original logic"""
        # Extract lateral profile at the actual peak depth
        lateral = self.tfm_data.lateral_profile_at_z(self.tfm_peak_idx)
        
        # Apply MIRROR THE DATA fix from original code
        if len(self.tfm_data.x_vals) == len(self.tfm_data.z_vals):
            lateral = np.flip(lateral)
        
        return lateral


# =============================================================================
# DATA LOADER (REPLICATING ORIGINAL)
# =============================================================================

class DataLoader:
    """Loads pipeline data from files using original logic"""
    
    @staticmethod
    def load_from_directory(root_dir: Path) -> PipelineDataset:
        """Load both Digital Twin and TFM data using original logic"""
        print(f"[INFO] Loading data from: {root_dir}")
        
        try:
            # Load Digital Twin data (replicating original load_data_2d)
            dt_dir = root_dir / "digital_twin"
            print(f"[INFO] Loading Digital Twin from: {dt_dir}")
            
            z_dt = np.loadtxt(dt_dir / "field_z_vals.csv", delimiter=',')
            x_dt = np.loadtxt(dt_dir / "field_x_vals.csv", delimiter=',')
            p_field_raw = np.loadtxt(dt_dir / "field_p_field.csv", delimiter=',')
            
            # Apply original orientation logic
            if p_field_raw.shape[0] == len(z_dt) and p_field_raw.shape[1] == len(x_dt):
                p_field = p_field_raw  # Already in (z, x) format
            elif p_field_raw.shape[0] == len(x_dt) and p_field_raw.shape[1] == len(z_dt):
                p_field = p_field_raw.T  # Transpose to (z, x)
            else:
                print(f"[WARNING] Unexpected DT shape: {p_field_raw.shape}")
                p_field = p_field_raw
            
            dt_data = FieldData(x_dt, z_dt, p_field, source_type='dt')
            
            # Load TFM data (replicating original load_data_2d)
            tfm_dir = root_dir / "fmc_tfm"
            print(f"[INFO] Loading TFM from: {tfm_dir}")
            
            z_tfm = np.loadtxt(tfm_dir / "results_z_vals.csv", delimiter=',')
            x_tfm = np.loadtxt(tfm_dir / "results_x_vals.csv", delimiter=',')
            envelope_2d = np.loadtxt(tfm_dir / "results_envelope_2d.csv", delimiter=',')
            
            # Apply original FIX for TFM orientation
            if envelope_2d.shape[0] == len(x_tfm) and envelope_2d.shape[1] == len(z_tfm):
                # Already in (x, z) format as per original
                pass
            elif envelope_2d.shape[0] == len(z_tfm) and envelope_2d.shape[1] == len(x_tfm):
                # Needs transpose to (x, z) as per original
                envelope_2d = envelope_2d.T
            else:
                print(f"Warning: envelope_2d shape {envelope_2d.shape} doesn't match")
            
            tfm_data = FieldData(x_tfm, z_tfm, envelope_2d, source_type='tfm')
            
            print(f"[INFO] Data loaded successfully:")
            print(f"  Digital Twin: {dt_data.shape} (z x x)")
            print(f"  TFM: {tfm_data.shape} (x x z)")
            
            return PipelineDataset(dt_data, tfm_data)
            
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            traceback.print_exc()
            # Create minimal dataset to allow continuation
            x = np.linspace(-10, 10, 21)
            z = np.linspace(0, 100, 101)
            dummy_field = np.ones((len(z), len(x)))
            dt_data = FieldData(x, z, dummy_field, source_type='dt')
            tfm_data = FieldData(x, z, dummy_field, source_type='tfm')
            return PipelineDataset(dt_data, tfm_data)


# =============================================================================
# PLOT CONFIGURATION
# =============================================================================

class PlotConfig:
    """Immutable plot configuration matching original"""
    
    def __init__(self, **kwargs):
        self.figsize = kwargs.get('figsize', (20, 12))
        self.dpi = kwargs.get('dpi', 300)
        self.font_size = kwargs.get('font_size', 18)
        self.colormap = kwargs.get('colormap', 'jet')
        self.vmin = kwargs.get('vmin', -40)
        self.vmax = kwargs.get('vmax', 0)
        
        # Set matplotlib defaults to match original
        plt.rcParams.update({
            'font.size': self.font_size,
            'axes.titlesize': self.font_size,
            'axes.labelsize': self.font_size,
            'xtick.labelsize': self.font_size,
            'ytick.labelsize': self.font_size,
            'legend.fontsize': self.font_size,
            'figure.titlesize': self.font_size
        })


# =============================================================================
# PLOT BASE CLASSES
# =============================================================================

class BasePlot(ABC):
    """Abstract base class for all plots"""
    
    def __init__(self, dataset: PipelineDataset, params: PipelineParameters, config: PlotConfig):
        self.dataset = dataset
        self.params = params
        self.config = config
        self.fig = None
        self.output_path = None
    
    def generate(self, output_path: Path) -> None:
        """Template method for plot generation"""
        try:
            self.output_path = output_path
            self._setup_figure()
            self._create_plot()
            self._add_annotations()
            self._save_plot()
            self._cleanup()
            print(f"✅ Saved plot to: {output_path}")
        except Exception as e:
            print(f"❌ Failed to generate plot {self.__class__.__name__}: {e}")
            traceback.print_exc()
            self._cleanup()
    
    def _setup_figure(self) -> None:
        """Initialize figure with configuration"""
        self.fig = plt.figure(figsize=self.config.figsize)
    
    @abstractmethod
    def _create_plot(self) -> None:
        """Create the actual plot (to be implemented by subclasses)"""
        pass
    
    def _add_annotations(self) -> None:
        """Add default annotations (can be overridden)"""
        pass
    
    def _save_plot(self) -> None:
        """Save plot to file"""
        if self.fig is None:
            return
        plt.tight_layout()
        plt.savefig(self.output_path, dpi=self.config.dpi, bbox_inches='tight')
    
    def _cleanup(self) -> None:
        """Clean up resources"""
        if self.fig is not None:
            plt.close(self.fig)
        self.fig = None


# =============================================================================
# CONCRETE PLOT CLASSES (REPLICATING ORIGINAL)
# =============================================================================

class FieldMapPlot(BasePlot):
    """Field maps comparison plot replicating original plot_field_maps_2d"""
    
    def _setup_figure(self) -> None:
        self.fig = plt.figure(figsize=(20, 12))
    
    def _create_plot(self) -> None:
        # Replicate original plot_field_maps_2d logic
        Dt0 = self.params.dt0
        target_z = self.params.target_z
        
        # 1. Digital Twin (Full) - matching original
        ax1 = plt.subplot(2, 3, 1)
        p_db = 20 * np.log10(self.dataset.dt_data.normalized_matrix() + 1e-6)
        im1 = ax1.pcolormesh(self.dataset.dt_data.x_vals, self.dataset.dt_data.z_vals, p_db, 
                             cmap='jet', vmin=-40, vmax=0, shading='auto')
        ax1.set_title("Digital Twin Field Map (dB)")
        ax1.axvline(0, color='gray', ls='--', alpha=0.5)
        ax1.axhline(Dt0, color='cyan', ls='--', alpha=0.7, label='Interface')
        ax1.axhline(target_z, color='white', ls='--', alpha=0.7, label='Focus')
        ax1.set_xlabel("x (mm)")
        ax1.set_ylabel("z (mm)")
        plt.colorbar(im1, ax=ax1, label='dB')
        ax1.legend(loc='upper right', fontsize=10)
        
        # 2. Digital Twin (Zoom) - matching original
        ax2 = plt.subplot(2, 3, 2)
        z_mask = (self.dataset.dt_data.z_vals > target_z - 30) & (self.dataset.dt_data.z_vals < target_z + 30)
        if np.any(z_mask):
            im2 = ax2.pcolormesh(self.dataset.dt_data.x_vals, self.dataset.dt_data.z_vals[z_mask], 
                                 p_db[z_mask, :], cmap='jet', vmin=-20, vmax=0, shading='auto')
            ax2.set_title("Digital Twin - Focal Zone")
            ax2.axvline(0, color='gray', ls='--', alpha=0.5)
            ax2.axhline(target_z, color='white', ls='--', alpha=0.7)
            ax2.set_xlabel("x (mm)")
            ax2.set_ylabel("z (mm)")
        
        # 3. TFM Reconstruction (Full) - with original SMART LOGIC
        ax3 = plt.subplot(2, 3, 4)
        envelope_norm = 20 * np.log10(self.dataset.tfm_data.normalized_matrix() + 1e-6)
        
        # Apply original SMART LOGIC for orientation correction
        if len(self.dataset.tfm_data.x_vals) != len(self.dataset.tfm_data.z_vals):
            envelope_norm = envelope_norm.T
        
        # Plot with corrected orientation
        im3 = ax3.pcolormesh(self.dataset.tfm_data.x_vals, self.dataset.tfm_data.z_vals, envelope_norm, 
                             cmap='jet', vmin=-40, vmax=0, shading='auto')
        ax3.set_title("TFM Reconstruction Field Map (dB)")
        ax3.axvline(0, color='gray', ls='--', alpha=0.5)
        ax3.axhline(Dt0, color='cyan', ls='--', alpha=0.7, label='Interface')
        ax3.axhline(target_z, color='white', ls='--', alpha=0.7, label='Focus')
        ax3.set_xlabel("x (mm)")
        ax3.set_ylabel("z (mm)")
        plt.colorbar(im3, ax=ax3, label='dB')
        ax3.legend(loc='upper right', fontsize=10)
        
        # 4. TFM (Zoom) - with corrected slicing
        ax4 = plt.subplot(2, 3, 5)
        z_mask_tfm = (self.dataset.tfm_data.z_vals > target_z - 30) & (self.dataset.tfm_data.z_vals < target_z + 30)
        
        if np.any(z_mask_tfm):
            # CORRECT SLICING as in original
            zoom_envelope = envelope_norm[z_mask_tfm, :]
            im4 = ax4.pcolormesh(self.dataset.tfm_data.x_vals, self.dataset.tfm_data.z_vals[z_mask_tfm], 
                                 zoom_envelope, cmap='jet', vmin=-40, vmax=0, shading='auto')
            ax4.set_title("TFM - Focal Zone")
            ax4.axvline(0, color='gray', ls='--', alpha=0.5)
            ax4.axhline(target_z, color='white', ls='--', alpha=0.7)
            ax4.set_xlabel("x (mm)")
            ax4.set_ylabel("z (mm)")
        
        # 5. Difference between DT and TFM - matching original
        ax5 = plt.subplot(2, 3, 6)
        
        # Digital Twin grid
        X_dt, Z_dt = np.meshgrid(self.dataset.dt_data.x_vals, self.dataset.dt_data.z_vals, indexing='xy')
        points_dt = np.column_stack((X_dt.ravel(), Z_dt.ravel()))
        values_dt = p_db.ravel()
        
        # TFM grid for interpolation (Target) - using original indexing='ij'
        X_tfm, Z_tfm = np.meshgrid(self.dataset.tfm_data.x_vals, self.dataset.tfm_data.z_vals, indexing='ij')
        points_tfm = np.column_stack((X_tfm.ravel(), Z_tfm.ravel()))
        
        # Interpolate
        p_db_interp_flat = griddata(points_dt, values_dt, points_tfm, method='linear', fill_value=np.nan)
        p_db_interp = p_db_interp_flat.reshape(X_tfm.shape)
        
        mask = ~np.isnan(p_db_interp)
        if np.any(mask):
            diff = np.zeros_like(p_db_interp)
            
            # MATH FIX from original: p_db_interp is (Lat, Depth). envelope_norm is (Depth, Lat).
            # We must transpose envelope_norm back to subtract correctly.
            diff[mask] = p_db_interp[mask] - envelope_norm.T[mask]
            
            # Plot Diff: diff is (Lat, Depth), so we use .T in pcolormesh to make it (Depth, Lat)
            im5 = ax5.pcolormesh(self.dataset.tfm_data.x_vals, self.dataset.tfm_data.z_vals, diff.T, 
                                 cmap='RdBu_r', vmin=-20, vmax=20, shading='auto')
            
            ax5.axvline(0, color='gray', ls='--', alpha=0.5)
            ax5.axhline(Dt0, color='cyan', ls='--', alpha=0.7, label='Interface')
            ax5.axhline(target_z, color='white', ls='--', alpha=0.7, label='Focus')
            ax5.set_title("Difference: DT(dB) - TFM(dB)")
            ax5.set_xlabel("x (mm)")
            ax5.set_ylabel("z (mm)")
            plt.colorbar(im5, ax=ax5, label='dB Difference')
    
    def _save_plot(self) -> None:
        """Save plot to file with original name"""
        if self.fig is None:
            return
        plt.tight_layout()
        plt.savefig(self.output_path, dpi=300, bbox_inches='tight')  # Original used dpi=300


class ProfileComparisonPlot(BasePlot):
    """Profiles comparison plot replicating original plot_profiles_comparison"""
    
    def _setup_figure(self) -> None:
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
    
    def _create_plot(self) -> None:
        # Replicate original plot_profiles_comparison logic
        Dt0 = self.params.dt0
        target_z = self.params.target_z
        
        # Get profiles using original logic
        dt_axial = self.dataset.dt_data.axial_profile()
        dt_lateral = self.dataset.dt_data.lateral_profile_at_z(
            np.argmin(np.abs(self.dataset.dt_data.z_vals - 0.0))
        )
        
        # TFM profiles with original fixes
        tfm_axial = self.dataset.tfm_data.axial_profile()
        tfm_lateral = self.dataset.get_tfm_lateral_profile()
        
        # FIXED: Normalizar pelo máximo global, não local (from original)
        global_max_dt = np.max(self.dataset.dt_data.field_matrix)
        global_max_tfm = np.max(self.dataset.tfm_data.field_matrix)
        
        dt_axial_norm = dt_axial / global_max_dt
        dt_lateral_norm = dt_lateral / global_max_dt
        tfm_axial_norm = tfm_axial / global_max_tfm
        tfm_lateral_norm = tfm_lateral / global_max_tfm
        
        # MIRROR THE DATA (Flip the array order) - from original
        if len(self.dataset.tfm_data.x_vals) == len(self.dataset.tfm_data.z_vals):
            tfm_axial_norm = np.flip(tfm_axial_norm)
        # tfm_lateral_norm = np.flip(tfm_lateral_norm)
        
        # 1. Perfis Axiais (DT vs TFM) - matching original
        ax1 = self.axes[0, 0]
        ax1.plot(self.dataset.dt_data.z_vals, dt_axial_norm, 'b-', linewidth=2, label='Digital Twin')
        ax1.plot(self.dataset.tfm_data.z_vals, tfm_axial_norm, 'r--', linewidth=2, label='TFM Reconstruction')
        ax1.axvline(0, color='gray', ls='--', alpha=0.5)
        ax1.axvline(Dt0, color='gray', linestyle=':', alpha=0.7, label='Interface')
        ax1.axvline(target_z, color='green', linestyle='--', alpha=0.7, label='Target Focus')
        ax1.set_xlabel('Depth z (mm)')
        ax1.set_ylabel('Normalized Amplitude')
        ax1.set_title('Axial Profiles Comparison (x=0)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Perfis Laterais (DT vs TFM) - matching original
        ax2 = self.axes[0, 1]
        ax2.plot(self.dataset.dt_data.x_vals, dt_lateral_norm, 'b-', linewidth=2, label='Digital Twin')
        ax2.plot(self.dataset.tfm_data.x_vals, tfm_lateral_norm, 'r--', linewidth=2, label='TFM Reconstruction')
        ax2.axvline(0, color='gray', linestyle=':', alpha=0.7, label='Center')
        ax2.set_xlabel('Lateral Position x (mm)')
        ax2.set_ylabel('Normalized Amplitude')
        ax2.set_title(f'Lateral Profiles at Focus (z={target_z:.1f}mm)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Zoom do Perfil Axial (região focal) - matching original
        ax3 = self.axes[1, 0]
        z_zoom_min = max(target_z - 20, self.dataset.dt_data.z_vals.min())
        z_zoom_max = min(target_z + 20, self.dataset.dt_data.z_vals.max())
        
        z_mask_dt = (self.dataset.dt_data.z_vals >= z_zoom_min) & (self.dataset.dt_data.z_vals <= z_zoom_max)
        z_mask_tfm = (self.dataset.tfm_data.z_vals >= z_zoom_min) & (self.dataset.tfm_data.z_vals <= z_zoom_max)
        
        if np.any(z_mask_dt) and np.any(z_mask_tfm):
            ax3.plot(self.dataset.dt_data.z_vals[z_mask_dt], dt_axial_norm[z_mask_dt], 'b-', linewidth=2, label='Digital Twin')
            ax3.plot(self.dataset.tfm_data.z_vals[z_mask_tfm], tfm_axial_norm[z_mask_tfm], 'r--', linewidth=2, label='TFM')
            ax3.axvline(target_z, color='green', linestyle='--', alpha=0.7)
            ax3.set_xlabel('Depth z (mm)')
            ax3.set_ylabel('Normalized Amplitude')
            ax3.set_title('Axial Profiles - Focal Region Zoom')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Resíduos Axiais - matching original
        ax4 = self.axes[1, 1]
        if len(self.dataset.tfm_data.z_vals) > 1 and len(tfm_axial_norm) > 1:
            tfm_interp = interp1d(self.dataset.tfm_data.z_vals, tfm_axial_norm, 
                                 kind='linear', bounds_error=False, fill_value=0)
            tfm_on_dt_grid = tfm_interp(self.dataset.dt_data.z_vals)
            
            residuals = dt_axial_norm - tfm_on_dt_grid
            ax4.plot(self.dataset.dt_data.z_vals, residuals, 'k-', linewidth=1.5)
            ax4.axhline(0, color='gray', linestyle='-', alpha=0.5)
            ax4.fill_between(self.dataset.dt_data.z_vals, 0, residuals, where=residuals>=0, 
                            alpha=0.3, color='blue', label='DT > TFM')
            ax4.fill_between(self.dataset.dt_data.z_vals, 0, residuals, where=residuals<0, 
                            alpha=0.3, color='red', label='DT < TFM')
            ax4.set_xlabel('Depth z (mm)')
            ax4.set_ylabel('Residual (DT - TFM)')
            ax4.set_title('Axial Profiles Residuals')
            ax4.legend()
            ax4.grid(True, alpha=0.3)


class FWHMComparisonPlot(BasePlot):
    """FWHM comparison plot replicating original calculate_and_plot_fwhm_comparison"""
    
    def _setup_figure(self) -> None:
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(22, 10))
    
    def _create_plot(self) -> None:
        # Calculate FWHM results using original logic
        results = self._calculate_fwhm_results_with_original_logic()
        
        self._plot_axial_comparison(self.ax1, results)
        self._plot_lateral_comparison(self.ax2, results)
        
        # Add legends from original
        self._add_original_legends()
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Space for legends
    
    def _calculate_fwhm_results_with_original_logic(self) -> Dict[str, Dict[str, float]]:
        """Calculate FWHM using all methods with original logic"""
        Dt0 = self.params.dt0
        
        # Get normalized profiles using original logic
        global_max_dt = np.max(self.dataset.dt_data.field_matrix)
        global_max_tfm = np.max(self.dataset.tfm_data.field_matrix)
        
        dt_axial_norm = self.dataset.dt_data.axial_profile() / global_max_dt
        dt_lateral_norm = self.dataset.dt_data.lateral_profile_at_z(
            np.argmin(np.abs(self.dataset.dt_data.z_vals - self.params.target_z))
        ) / global_max_dt
        
        tfm_axial_norm = self.dataset.tfm_data.axial_profile() / global_max_tfm
        
        # For TFM lateral, use the corrected profile
        tfm_lateral_norm = self.dataset.get_tfm_lateral_profile() / global_max_tfm
        
        # Calcular FWHM para todos os métodos - replicating original
        
        # Axial - Digital Twin (na profundidade focal)
        dt_axial_results = estimate_all_fwhm_methods(
            self.dataset.dt_data.z_vals, dt_axial_norm, 
            theoretical_fwhm=self.params.theoretical_fwhm, 
            show_plot=False
        )
        
        # Axial - TFM with original search mask
        search_mask = (self.dataset.tfm_data.z_vals > Dt0 + 5.0)
        peak_idx = np.where(search_mask, tfm_axial_norm, -np.inf).argmax()
        fwhm_mask = (self.dataset.tfm_data.z_vals >= self.dataset.tfm_data.z_vals[peak_idx] - 25) & \
                    (self.dataset.tfm_data.z_vals <= self.dataset.tfm_data.z_vals[peak_idx] + 25)
        
        tfm_axial_results = estimate_all_fwhm_methods(
            self.dataset.tfm_data.z_vals[fwhm_mask], tfm_axial_norm[fwhm_mask],
            theoretical_fwhm=None, 
            show_plot=False
        )
        
        # Lateral - Digital Twin
        dt_lateral_results = estimate_all_fwhm_methods(
            self.dataset.dt_data.x_vals, dt_lateral_norm,
            theoretical_fwhm=None,
            show_plot=False
        )
        
        # Lateral - TFM
        tfm_lateral_results = estimate_all_fwhm_methods(
            self.dataset.tfm_data.x_vals, tfm_lateral_norm,
            theoretical_fwhm=None,
            show_plot=False
        )
        
        return {
            'dt_axial': dt_axial_results,
            'tfm_axial': tfm_axial_results,
            'dt_lateral': dt_lateral_results,
            'tfm_lateral': tfm_lateral_results
        }
    
    def _plot_axial_comparison(self, ax: plt.Axes, results: Dict) -> None:
        """Plot axial FWHM comparison matching original"""
        methods = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
        width = 0.35
        x = np.arange(len(methods))
        
        # Axial Comparison - COM CORES DO PLOT_FWHM
        dt_axial_vals = [results['dt_axial'].get(m, 0.0) for m in methods]
        tfm_axial_vals = [results['tfm_axial'].get(m, 0.0) for m in methods]
        
        # Cores baseadas na precisão (axial usa referência teórica)
        dt_axial_colors = [classify_result(v, self.params.theoretical_fwhm, is_axial=True)[0] for v in dt_axial_vals]
        tfm_axial_colors = [classify_result(v, self.params.theoretical_fwhm, is_axial=True)[0] for v in tfm_axial_vals]
        
        bars1a = ax.bar(x - width/2, dt_axial_vals, width, 
                       color=dt_axial_colors, hatch='///', 
                       edgecolor='black', alpha=0.7, label='Digital Twin')
        bars1b = ax.bar(x + width/2, tfm_axial_vals, width,
                       color=tfm_axial_colors,
                       edgecolor='black', alpha=0.9, label='TFM')
        
        # Linha teórica (Preta, como no FWHM plot)
        if self.params.theoretical_fwhm > 0:
            ax.axhline(self.params.theoretical_fwhm, color='black', ls='-', lw=3, 
                     label=f'Theoretical Baseline ({self.params.theoretical_fwhm:.2f} mm)')
        
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel('FWHM (mm)', fontweight='bold')
        ax.set_title(f'Axial FWHM: Deviation from Theoretical Value', fontsize=20, fontweight='bold')
        
        # Anotar diferenças absolutas (como no FWHM plot)
        for i, (v_dt, v_tfm) in enumerate(zip(dt_axial_vals, tfm_axial_vals)):
            if self.params.theoretical_fwhm > 0 and v_dt > 0 and v_tfm > 0:
                # Diferença absoluta como no plot FWHM
                diff_dt = v_dt - self.params.theoretical_fwhm
                diff_tfm = v_tfm - self.params.theoretical_fwhm
                ax.text(i - width/2, v_dt + 0.1, f'{diff_dt:+.1f}', 
                       ha='center', fontsize=16, fontweight='bold')
                ax.text(i + width/2, v_tfm + 0.1, f'{diff_tfm:+.1f}', 
                       ha='center', fontsize=16, fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_lateral_comparison(self, ax: plt.Axes, results: Dict) -> None:
        """Plot lateral FWHM comparison matching original"""
        methods = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
        width = 0.35
        x = np.arange(len(methods))
        
        # Lateral Comparison - COM CORES DO PLOT_FWHM
        dt_lateral_vals = [results['dt_lateral'].get(m, 0.0) for m in methods]
        tfm_lateral_vals = [results['tfm_lateral'].get(m, 0.0) for m in methods]
        
        # Para lateral, comparar com Digital Twin como referência
        dt_lateral_colors = ['steelblue'] * len(methods)  # Azul padrão para DT como no FWHM plot
        tfm_lateral_colors = []
        for i, v_tfm in enumerate(tfm_lateral_vals):
            v_dt_ref = dt_lateral_vals[i] if dt_lateral_vals[i] > 0 else None
            color, _ = classify_result(v_tfm, v_dt_ref, is_axial=False)
            tfm_lateral_colors.append(color)
        
        bars2a = ax.bar(x - width/2, dt_lateral_vals, width,
                       color=dt_lateral_colors, hatch='///',
                       edgecolor='black', alpha=0.7, label='Digital Twin')
        bars2b = ax.bar(x + width/2, tfm_lateral_vals, width,
                       color=tfm_lateral_colors,
                       edgecolor='black', alpha=0.9, label='TFM')
        
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel('FWHM (mm)', fontweight='bold')
        ax.set_title('Lateral FWHM: TFM vs Digital Twin Reference', fontsize=20, fontweight='bold')
        
        # Anotar diferenças percentuais
        for i, (v_dt, v_tfm) in enumerate(zip(dt_lateral_vals, tfm_lateral_vals)):
            if v_dt > 0 and v_tfm > 0:
                diff_pct = ((v_tfm - v_dt) / v_dt) * 100
                ax.text(i, max(v_dt, v_tfm) + 0.5, f'{diff_pct:+.1f}%', 
                       ha='center', fontweight='bold', color='darkred', fontsize=16)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    def _add_original_legends(self) -> None:
        """Add legends matching original"""
        # LEGENDA DO LADO DIREITO (como no FWHM plot)
        self.ax1.legend().remove()
        legend_elements_axial = create_legend_elements(include_theoretical=True, include_patterns=True)
        self.ax1.legend(handles=legend_elements_axial, loc='upper left', bbox_to_anchor=(1.05, 1), 
                       fontsize=14, frameon=True, shadow=True)
        
        # LEGENDA DO LADO DIREITO (sem linha teórica para lateral)
        self.ax2.legend().remove()
        legend_elements_lateral = create_legend_elements(include_theoretical=False, include_patterns=True)
        self.ax2.legend(handles=legend_elements_lateral, loc='upper left', bbox_to_anchor=(1.05, 1), 
                       fontsize=14, frameon=True, shadow=True)


class DebugLateralPlot(BasePlot):
    """Debug plot for lateral profile extraction matching original plot_debug_lateral"""
    
    def _setup_figure(self) -> None:
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
    
    def _create_plot(self) -> None:
        # Replicate original plot_debug_lateral logic
        Dt0 = self.params.dt0
        target_z = self.params.target_z
        
        # Find multiple depths around focus
        depth_indices = []
        depth_labels = []
        
        # Target depth
        iz_target = np.argmin(np.abs(self.dataset.tfm_data.z_vals - target_z))
        depth_indices.append(iz_target)
        depth_labels.append(f"Target z={target_z:.1f}mm")
        
        # Actual peak
        mask_post_interface = self.dataset.tfm_data.z_vals > Dt0 + 5.0
        if np.any(mask_post_interface):
            valid_indices = np.where(mask_post_interface)[0]
            tfm_axial = self.dataset.tfm_data.axial_profile()
            peak_idx = valid_indices[np.argmax(tfm_axial[valid_indices])]
            depth_indices.append(peak_idx)
            depth_labels.append(f"Peak z={self.dataset.tfm_data.z_vals[peak_idx]:.1f}mm")
        
        # ±5mm from target
        for offset in [-5, 5]:
            z_test = target_z + offset
            if z_test >= self.dataset.tfm_data.z_vals.min() and z_test <= self.dataset.tfm_data.z_vals.max():
                iz = np.argmin(np.abs(self.dataset.tfm_data.z_vals - z_test))
                depth_indices.append(iz)
                depth_labels.append(f"z={z_test:.1f}mm")
        
        # 1. TFM 2D map with depth lines (Corrected Orientation)
        ax1 = self.axes[0, 0]
        envelope_norm = 20 * np.log10(self.dataset.tfm_data.normalized_matrix() + 1e-6)
        
        # Apply original SMART LOGIC
        if len(self.dataset.tfm_data.x_vals) != len(self.dataset.tfm_data.z_vals):
            envelope_norm = envelope_norm.T
        
        im1 = ax1.pcolormesh(self.dataset.tfm_data.x_vals, self.dataset.tfm_data.z_vals, envelope_norm, 
                            cmap='jet', vmin=-40, vmax=0, shading='auto')
        ax1.axvline(0, color='gray', ls='--', alpha=0.5)
        ax1.axhline(Dt0, color='white', ls='--', alpha=0.7, label='Interface')
        ax1.axhline(target_z, color='cyan', ls='--', alpha=0.7, label='Focus')
        
        ax1.set_xlabel('Lateral x (mm)')
        ax1.set_ylabel('Depth z (mm)')
        ax1.set_title('TFM Reconstruction Field Map (dB)')
        plt.colorbar(im1, ax=ax1, label='dB')
        ax1.legend(fontsize=10)
        
        # 2. Axial profile at x=0
        ax2 = self.axes[0, 1]
        tfm_axial = self.dataset.tfm_data.axial_profile()
        ax2.plot(self.dataset.tfm_data.z_vals, tfm_axial, 'b-', linewidth=2)
        for iz, label in zip(depth_indices, depth_labels):
            ax2.axvline(self.dataset.tfm_data.z_vals[iz], color='red', ls='--', alpha=0.7)
            ax2.plot(self.dataset.tfm_data.z_vals[iz], tfm_axial[iz], 'ro', markersize=8)
        ax2.set_xlabel('Depth z (mm)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('TFM Axial Profile (x=0)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Lateral profiles at different depths
        ax3 = self.axes[1, 0]
        for iz, label in zip(depth_indices, depth_labels):
            lateral_profile = self.dataset.tfm_data.lateral_profile_at_z(iz)
            lateral_norm = lateral_profile / np.max(lateral_profile) if np.max(lateral_profile) > 0 else lateral_profile
            # MIRROR THE DATA (Flip the array order) - from original
            lateral_norm = np.flip(lateral_norm)
            ax3.plot(self.dataset.tfm_data.x_vals, lateral_norm, label=label, linewidth=2)
        
        ax3.set_xlabel('Lateral x (mm)')
        ax3.set_ylabel('Normalized Amplitude')
        ax3.set_title('Lateral Profiles at Different Depths')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Find peak positions in lateral profiles
        ax4 = self.axes[1, 1]
        peak_positions = []
        depths = []
        
        for iz, label in zip(depth_indices, depth_labels):
            lateral_profile = self.dataset.tfm_data.lateral_profile_at_z(iz)
            
            # MIRROR THE DATA (Flip the array order) - from original
            lateral_profile = np.flip(lateral_profile)
            
            peak_idx = np.argmax(lateral_profile)
            peak_x = self.dataset.tfm_data.x_vals[peak_idx]
            peak_positions.append(peak_x)
            depths.append(self.dataset.tfm_data.z_vals[iz])
            
            ax4.plot(peak_x, self.dataset.tfm_data.z_vals[iz], 'o', markersize=10, label=f"{label}: x={peak_x:.1f}mm")
        
        ax4.set_xlabel('Peak Lateral Position (mm)')
        ax4.set_ylabel('Depth (mm)')
        ax4.set_title('Peak Position vs Depth')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Print summary as in original
        print("\n[DEBUG] Lateral Profile Peak Positions:")
        for iz, label, peak_x in zip(depth_indices, depth_labels, peak_positions):
            print(f"  {label}: peak at x = {peak_x:.1f} mm")


class RadarChartPlot(BasePlot):
    """Radar chart for FWHM methods matching original"""
    
    def _setup_figure(self) -> None:
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='polar')
    
    def _create_plot(self) -> None:
        # Reuse FWHM calculation from FWHMComparisonPlot
        fwhm_plot = FWHMComparisonPlot(self.dataset, self.params, self.config)
        results = fwhm_plot._calculate_fwhm_results_with_original_logic()
        
        methods = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
        
        # Extract values
        dt_axial_vals = [results['dt_axial'].get(m, 0.0) for m in methods]
        tfm_axial_vals = [results['tfm_axial'].get(m, 0.0) for m in methods]
        dt_lateral_vals = [results['dt_lateral'].get(m, 0.0) for m in methods]
        tfm_lateral_vals = [results['tfm_lateral'].get(m, 0.0) for m in methods]
        
        # Valores normalizados (0-1) para melhor visualização (from original)
        def normalize_vals(vals):
            max_val = max(v for v in vals if v > 0)
            return [v/max_val if v > 0 else 0 for v in vals]
        
        dt_axial_norm = normalize_vals(dt_axial_vals)
        tfm_axial_norm = normalize_vals(tfm_axial_vals)
        dt_lateral_norm = normalize_vals(dt_lateral_vals)
        tfm_lateral_norm = normalize_vals(tfm_lateral_vals)
        
        # Convert to polar coordinates as in original
        angles = np.linspace(0, 2*np.pi, len(methods), endpoint=False).tolist()
        angles += angles[:1]  # Fechar o polígono
        
        # Fechar polígonos
        dt_axial_norm += dt_axial_norm[:1]
        tfm_axial_norm += tfm_axial_norm[:1]
        dt_lateral_norm += dt_lateral_norm[:1]
        tfm_lateral_norm += tfm_lateral_norm[:1]
        
        # Plotar as in original
        self.ax.plot(angles, dt_axial_norm, 'b-', linewidth=2, label='DT Axial')
        self.ax.fill(angles, dt_axial_norm, 'b', alpha=0.1)
        
        self.ax.plot(angles, tfm_axial_norm, 'r-', linewidth=2, label='TFM Axial')
        self.ax.fill(angles, tfm_axial_norm, 'r', alpha=0.1)
        
        self.ax.plot(angles, dt_lateral_norm, 'g--', linewidth=2, label='DT Lateral')
        self.ax.fill(angles, dt_lateral_norm, 'g', alpha=0.1)
        
        self.ax.plot(angles, tfm_lateral_norm, 'm--', linewidth=2, label='TFM Lateral')
        self.ax.fill(angles, tfm_lateral_norm, 'm', alpha=0.1)
        
        # Configurações as in original
        self.ax.set_xticks(angles[:-1])
        self.ax.set_xticklabels(methods)
        self.ax.set_ylim(0, 1.2)
        self.ax.set_title('FWHM Methods Radar Chart (Normalized)', fontsize=16, pad=20)
        self.ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        self.ax.grid(True)


class OnePageSummaryPlot(BasePlot):
    """One-page summary plot matching original create_one_page_summary"""
    
    def _setup_figure(self) -> None:
        self.fig = plt.figure(figsize=(20, 15))
    
    def _create_plot(self) -> None:
        # Calculate metrics using original logic
        metrics = self._calculate_metrics_with_original_logic()
        
        # Create 3x3 grid as in original
        gs = self.fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)
        
        # 1. TFM 2D Map (top left) - matching original
        ax1 = self.fig.add_subplot(gs[0, 0])
        self._plot_tfm_map_with_original_logic(ax1)
        
        # 2. Axial Profile Comparison (top middle) - matching original
        ax2 = self.fig.add_subplot(gs[0, 1])
        self._plot_axial_profiles_with_original_logic(ax2)
        
        # 3. Lateral Profile Comparison (top right) - matching original
        ax3 = self.fig.add_subplot(gs[0, 2])
        self._plot_lateral_profiles_with_original_logic(ax3)
        
        # 4. Axial FWHM Comparison (middle left) - matching original
        ax4 = self.fig.add_subplot(gs[1, 0])
        self._plot_axial_fwhm_with_original_logic(ax4)
        
        # 5. Lateral FWHM Comparison (middle middle) - matching original
        ax5 = self.fig.add_subplot(gs[1, 1])
        self._plot_lateral_fwhm_with_original_logic(ax5)
        
        # 6. FWHM Ratio (middle right) - matching original
        ax6 = self.fig.add_subplot(gs[1, 2])
        self._plot_fwhm_ratio_with_original_logic(ax6)
        
        # 7. Text Summary (bottom row, spanning all columns) - matching original
        ax7 = self.fig.add_subplot(gs[2, :])
        self._plot_text_summary_with_original_logic(ax7, metrics)
        
        # Add overall title
        self.fig.suptitle('Enhanced 2D Pipeline Analysis Summary', fontsize=20, fontweight='bold', y=0.98)
    
    def _calculate_metrics_with_original_logic(self) -> Dict:
        """Calculate key metrics using original logic"""
        # Get FWHM results
        fwhm_plot = FWHMComparisonPlot(self.dataset, self.params, self.config)
        fwhm_results = fwhm_plot._calculate_fwhm_results_with_original_logic()
        methods = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
        
        # TFM peak using original logic
        search_mask = (self.dataset.tfm_data.z_vals > self.params.dt0 + 5.0)
        tfm_axial = self.dataset.tfm_data.axial_profile()
        peak_idx = np.where(search_mask, tfm_axial, -np.inf).argmax()
        peak_z = self.dataset.tfm_data.z_vals[peak_idx]
        peak_value = tfm_axial[peak_idx]
        
        # FWHM médios as in original
        tfm_axial_vals = [fwhm_results['tfm_axial'].get(m, 0.0) for m in methods]
        tfm_axial_vals_valid = [v for v in tfm_axial_vals if v > 0]
        tfm_axial_mean = np.mean(tfm_axial_vals_valid) if tfm_axial_vals_valid else 0
        
        tfm_lateral_vals = [fwhm_results['tfm_lateral'].get(m, 0.0) for m in methods]
        tfm_lateral_vals_valid = [v for v in tfm_lateral_vals if v > 0]
        tfm_lateral_mean = np.mean(tfm_lateral_vals_valid) if tfm_lateral_vals_valid else 0
        
        # Calcular erros as in original
        depth_error = abs(peak_z - self.params.target_z) / self.params.target_z * 100 if self.params.target_z > 0 else 0
        fwhm_error = abs(tfm_axial_mean - self.params.theoretical_fwhm) / self.params.theoretical_fwhm * 100 if self.params.theoretical_fwhm > 0 else 0
        
        # Calcular razão lateral/axial (com verificação de divisão por zero) as in original
        if tfm_axial_mean > 0:
            lateral_axial_ratio = tfm_lateral_mean / tfm_axial_mean
        else:
            lateral_axial_ratio = 0
        
        return {
            'peak_z': peak_z,
            'peak_value': peak_value,
            'tfm_axial_mean': tfm_axial_mean,
            'tfm_lateral_mean': tfm_lateral_mean,
            'depth_error': depth_error,
            'fwhm_error': fwhm_error,
            'lateral_axial_ratio': lateral_axial_ratio
        }
    
    def _plot_tfm_map_with_original_logic(self, ax: plt.Axes) -> None:
        """Plot TFM 2D map with original SMART LOGIC"""
        envelope_norm = 20 * np.log10(self.dataset.tfm_data.normalized_matrix() + 1e-6)
        
        # Apply original SMART LOGIC
        if len(self.dataset.tfm_data.x_vals) != len(self.dataset.tfm_data.z_vals):
            envelope_norm = envelope_norm.T
        
        im = ax.pcolormesh(self.dataset.tfm_data.x_vals, self.dataset.tfm_data.z_vals, envelope_norm, 
                          cmap='jet', vmin=-40, vmax=0, shading='auto')
        ax.axvline(0, color='gray', ls='--', alpha=0.5)
        ax.axhline(self.params.dt0, color='white', ls='--', alpha=0.7, label='Interface')
        ax.axhline(self.params.target_z, color='cyan', ls='--', alpha=0.7, label='Focus')
        
        ax.set_xlabel('Lateral x (mm)')
        ax.set_ylabel('Depth z (mm)')
        ax.set_title('TFM Reconstruction Field Map (dB)')
        plt.colorbar(im, ax=ax, label='dB', fraction=0.046, pad=0.04)
        ax.legend(fontsize=8, loc='upper right')
    
    def _plot_axial_profiles_with_original_logic(self, ax: plt.Axes) -> None:
        """Plot axial profiles with original normalization and flipping"""
        global_max_dt = np.max(self.dataset.dt_data.field_matrix)
        global_max_tfm = np.max(self.dataset.tfm_data.field_matrix)
        
        dt_axial_norm = self.dataset.dt_data.axial_profile() / global_max_dt
        tfm_axial_norm = self.dataset.tfm_data.axial_profile() / global_max_tfm
        
        # MIRROR THE DATA (Flip the array order) - from original
        if len(self.dataset.tfm_data.x_vals) == len(self.dataset.tfm_data.z_vals):
            tfm_axial_norm = np.flip(tfm_axial_norm)
        
        ax.plot(self.dataset.dt_data.z_vals, dt_axial_norm, 'b-', linewidth=2, label='Digital Twin')
        ax.plot(self.dataset.tfm_data.z_vals, tfm_axial_norm, 'r--', linewidth=2, label='TFM')
        ax.axvline(self.params.target_z, color='green', ls=':', alpha=0.7, label='Target')
        ax.set_xlabel('Depth (mm)')
        ax.set_ylabel('Normalized Amplitude')
        ax.set_title('Axial Profiles (x=0)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_lateral_profiles_with_original_logic(self, ax: plt.Axes) -> None:
        """Plot lateral profiles with original normalization and flipping"""
        global_max_dt = np.max(self.dataset.dt_data.field_matrix)
        global_max_tfm = np.max(self.dataset.tfm_data.field_matrix)
        
        dt_lateral_norm = self.dataset.dt_data.lateral_profile_at_z(
            np.argmin(np.abs(self.dataset.dt_data.z_vals - self.params.target_z))
        ) / global_max_dt
        
        tfm_lateral_norm = self.dataset.get_tfm_lateral_profile() / global_max_tfm
        
        ax.plot(self.dataset.dt_data.x_vals, dt_lateral_norm, 'b-', linewidth=2, label='Digital Twin')
        ax.plot(self.dataset.tfm_data.x_vals, tfm_lateral_norm, 'r--', linewidth=2, label='TFM')
        ax.axvline(0, color='gray', ls=':', alpha=0.5, label='Center')
        ax.set_xlabel('Lateral Position (mm)')
        ax.set_ylabel('Normalized Amplitude')
        ax.set_title(f'Lateral Profiles (z={self.params.target_z:.1f}mm)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_axial_fwhm_with_original_logic(self, ax: plt.Axes) -> None:
        """Plot axial FWHM with original logic"""
        fwhm_plot = FWHMComparisonPlot(self.dataset, self.params, self.config)
        fwhm_results = fwhm_plot._calculate_fwhm_results_with_original_logic()
        methods = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
        
        dt_vals = [fwhm_results['dt_axial'].get(m, 0.0) for m in methods]
        tfm_vals = [fwhm_results['tfm_axial'].get(m, 0.0) for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax.bar(x - width/2, dt_vals, width, color='steelblue', alpha=0.6, label='DT')
        ax.bar(x + width/2, tfm_vals, width, color='seagreen', alpha=0.6, label='TFM')
        
        if self.params.theoretical_fwhm > 0:
            ax.axhline(self.params.theoretical_fwhm, color='black', ls='--', 
                      linewidth=2, label=f'Theory: {self.params.theoretical_fwhm:.2f}mm')
        
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45)
        ax.set_ylabel('FWHM (mm)')
        ax.set_title('Axial FWHM Methods')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_lateral_fwhm_with_original_logic(self, ax: plt.Axes) -> None:
        """Plot lateral FWHM with original logic"""
        fwhm_plot = FWHMComparisonPlot(self.dataset, self.params, self.config)
        fwhm_results = fwhm_plot._calculate_fwhm_results_with_original_logic()
        methods = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
        
        dt_vals = [fwhm_results['dt_lateral'].get(m, 0.0) for m in methods]
        tfm_vals = [fwhm_results['tfm_lateral'].get(m, 0.0) for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax.bar(x - width/2, dt_vals, width, color='steelblue', alpha=0.6, label='DT')
        ax.bar(x + width/2, tfm_vals, width, color='seagreen', alpha=0.6, label='TFM')
        
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45)
        ax.set_ylabel('FWHM (mm)')
        ax.set_title('Lateral FWHM Methods')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_fwhm_ratio_with_original_logic(self, ax: plt.Axes) -> None:
        """Plot FWHM ratio with original coloring logic"""
        fwhm_plot = FWHMComparisonPlot(self.dataset, self.params, self.config)
        fwhm_results = fwhm_plot._calculate_fwhm_results_with_original_logic()
        methods = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
        
        tfm_axial_vals = [fwhm_results['tfm_axial'].get(m, 0.0) for m in methods]
        tfm_lateral_vals = [fwhm_results['tfm_lateral'].get(m, 0.0) for m in methods]
        
        ratios = []
        for i in range(len(methods)):
            if tfm_axial_vals[i] > 0:
                ratio = tfm_lateral_vals[i] / tfm_axial_vals[i]
                ratios.append(ratio)
            else:
                ratios.append(0)
        
        x = np.arange(len(methods))
        bars = ax.bar(x, ratios, color='purple', alpha=0.7)
        ax.axhline(1.0, color='red', ls='--', alpha=0.5, label='Ideal (1:1)')
        
        # Colorir barras baseado no desvio do ideal - from original
        for i, bar in enumerate(bars):
            if ratios[i] > 1.5:
                bar.set_color('red')
            elif ratios[i] > 1.2:
                bar.set_color('orange')
            elif ratios[i] >= 0.8:
                bar.set_color('green')
            else:
                bar.set_color('blue')
        
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45)
        ax.set_ylabel('Ratio (Lateral/Axial)')
        ax.set_title('FWHM Lateral/Axial Ratio')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_text_summary_with_original_logic(self, ax: plt.Axes, metrics: Dict) -> None:
        """Plot text summary with original quality assessment"""
        ax.axis('off')
        
        # Determine quality assessment as in original
        if metrics['depth_error'] < 5 and metrics['fwhm_error'] < 15:
            quality = "✅ EXCELLENT: Both depth and FWHM within specifications"
        elif metrics['depth_error'] < 10 and metrics['fwhm_error'] < 25:
            quality = "⚠️ GOOD: Minor deviations from target"
        elif metrics['depth_error'] < 20 or metrics['fwhm_error'] < 40:
            quality = "⚠️ MODERATE: Significant deviations detected"
        else:
            quality = "❌ POOR: Large deviations from target"
        
        summary_text = (
            f"PIPELINE ANALYSIS SUMMARY\n\n"
            f"Target Parameters:\n"
            f"  • Focus Depth: {self.params.target_z:.1f} mm\n"
            f"  • Theoretical FWHM: {self.params.theoretical_fwhm:.2f} mm\n\n"
            f"TFM Results:\n"
            f"  • Peak Depth: {metrics['peak_z']:.1f} mm ({metrics['depth_error']:.1f}% error)\n"
            f"  • Axial FWHM: {metrics['tfm_axial_mean']:.2f} mm ({metrics['fwhm_error']:.1f}% error)\n"
            f"  • Lateral FWHM: {metrics['tfm_lateral_mean']:.2f} mm\n"
            f"  • Lateral/Axial Ratio: {metrics['lateral_axial_ratio']:.2f}\n\n"
            f"Quality Assessment:\n"
            f"  • {quality}"
        )
        
        ax.text(0.02, 0.95, summary_text, fontsize=14, 
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# =============================================================================
# REPORT GENERATOR (REPLICATING ORIGINAL)
# =============================================================================

class ReportGenerator:
    """Generates comprehensive CSV report matching original generate_comprehensive_report"""
    
    @staticmethod
    def generate(dataset: PipelineDataset, params: PipelineParameters, 
                 fwhm_results: Dict, output_dir: Path) -> None:
        """Generate comprehensive analysis report matching original"""
        report_path = output_dir / "comprehensive_analysis_report.csv"
        
        # Calculate additional metrics using original logic
        Dt0 = params.dt0
        target_z = params.target_z
        theoretical_fwhm = params.theoretical_fwhm
        
        # Encontrar pico no TFM as in original
        search_mask = (dataset.tfm_data.z_vals > Dt0 + 5.0)
        tfm_axial = dataset.tfm_data.axial_profile()
        peak_idx = np.where(search_mask, tfm_axial, -np.inf).argmax()
        peak_z = dataset.tfm_data.z_vals[peak_idx]
        peak_value = tfm_axial[peak_idx]
        
        with open(report_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Cabeçalho as in original
            writer.writerow(['Comprehensive Pipeline Analysis Report'])
            writer.writerow(['Generated by: Enhanced 2D Pipeline Plotter'])
            writer.writerow([''])
            writer.writerow(['Pipeline Parameters'])
            writer.writerow(['Parameter', 'Value', 'Units'])
            for key, value in params.raw_params.items():
                if isinstance(value, (int, float, str)):
                    writer.writerow([key, value, ''])
            writer.writerow([''])
            
            # Seção de métricas gerais as in original
            writer.writerow(['General Metrics'])
            writer.writerow(['Metric', 'Value', 'Units', 'Notes'])
            writer.writerow(['TFM Peak Depth', f'{peak_z:.2f}', 'mm', 
                            f'Target was {target_z:.1f} mm'])
            writer.writerow(['TFM Peak Value', f'{peak_value:.4f}', '', ''])
            writer.writerow(['Theoretical FWHM', f'{theoretical_fwhm:.4f}', 'mm', ''])
            writer.writerow(['Digital Twin FWHM (F1)', 
                            f"{fwhm_results['dt_axial'].get('F1', 0.0):.4f}", 
                            'mm', 'Axial at focus'])
            writer.writerow([''])
            
            # Seção de FWHM detalhada as in original
            writer.writerow(['FWHM Analysis'])
            writer.writerow(['Method', 'DT Axial (mm)', 'TFM Axial (mm)', 
                            'DT Lateral (mm)', 'TFM Lateral (mm)', 
                            'Axial Error vs Theory (%)', 'Lateral Error vs DT (%)',
                            'Axial Quality', 'Lateral Quality'])
            
            methods = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
            for method in methods:
                row = ReportGenerator._create_fwhm_row_original(method, fwhm_results, params)
                writer.writerow(row)
            
            writer.writerow([''])
            
            # Resumo estatístico as in original
            writer.writerow(['Statistical Summary'])
            writer.writerow(['Metric', 'Mean', 'Std Dev', 'Min', 'Max', 'Units'])
            
            # Calcular estatísticas para TFM Axial as in original
            tfm_axial_vals = [fwhm_results['tfm_axial'].get(m, 0.0) for m in methods]
            tfm_axial_vals = [v for v in tfm_axial_vals if v > 0]
            
            if tfm_axial_vals:
                writer.writerow([
                    'TFM Axial FWHM',
                    f'{np.mean(tfm_axial_vals):.4f}',
                    f'{np.std(tfm_axial_vals):.4f}',
                    f'{np.min(tfm_axial_vals):.4f}',
                    f'{np.max(tfm_axial_vals):.4f}',
                    'mm'
                ])
            
            # Calcular estatísticas para TFM Lateral as in original
            tfm_lateral_vals = [fwhm_results['tfm_lateral'].get(m, 0.0) for m in methods]
            tfm_lateral_vals = [v for v in tfm_lateral_vals if v > 0]
            
            if tfm_lateral_vals:
                writer.writerow([
                    'TFM Lateral FWHM',
                    f'{np.mean(tfm_lateral_vals):.4f}',
                    f'{np.std(tfm_lateral_vals):.4f}',
                    f'{np.min(tfm_lateral_vals):.4f}',
                    f'{np.max(tfm_lateral_vals):.4f}',
                    'mm'
                ])
            
            # Razão Lateral/Axial as in original
            if tfm_axial_vals and tfm_lateral_vals:
                ratio_mean = np.mean(tfm_lateral_vals) / np.mean(tfm_axial_vals) if np.mean(tfm_axial_vals) > 0 else 0
                writer.writerow([
                    'Lateral/Axial Ratio',
                    f'{ratio_mean:.4f}',
                    '', '', '',
                    ''
                ])
        
        print(f"✅ Comprehensive report saved to: {report_path}")
    
    @staticmethod
    def _create_fwhm_row_original(method: str, fwhm_results: Dict, params: PipelineParameters) -> List:
        """Create a row for FWHM analysis table matching original logic"""
        dt_axial = fwhm_results['dt_axial'].get(method, 0.0)
        tfm_axial = fwhm_results['tfm_axial'].get(method, 0.0)
        dt_lateral = fwhm_results['dt_lateral'].get(method, 0.0)
        tfm_lateral = fwhm_results['tfm_lateral'].get(method, 0.0)
        
        # Calcular erros as in original
        axial_error = 0.0
        if params.theoretical_fwhm > 0 and tfm_axial > 0:
            axial_error = ((tfm_axial - params.theoretical_fwhm) / params.theoretical_fwhm) * 100
        
        lateral_error = 0.0
        if dt_lateral > 0 and tfm_lateral > 0:
            lateral_error = ((tfm_lateral - dt_lateral) / dt_lateral) * 100
        
        # Classificar qualidade usando o mesmo esquema as in original
        axial_color, axial_quality = classify_result(tfm_axial, params.theoretical_fwhm, is_axial=True)
        lateral_color, lateral_quality = classify_result(tfm_lateral, dt_lateral, is_axial=False)
        
        return [
            method,
            f'{dt_axial:.4f}',
            f'{tfm_axial:.4f}',
            f'{dt_lateral:.4f}',
            f'{tfm_lateral:.4f}',
            f'{axial_error:.2f}',
            f'{lateral_error:.2f}',
            axial_quality,
            lateral_quality
        ]


# =============================================================================
# PIPELINE CONTROLLER
# =============================================================================

class PipelineController:
    """Main controller for the plotting pipeline matching original main()"""
    
    def __init__(self, root_dir: Path, output_dir: Path):
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        ensure_dir(str(self.output_dir))
        
        # Initialize components
        self.dataset = None
        self.params = None
        self.fwhm_results = None
        
    def run(self) -> None:
        """Execute the complete plotting pipeline matching original main()"""
        print(f"{'='*60}")
        print("Enhanced 2D Pipeline Results Plotter (Refactored with Original Fixes)")
        print(f"{'='*60}")
        
        try:
            # 1. Localizar arquivo de parâmetros as in original
            param_file = self._find_parameter_file_original()
            print(f"[INFO] Using parameters from: {param_file}")
            
            # 2. Carregar parâmetros as in original
            with open(param_file, "r", encoding='utf-8') as f:
                params_dict = json.load(f)
            self.params = PipelineParameters(**params_dict)
            
            # 3. Preparar diretório de saída as in original
            ensure_dir("plots")
            
            # 4. Carregar dados as in original
            self.dataset = DataLoader.load_from_directory(self.root_dir)
            
            # 5. Gerar todos os plots as in original
            self._generate_all_plots_original()
            
            # 6. Gerar relatório as in original
            if self.fwhm_results:
                ReportGenerator.generate(self.dataset, self.params, self.fwhm_results, self.output_dir)
            
            print(f"\n{'='*60}")
            print("✅ PROCESSING COMPLETE!")
            print(f"{'='*60}")
            print("Generated files in 'plots/' directory:")
            print("  1. field_maps_2d_comparison.png - Full 2D field maps")
            print("  2. profiles_comparison_2d.png - Axial/Lateral profiles")
            print("  3. fwhm_axial_vs_lateral_comparison.png - FWHM comparison")
            print("  4. fwhm_methods_radar_chart.png - Radar chart of methods")
            print("  5. comprehensive_analysis_report.csv - Full analysis report")
            print("  6. one_page_summary.png - Quick overview")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def _find_parameter_file_original(self) -> Path:
        """Find the run_params.json file using original logic"""
        possible_paths = [
            self.root_dir / "run_params.json",
            Path("run_params.json")
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        raise FileNotFoundError(f"run_params.json not found in any expected location.")
    
    def _generate_all_plots_original(self) -> None:
        """Generate all plots from the original script"""
        plot_config = PlotConfig(figsize=(20, 12), dpi=300, font_size=18)
        
        # List of all plots to generate (matching original script)
        plots_to_generate = [
            ('field_maps_2d_comparison', FieldMapPlot),
            ('debug_lateral_extraction', DebugLateralPlot),
            ('profiles_comparison_2d', ProfileComparisonPlot),
            ('fwhm_axial_vs_lateral_comparison', FWHMComparisonPlot),
            ('fwhm_methods_radar_chart', RadarChartPlot),
            ('one_page_summary', OnePageSummaryPlot),
        ]
        
        for filename, plot_class in plots_to_generate:
            print(f"[PLOT] Generating {filename}...")
            plot = plot_class(self.dataset, self.params, plot_config)
            output_path = self.output_dir / f"{filename}.png"
            plot.generate(output_path)
            
            # Store FWHM results for report (from FWHM comparison plot)
            if filename == 'fwhm_axial_vs_lateral_comparison':
                self.fwhm_results = plot._calculate_fwhm_results_with_original_logic()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point matching original"""
    if len(sys.argv) < 2:
        print("Usage: python plot_pipeline_results.py <results_dir> [output_dir]")
        print("Example: python plot_pipeline_results.py results/final/test_20 plots")
        sys.exit(1)
    
    root_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("plots")
    
    # Validate input directory
    if not root_dir.exists():
        print(f"❌ Results directory not found: {root_dir}")
        sys.exit(1)
    
    # Run pipeline
    controller = PipelineController(root_dir, output_dir)
    controller.run()


if __name__ == "__main__":
    main()