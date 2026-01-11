#!/usr/bin/env python3
"""
Enhanced Digital Twin → FMC → TFM Pipeline with Checkpoint Recovery
Memory-optimized with batch processing and fault tolerance
"""

import sys
import os
import numpy as np
import argparse
import warnings
import json
import hashlib
import h5py
import tempfile
import shutil
import pickle
import time
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import psutil

# Block deprecation warnings
warnings.filterwarnings("ignore")

# Add service paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from application.delay_laws3Dint_service import run_delay_laws3Dint_service
from application.mps_array_model_int_service import run_mps_array_model_int_service
from application.discrete_windows_service import run_discrete_windows_service
from interface.cli_utils import safe_float

# =============================================================================
# CHECKPOINT MANAGER - Core Infrastructure
# =============================================================================

class CheckpointManager:
    """Manages pipeline state persistence and recovery"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.state_file = self.checkpoint_dir / "pipeline_state.json"
        self.checkpoint_version = "1.0"
        
    def init_checkpoint(self, args: argparse.Namespace) -> Dict:
        """Initialize fresh checkpoint state"""
        return {
            "version": self.checkpoint_version,
            "created": datetime.utcnow().isoformat(),
            "parameters": vars(args),
            "stages": {
                "digital_twin": {"status": "pending", "checksum": None},
                "fmc_generation": {
                    "status": "pending",
                    "completed_transmitters": [],
                    "last_batch": 0,
                    "batches": {}
                },
                "tfm_reconstruction": {"status": "pending", "checksum": None}
            },
            "current_stage": "digital_twin",
            "memory_footprint_mb": 0.0
        }
    
    def save_state(self, state: Dict) -> bool:
        """Atomically save pipeline state with checksum"""
        try:
            # Compute checksum of critical data
            state["state_checksum"] = self._compute_state_checksum(state)
            
            # Write to temporary file first
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Atomic rename
            # For Windows compatibility - replace if exists
            if os.name == 'nt':  # Windows
                if self.state_file.exists():
                    os.replace(str(temp_file), str(self.state_file))
                else:
                    temp_file.rename(self.state_file)
            else:
                temp_file.rename(self.state_file)
            
            # Update memory footprint
            state["memory_footprint_mb"] = psutil.Process().memory_info().rss / 1e6
            return True
        except Exception as e:
            print(f"[WARN] Failed to save checkpoint state: {e}")
            return False
    
    def load_state(self) -> Optional[Dict]:
        """Load and validate checkpoint state"""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Validate checksum
            saved_checksum = state.pop("state_checksum", None)
            computed_checksum = self._compute_state_checksum(state)
            
            if saved_checksum != computed_checksum:
                print("[WARN] Checkpoint checksum mismatch - state may be corrupted")
                return None
            
            return state
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}")
            return None
    
    def mark_stage_complete(self, state: Dict, stage: str, data_checksum: str = None):
        """Mark a stage as completed"""
        state["stages"][stage]["status"] = "completed"
        state["stages"][stage]["completed_time"] = datetime.utcnow().isoformat()
        if data_checksum:
            state["stages"][stage]["checksum"] = data_checksum
        state["current_stage"] = self._get_next_stage(stage)
    
    def _compute_state_checksum(self, state: Dict) -> str:
        """Compute SHA256 checksum of critical state data"""
        # Exclude volatile fields from checksum
        check_data = {
            "version": state.get("version"),
            "parameters": state.get("parameters", {}),
            "stages": {
                stage: {
                    "status": info.get("status"),
                    "checksum": info.get("checksum"),
                    "completed_transmitters": info.get("completed_transmitters", [])
                }
                for stage, info in state.get("stages", {}).items()
            }
        }
        return hashlib.sha256(json.dumps(check_data, sort_keys=True).encode()).hexdigest()
    
    def _get_next_stage(self, current: str) -> str:
        """Determine next pipeline stage"""
        stages = ["digital_twin", "fmc_generation", "tfm_reconstruction"]
        idx = stages.index(current)
        return stages[idx + 1] if idx + 1 < len(stages) else "completed"

# =============================================================================
# DATA MANAGER - Memory-Optimized Storage
# =============================================================================

class DataManager:
    """Handles memory-efficient data storage with lazy loading"""
    
    def __init__(self, checkpoint_dir: str, save_fmt: str = 'csv'):
        self.data_dir = Path(checkpoint_dir) / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.save_fmt = save_fmt  # NEW: Store format
        self.compression = "gzip"
        self.compression_level = 1
    
    def save_field_data(self, name: str, data: Dict, dtype=np.float32) -> str:
        """Save field simulation data in requested format"""
        if self.save_fmt.lower() == 'csv':
            # Save as CSV files (compatible with plotting scripts)
            for key, value in data.items():
                filename = self.data_dir / f"{name}_{key}.csv"
                np.savetxt(str(filename), value, delimiter=',')
            return "CSV_SAVED"
        else:
            # Original HDF5 code
            filename = self.data_dir / f"{name}.h5"
            with h5py.File(filename, 'w') as f:
                for key, value in data.items():
                    if key.endswith("_vals"):
                        # Save coordinate arrays
                        f.create_dataset(key, data=value.astype(dtype))
                    elif key == "p_field":
                        # Chunk large field data for efficient I/O
                        chunks = self._calculate_chunks(value.shape, dtype)
                        f.create_dataset(
                            key, 
                            data=value.astype(dtype), 
                            chunks=chunks,
                            compression=self.compression,
                            compression_opts=self.compression_level
                        )
        
        return self._compute_file_checksum(filename)
    
    def save_fmc_batch(self, batch_id: int, tx_range: Tuple[int, int], 
                      fmc_data: np.ndarray, z_vals: np.ndarray) -> str:
        """Save FMC batch data incrementally"""
        batch_file = self.data_dir / f"fmc_batch_{batch_id:04d}.h5"
        
        with h5py.File(batch_file, 'w') as f:
            # Store batch metadata
            f.attrs["batch_id"] = batch_id
            f.attrs["tx_start"] = tx_range[0]
            f.attrs["tx_end"] = tx_range[1]
            f.attrs["z_count"] = len(z_vals)
            
            # Store data with chunking
            chunks = self._calculate_chunks(fmc_data.shape, fmc_data.dtype)
            f.create_dataset(
                "fmc_real", 
                data=fmc_data.real.astype(np.float32),
                chunks=chunks,
                compression=self.compression
            )
            f.create_dataset(
                "fmc_imag",
                data=fmc_data.imag.astype(np.float32),
                chunks=chunks,
                compression=self.compression
            )
            f.create_dataset("z_vals", data=z_vals.astype(np.float32))
        
        return self._compute_file_checksum(batch_file)
    
    def load_fmc_slice(self, z_index: int, tx_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Load specific depth slice from FMC data (memory efficient)"""
        fmc_slice = None
        
        # Find all batch files
        batch_files = sorted(self.data_dir.glob("fmc_batch_*.h5"))
        
        for batch_file in batch_files:
            with h5py.File(batch_file, 'r') as f:
                tx_start = f.attrs["tx_start"]
                tx_end = f.attrs["tx_end"]
                
                # Filter by transmitter range if specified
                if tx_range and (tx_end <= tx_range[0] or tx_start >= tx_range[1]):
                    continue
                
                # Load only the required z-slice
                batch_slice = f["fmc_real"][:, :, z_index] + 1j * f["fmc_imag"][:, :, z_index]
                
                if fmc_slice is None:
                    # Initialize result array
                    M = sum(f.attrs["tx_end"] - f.attrs["tx_start"] for f in 
                           [h5py.File(bf, 'r') for bf in batch_files])
                    N = batch_slice.shape[1]
                    fmc_slice = np.zeros((M, N), dtype=np.complex64)
                
                # Place batch data in correct position
                idx_start = tx_start
                idx_end = tx_end
                fmc_slice[idx_start:idx_end] = batch_slice
        
        return fmc_slice
    
    def save_tfm_result(self, envelope: np.ndarray, z_vals: np.ndarray, 
                       metadata: Dict) -> str:
        """Save TFM reconstruction results"""
        filename = self.data_dir / "tfm_results.h5"
        
        with h5py.File(filename, 'w') as f:
            # Store main results
            f.create_dataset("envelope", data=envelope.astype(np.float32))
            f.create_dataset("z_vals", data=z_vals.astype(np.float32))
            
            # Store metadata
            for key, value in metadata.items():
                if isinstance(value, (int, float, str)):
                    f.attrs[key] = value
        
        return self._compute_file_checksum(filename)
    
    def _calculate_chunks(self, shape: Tuple, dtype: np.dtype) -> Tuple:
        """Calculate optimal chunk size for HDF5 storage"""
        # Aim for chunks of ~1MB
        element_size = np.dtype(dtype).itemsize
        target_chunk_size = 1024 * 1024  # 1MB
        
        if len(shape) == 3:
            # For 3D arrays: keep Z dimension together
            chunk_z = min(shape[2], max(1, target_chunk_size // (shape[0] * shape[1] * element_size)))
            return (shape[0], shape[1], chunk_z)
        elif len(shape) == 2:
            chunk_y = min(shape[1], max(1, target_chunk_size // (shape[0] * element_size)))
            return (shape[0], chunk_y)
        else:
            return shape
    
    def _compute_file_checksum(self, filepath: Path) -> str:
        """Compute SHA256 checksum of file content"""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
        
    # ADICIONAR NOVO MÉTODO NA CLASSE DataManager:
    def save_tfm_result_2d(self, envelope: np.ndarray, x_vals: np.ndarray, 
                           z_vals: np.ndarray, metadata: Dict) -> str:
        """Save 2D TFM reconstruction results"""
        filename = self.data_dir / "tfm_results_2d.h5"
        
        with h5py.File(filename, 'w') as f:
            # Store 2D results
            f.create_dataset("envelope", data=envelope.astype(np.float32))
            f.create_dataset("x_vals", data=x_vals.astype(np.float32))
            f.create_dataset("z_vals", data=z_vals.astype(np.float32))
            
            # Store metadata
            for key, value in metadata.items():
                if isinstance(value, (int, float, str)):
                    f.attrs[key] = value
        
        return self._compute_file_checksum(filename)
        
    # ADICIONAR ESTE MÉTODO NA CLASSE DataManager (após save_tfm_result_2d):

    def save_tfm_checkpoint_2d(self, envelope: np.ndarray, x_vals: np.ndarray, 
                              z_vals: np.ndarray, metadata: Dict) -> str:
        """Save 2D TFM checkpoint results"""
        filename = self.data_dir / "tfm_checkpoint_2d.h5"
        
        with h5py.File(filename, 'w') as f:
            # Store 2D results
            f.create_dataset("envelope", data=envelope.astype(np.float32))
            f.create_dataset("x_vals", data=x_vals.astype(np.float32))
            f.create_dataset("z_vals", data=z_vals.astype(np.float32))
            
            # Store metadata
            for key, value in metadata.items():
                if isinstance(value, (int, float, str)):
                    f.attrs[key] = value
        
        return self._compute_file_checksum(filename)

# =============================================================================
# BATCH PROCESSOR - Memory-Optimized Computation
# =============================================================================

class BatchProcessor:
    """Processes FMC data in memory-controlled batches"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.memory_limit_mb = self._get_available_memory()
        self.batch_size = self._calculate_optimal_batch_size()
    
    def process_digital_twin_batch(self, xs: np.ndarray, zs: np.ndarray, 
                                  ys: np.ndarray) -> Dict:
        """Process field simulation with memory control"""
        # Field simulation is already memory-efficient, but we ensure cleanup
        result = run_mps_array_model_int_service(
            self.args.lx, self.args.ly, self.args.gx, self.args.gy,
            self.args.f, self.args.d1, self.args.c1,
            self.args.d2, self.args.c2, self.args.cs2,
            self.args.wave_type, self.args.L1, self.args.L2,
            self.args.angt, self.args.Dt0, self.args.theta20, 
            self.args.phi, self.args.DF,
            self.args.ampx_type, self.args.ampy_type,
            xs, zs, ys
        )
        
        # Convert to float32 to save memory
        result["p"] = result["p"].astype(np.complex64)
        
        return result
    
    def process_fmc_batch(self, tx_start: int, tx_end: int, 
                         z_vals: np.ndarray) -> np.ndarray:
        """Process a batch of transmitters for FMC generation"""
        M, N = self.args.L1, self.args.L2
        batch_size = tx_end - tx_start
        
        # Calculate element positions
        x_elem = (np.arange(M) - (M-1)/2) * (self.args.lx + self.args.gx)
        y_elem = (np.arange(N) - (N-1)/2) * (self.args.ly + self.args.gy) if N > 1 else np.array([0.0])
        
        # Pre-allocate batch array (complex64 to save memory)
        fmc_batch = np.zeros((batch_size, N, len(z_vals)), dtype=np.complex64)
        
        for tx_idx, tx in enumerate(range(tx_start, tx_end)):
            for rx in range(N):
                ex, ey = x_elem[tx], y_elem[rx]
                p_scan = []
                
                # Process each depth point
                for zv in z_vals:
                    res = run_mps_array_model_int_service(
                        self.args.lx, self.args.ly, self.args.gx, self.args.gy,
                        self.args.f, self.args.d1, self.args.c1,
                        self.args.d2, self.args.c2, self.args.cs2,
                        self.args.wave_type, 1, 1,
                        self.args.angt, self.args.Dt0, self.args.theta20,
                        self.args.phi, self.args.DF,
                        'rect', 'rect', [ex], [zv], ey
                    )
                    
                    p_val = res["p"].item() if hasattr(res["p"], "item") else res["p"][0]
                    p_scan.append(p_val)
                
                # Pulse-echo approximation: P^2
                fmc_batch[tx_idx, rx, :] = np.array(p_scan, dtype=np.complex64) ** 2
        
        return fmc_batch
    
    def _get_available_memory(self) -> float:
        """Get available memory in MB, leaving 30% for OS"""
        available_mb = psutil.virtual_memory().available / 1e6
        return available_mb * 0.7  # Use only 70% of available memory
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory"""
        # Estimate memory per transmitter
        z_points = 171  # Default from args
        N = self.args.L2
        
        # Memory per transmitter: N * Z * 16 bytes (complex128)
        # Using complex64 instead: 8 bytes
        bytes_per_tx = N * z_points * 8  # complex64
        
        # Available memory in bytes
        available_bytes = self.memory_limit_mb * 1e6
        
        # Calculate max transmitters that fit in memory
        max_tx_per_batch = max(1, int(available_bytes / bytes_per_tx))
        
        # Don't make batches too small (overhead) or too large (memory pressure)
        optimal = min(self.args.L1, max(4, max_tx_per_batch))
        
        print(f"[INFO] Memory: {self.memory_limit_mb:.1f} MB available")
        print(f"[INFO] Batch size: {optimal} transmitters")
        
        return optimal

# =============================================================================
# RECOVERY MANAGER - Fault Tolerance
# =============================================================================

class RecoveryManager:
    """Handles pipeline recovery from checkpoints"""
    
    def __init__(self, checkpoint_dir: str, args: argparse.Namespace):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.args = args
        self.checkpoint_mgr = CheckpointManager(checkpoint_dir)
        self.data_mgr = DataManager(checkpoint_dir)
    
    def should_resume(self) -> bool:
        """Check if recovery is possible and desired"""
        if not self.checkpoint_dir.exists():
            return False
        
        state = self.checkpoint_mgr.load_state()
        if not state:
            return False
        
        # Check if parameters match
        saved_params = state.get("parameters", {})
        current_params = vars(self.args)
        
        # Only check critical parameters
        critical_params = ['lx', 'ly', 'gx', 'gy', 'f', 'L1', 'L2', 'Dt0', 'c1', 'c2']
        for param in critical_params:
            if saved_params.get(param) != current_params.get(param):
                print(f"[WARN] Parameter mismatch: {param}")
                print(f"  Saved: {saved_params.get(param)}, Current: {current_params.get(param)}")
                return False
        
        return True
    
    def get_recovery_point(self) -> Tuple[str, Dict]:
        """Determine where to resume from"""
        state = self.checkpoint_mgr.load_state()
        if not state:
            return "start", {}
        
        current_stage = state.get("current_stage", "start")
        stage_info = state.get("stages", {}).get(current_stage, {})
        
        print(f"[RECOVERY] Resuming from stage: {current_stage}")
        
        if current_stage == "fmc_generation":
            completed_tx = stage_info.get("completed_transmitters", [])
            if completed_tx:
                print(f"[RECOVERY] {len(completed_tx)} transmitters already processed")
        
        return current_stage, state

# =============================================================================
# MAIN PIPELINE - Enhanced with Checkpoints
# =============================================================================

class EnhancedPipeline:
    """Main pipeline with checkpointing and memory optimization"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.checkpoint_dir = Path(args.out_root) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_mgr = CheckpointManager(self.checkpoint_dir)
        self.data_mgr = DataManager(self.checkpoint_dir, args.save_fmt)  # FIXED
        self.batch_processor = BatchProcessor(args)
        self.recovery_mgr = RecoveryManager(self.checkpoint_dir, args)
        
        # Parse scan vectors
        self.xs = self._parse_scan_vector(args.xs, -15, 15, 31)
        self.zs = self._parse_scan_vector(args.zs, 1, 171, 171)
        self.ys = self._parse_scan_vector(args.y_vec, 0, 0, 1)
        self.x_tfm = self._parse_scan_vector(args.x_mm, -15, 15, 31)  # Adicionar argumento
        self.z_tfm = self._parse_scan_vector(args.z_mm, 1, 171, 171)

    @staticmethod
    def save_run_params(args, out_root: str) -> None:
        """Persist runtime arguments with theoretical FWHM for plotting scripts"""
        params_dict = vars(args)
        D = (args.L1 * args.lx) + (args.L1 - 1) * args.gx
        if D > 0 and args.f > 0:
            FWHM_theoretical = 1.206 * (args.c2 / args.f / 1000) * ((args.DF + args.Dt0) / D)
            params_dict['theoretical_fwhm_mm'] = FWHM_theoretical
        else:
            params_dict['theoretical_fwhm_mm'] = 0.0

        # Ensure output directory exists
        os.makedirs(out_root, exist_ok=True)
        
        with open(os.path.join(out_root, "run_params.json"), "w") as f:
            json.dump(params_dict, f, indent=4)
        print(f"[AP] Saved run parameters → {out_root}/run_params.json")
    
    def run(self, resume: bool = False):
        """Run the enhanced pipeline with checkpointing"""
        print("[PIPELINE] Starting enhanced pipeline")
        print(f"[PIPELINE] Checkpoint directory: {self.checkpoint_dir}")
        
        # NEW: Save parameters for plotting scripts
        self.save_run_params(self.args, str(self.args.out_root))
        
        if resume and self.recovery_mgr.should_resume():
            print("[PIPELINE] Recovery mode enabled")
            recovery_point, state = self.recovery_mgr.get_recovery_point()
        else:
            print("[PIPELINE] Starting fresh run")
            recovery_point = "start"
            state = self.checkpoint_mgr.init_checkpoint(self.args)
        
        # Execute pipeline from recovery point
        if recovery_point == "start" or recovery_point == "digital_twin":
            self._run_digital_twin(state)
        
        if recovery_point in ["start", "digital_twin", "fmc_generation"]:
            self._run_fmc_generation(state, recovery_point)
        
        if recovery_point in ["start", "digital_twin", "fmc_generation", "tfm_reconstruction"]:
            self._run_tfm_reconstruction(state)
        
        print("[PIPELINE] Pipeline completed successfully")
    
    def _run_digital_twin(self, state: Dict):
        """Run digital twin field simulation"""
        print("[STAGE] Digital Twin Field Simulation")
        
        # Check if already completed
        if state["stages"]["digital_twin"]["status"] == "completed":
            print("[STAGE] Digital Twin already completed, skipping")
            return
        
        # Process field simulation
        start_time = time.time()
        
        result = self.batch_processor.process_digital_twin_batch(
            self.xs, self.zs, self.ys
        )
        
        # Save results with checksum
        data_dict = {
            "p_field": np.abs(result["p"]),
            "x_vals": self.xs,
            "z_vals": self.zs,
            "y_vals": self.ys
        }
        
        checksum = self.data_mgr.save_field_data("field", data_dict)
        
        # Update state
        self.checkpoint_mgr.mark_stage_complete(state, "digital_twin", checksum)
        self.checkpoint_mgr.save_state(state)
        
        elapsed = time.time() - start_time
        print(f"[STAGE] Digital Twin completed in {elapsed:.1f}s")
        
        # ============ ADD THIS CODE HERE ============
        # Save data for plotting scripts (CSV format expected by plotting scripts)
        print("[PLOT] Saving Digital Twin data for plotting scripts...")
        dt_plot_dir = Path(self.args.out_root) / "digital_twin"
        dt_plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Save in the format expected by plotting scripts
        np.savetxt(dt_plot_dir / "field_z_vals.csv", self.zs, delimiter=',')
        np.savetxt(dt_plot_dir / "field_x_vals.csv", self.xs, delimiter=',')
        np.savetxt(dt_plot_dir / "field_y_vals.csv", self.ys, delimiter=',')
        
        # Reshape p_field to 2D (z, x) for plotting
        p_field_2d = np.abs(result["p"]).reshape((len(self.zs), len(self.xs)))
        np.savetxt(dt_plot_dir / "field_p_field.csv", p_field_2d, delimiter=',')
        # ============ END OF ADDED CODE ============

        # At the end of _run_digital_twin():
        self._save_plotting_data("digital_twin", {
            "p_field": np.abs(result["p"]),
            "x_vals": self.xs,
            "z_vals": self.zs,
            "y_vals": self.ys})
    
    def _run_fmc_generation(self, state: Dict, recovery_point: str):
        """Run FMC generation with batch processing and checkpoints"""
        print("[STAGE] FMC Generation with Batch Processing")
        
        stage_info = state["stages"]["fmc_generation"]
        
        # Determine starting point
        if stage_info["status"] == "completed":
            print("[STAGE] FMC generation already completed, skipping")
            return
        
        completed_tx = set(stage_info.get("completed_transmitters", []))
        batch_size = self.batch_processor.batch_size
        
        M = self.args.L1
        start_time = time.time()
        
        # Process in batches
        for batch_start in range(0, M, batch_size):
            batch_end = min(batch_start + batch_size, M)
            batch_id = batch_start // batch_size
            
            # Skip if batch already completed
            batch_tx = list(range(batch_start, batch_end))
            if all(tx in completed_tx for tx in batch_tx):
                print(f"[BATCH] Batch {batch_id} already completed, skipping")
                continue
            
            print(f"[BATCH] Processing batch {batch_id}: transmitters {batch_start}-{batch_end-1}")
            batch_start_time = time.time()
            
            # Process batch
            fmc_batch = self.batch_processor.process_fmc_batch(
                batch_start, batch_end, self.z_tfm
            )
            
            # Save batch to disk
            checksum = self.data_mgr.save_fmc_batch(
                batch_id, (batch_start, batch_end), fmc_batch, self.z_tfm
            )
            
            # Update state
            state["stages"]["fmc_generation"]["completed_transmitters"].extend(batch_tx)
            state["stages"]["fmc_generation"]["batches"][str(batch_id)] = {
                "checksum": checksum,
                "tx_range": [batch_start, batch_end],
                "saved_time": datetime.utcnow().isoformat()
            }
            state["stages"]["fmc_generation"]["last_batch"] = batch_id
            
            # Save checkpoint after each batch
            if not self.checkpoint_mgr.save_state(state):
                print("[WARN] Failed to save checkpoint, but batch data is saved")
            
            batch_time = time.time() - batch_start_time
            print(f"[BATCH] Batch {batch_id} completed in {batch_time:.1f}s")
            
            # Force garbage collection
            del fmc_batch
        
        # Mark stage as complete
        self.checkpoint_mgr.mark_stage_complete(state, "fmc_generation")
        self.checkpoint_mgr.save_state(state)
        
        total_time = time.time() - start_time
        print(f"[STAGE] FMC generation completed in {total_time:.1f}s")
    
    def _run_tfm_reconstruction(self, state: Dict):
        """Run TFM reconstruction with memory-mapped data access"""
        print("[STAGE] TFM Reconstruction")
        
        if state["stages"]["tfm_reconstruction"]["status"] == "completed":
            print("[STAGE] TFM already completed, skipping")
            return
        
        M, N = self.args.L1, self.args.L2
        start_time = time.time()
        
        # Calculate focusing delays for reference
        td_focus = run_delay_laws3Dint_service(
            M, N, self.args.lx + self.args.gx, self.args.ly + self.args.gy,
            self.args.angt, self.args.phi, self.args.theta20,
            self.args.Dt0, self.args.DF, self.args.c1, self.args.c2, 'n'
        )
        
        # Initialize TFM result array                
        tfm_envelope = np.zeros((len(self.x_tfm), len(self.z_tfm)), dtype=np.float32)

        # Pre-calcular posições dos elementos (para eficiência)
        M, N = self.args.L1, self.args.L2
        x_elem = (np.arange(M) - (M-1)/2) * (self.args.lx + self.args.gx)
        y_elem = (np.arange(N) - (N-1)/2) * (self.args.ly + self.args.gy) if N > 1 else np.array([0.0])
        
        # Processar cada ponto da grade 2D
        for x_idx, x_val in enumerate(self.x_tfm):
            print(f"[TFM] Processing X position {x_idx+1}/{len(self.x_tfm)}: x={x_val:.1f} mm")
            
            for z_idx, z_val in enumerate(self.z_tfm):
                # Carregar slice FMC para esta profundidade
                fmc_slice = self.data_mgr.load_fmc_slice(z_idx)
                
                if fmc_slice is None:
                    continue
                
                # Calcular atrasos para ponto (x_val, 0, z_val)
                if z_val > self.args.Dt0:
                    # NO AÇO: fórmula simplificada para distância em meio sólido
                    # Para reconstrução 2D completa, precisaríamos de refração, mas
                    # como aproximação, usamos caminho ótico equivalente
                    dist_water = np.sqrt((x_elem - x_val)**2 + y_elem**2 + self.args.Dt0**2)
                    dist_steel = np.sqrt((x_elem - x_val)**2 + y_elem**2 + (z_val - self.args.Dt0)**2)
                    td = dist_water/self.args.c1 + dist_steel/self.args.c2
                else:
                    # NA ÁGUA: distância euclidiana simples
                    dist = np.sqrt((x_elem - x_val)**2 + y_elem**2 + z_val**2)
                    td = dist / self.args.c1
                
                # Converter atrasos para matriz M×N
                td_matrix = td.reshape(M, N)
                
                # Correção de fase e soma coerente
                phase_corr = np.exp(-1j * 4 * np.pi * self.args.f * (td_matrix * 1e6))
                tfm_slice = np.sum(fmc_slice * phase_corr) * (z_val ** 2)
                
                # Armazenar no array 2D
                tfm_envelope[x_idx, z_idx] = np.abs(tfm_slice)               
              
                # Save intermediate checkpoint every 10% of X progress
                if (x_idx + 1) % max(1, len(self.x_tfm) // 10) == 0:
                    progress = (x_idx + 1) / len(self.x_tfm) * 100
                    print(f"[TFM] Saving checkpoint at {progress:.0f}% progress")
                    
                    metadata = {
                        "last_x_index": x_idx,
                        "last_x_value": float(x_val),
                        "progress_percent": progress
                    }
                    # Save the current 2D envelope (up to current x_idx)
                    checkpoint_envelope = tfm_envelope[:x_idx+1, :]
                    checkpoint_x_vals = self.x_tfm[:x_idx+1]
                    
                    self.data_mgr.save_tfm_checkpoint_2d(
                        checkpoint_envelope, checkpoint_x_vals, self.z_tfm, metadata
                    )
        
        # Save final results
        metadata = {
            "fwhm_theoretical": self._calculate_theoretical_fwhm(),
            "peak_index": int(np.argmax(tfm_envelope)),
            "peak_value": float(np.max(tfm_envelope))
        }
                
        checksum = self.data_mgr.save_tfm_result_2d(tfm_envelope, self.x_tfm, self.z_tfm, metadata)
        
        # Update state
        self.checkpoint_mgr.mark_stage_complete(state, "tfm_reconstruction", checksum)
        self.checkpoint_mgr.save_state(state)
        
        # Identify post-interface peak
        z_mask = self.z_tfm > self.args.Dt0 + 5.0
        if np.any(z_mask):
            roi = tfm_envelope[:, z_mask]
            peak_idx = np.unravel_index(np.argmax(roi), roi.shape)
            peak_x_idx = peak_idx[0]
            peak_z_idx = peak_idx[1] + np.where(z_mask)[0][0]
            print(f"[STAGE] TFM peak detected at x = {self.x_tfm[peak_x_idx]:.2f} mm, z = {self.z_tfm[peak_z_idx]:.2f} mm")
        
        elapsed = time.time() - start_time
        print(f"[STAGE] TFM reconstruction completed in {elapsed:.1f}s")
        
        # ============ ADD THIS CODE HERE ============
        # Save data for plotting scripts (CSV format expected by plotting scripts)
        print("[PLOT] Saving TFM data for plotting scripts...")
        tfm_plot_dir = Path(self.args.out_root) / "fmc_tfm"
        tfm_plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Save in the format expected by plotting scripts
        np.savetxt(tfm_plot_dir / "results_z_vals.csv", self.z_tfm, delimiter=',')
        np.savetxt(tfm_plot_dir / "results_envelope.csv", tfm_envelope, delimiter=',')
        
        # Also save raw complex TFM data for completeness
        if hasattr(self, 'tfm_raw_complex'):
            np.savetxt(tfm_plot_dir / "results_tfm_raw_real.csv", self.tfm_raw_complex.real, delimiter=',')
            np.savetxt(tfm_plot_dir / "results_tfm_raw_imag.csv", self.tfm_raw_complex.imag, delimiter=',')
        # ============ END OF ADDED CODE ============
        
        # At the end of _run_tfm_reconstruction():
        self._save_plotting_data("fmc_tfm", {
            "envelope": tfm_envelope,
            "x_vals": self.x_tfm,
            "z_vals": self.z_tfm
        })
    
    def _calculate_theoretical_fwhm(self) -> float:
        """Calculate theoretical FWHM baseline"""
        D = (self.args.L1 * self.args.lx) + (self.args.L1 - 1) * self.args.gx
        if D > 0 and self.args.f > 0:
            return 1.206 * (self.args.c2 / self.args.f / 1000) * ((self.args.DF + self.args.Dt0) / D)
        return 0.0
    
    def _parse_scan_vector(self, input_val: str, default_start: float, 
                          default_stop: float, default_num: int) -> np.ndarray:
        """Parse MATLAB-style vector specification"""
        if input_val is None:
            return np.linspace(default_start, default_stop, default_num)
        
        try:
            parts = [float(x) for x in str(input_val).replace('"', '').replace("'", "").split(',')]
            if len(parts) == 3:
                return np.linspace(parts[0], parts[1], int(parts[2]))
            return np.array(parts)
        except:
            return np.linspace(default_start, default_stop, default_num)
 
    def _save_plotting_data(self, stage: str, data_dict: dict):
        """Save data in format expected by plotting scripts"""
        if stage == "digital_twin":
            out_dir = Path(self.args.out_root) / "digital_twin"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Save coordinate arrays
            for key in ["z_vals", "x_vals", "y_vals"]:
                if key in data_dict:
                    np.savetxt(out_dir / f"field_{key}.csv", data_dict[key], delimiter=',')
            
            # Save field data (reshape to 2D for plotting)
            if "p_field" in data_dict:
                z_len = len(data_dict["z_vals"]) if "z_vals" in data_dict else 1
                x_len = len(data_dict["x_vals"]) if "x_vals" in data_dict else 1
                p_2d = data_dict["p_field"].reshape((z_len, x_len))
                np.savetxt(out_dir / "field_p_field.csv", p_2d, delimiter=',')                
        
        elif stage == "fmc_tfm":
            out_dir = Path(self.args.out_root) / "fmc_tfm"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Save 2D envelope
            if "envelope" in data_dict:
                np.savetxt(out_dir / "results_envelope_2d.csv", data_dict["envelope"], delimiter=',')
            
            # Save coordinate arrays
            if "x_vals" in data_dict:
                np.savetxt(out_dir / "results_x_vals.csv", data_dict["x_vals"], delimiter=',')
            if "z_vals" in data_dict:
                np.savetxt(out_dir / "results_z_vals.csv", data_dict["z_vals"], delimiter=',')

# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Ultrasound Pipeline with Checkpoint Recovery"
    )
    
    # Physical parameters
    parser.add_argument('--lx', type=safe_float, required=True)
    parser.add_argument('--ly', type=safe_float, required=True)
    parser.add_argument('--gx', type=safe_float, required=True)
    parser.add_argument('--gy', type=safe_float, required=True)
    parser.add_argument('--f', type=safe_float, required=True)
    parser.add_argument('--d1', type=safe_float, required=True)
    parser.add_argument('--c1', type=safe_float, required=True)
    parser.add_argument('--d2', type=safe_float, required=True)
    parser.add_argument('--c2', type=safe_float, required=True)
    parser.add_argument('--cs2', type=safe_float, required=True)
    
    # Array configuration
    parser.add_argument('--L1', type=int, default=32)
    parser.add_argument('--L2', type=int, default=1)
    parser.add_argument('--ampx_type', default='rect')
    parser.add_argument('--ampy_type', default='rect')
    parser.add_argument('--b', type=int, default=10)
    
    # Beam steering/focusing
    parser.add_argument('--angt', type=safe_float, default=0.0)
    parser.add_argument('--theta20', type=safe_float, default=0.0)
    parser.add_argument('--phi', type=safe_float, default=0.0)
    parser.add_argument('--DF', type=safe_float, default=47.2)
    parser.add_argument('--Dt0', type=safe_float, default=59.2)
    parser.add_argument('--wave_type', choices=['p','s'], default='p')
    
    # Scan parameters
    parser.add_argument('--xs', type=str, default="-15,15,31")
    parser.add_argument('--zs', type=str, default="1,171,171")
    parser.add_argument('--y_vec', type=str, default="0")
    parser.add_argument('--x_mm', type=str, default="-15,15,31")
    parser.add_argument('--z_mm', type=str, default="1,171,171")
    
    # Output control
    parser.add_argument('--out_root', default='results')
    parser.add_argument('--save_fmt', choices=['csv','npz','h5'], default='csv')
    
    # Recovery control
    parser.add_argument('--resume', action='store_true', 
                       help='Resume from last checkpoint')
    parser.add_argument('--force_restart', action='store_true',
                       help='Ignore existing checkpoints and start fresh')
    
    # Performance tuning
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Manual batch size override (auto-calculated if not specified)')
    parser.add_argument('--memory_limit_mb', type=float, default=None,
                       help='Manual memory limit in MB')
    
    args = parser.parse_args()
    
    # Override batch processor settings if specified
    if args.batch_size:
        BatchProcessor._calculate_optimal_batch_size = lambda self: args.batch_size
    if args.memory_limit_mb:
        BatchProcessor._get_available_memory = lambda self: args.memory_limit_mb
    
    # Run pipeline
    pipeline = EnhancedPipeline(args)
    pipeline.run(resume=args.resume and not args.force_restart)
    
    print("[MAIN] Pipeline execution complete")

if __name__ == "__main__":
    main()