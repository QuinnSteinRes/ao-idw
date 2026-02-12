#!/usr/bin/env python3
"""
Cluster training script for IDW-PINN
Adapted for new modular codebase with YAML configuration
"""

import os
import sys
import gc
import time
import signal
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import psutil
import numpy as np
import tensorflow as tf

# Configure TensorFlow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs available: {len(gpus)}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPUs found, using CPU")

# CPU thread limits
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

# Import your modular IDW-PINN code
try:
    from src.idw_pinn.config import load_config
    from src.idw_pinn.data import load_data, prepare_training_data
    from src.idw_pinn.models import create_model
    from src.idw_pinn.training import train_model
    from src.idw_pinn.utils.visualization import save_results
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure PYTHONPATH includes your project root")
    sys.exit(1)


class MemoryMonitor:
    """Monitor memory usage during training"""
    
    def __init__(self, log_file='memory_usage.log'):
        self.log_file = log_file
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()
        
    def log(self, message=""):
        """Log current memory usage"""
        mem_info = self.process.memory_info()
        system_mem = psutil.virtual_memory()
        elapsed = time.time() - self.start_time
        
        log_entry = (
            f"[{elapsed:.1f}s] {message}\n"
            f"  Process: RSS={mem_info.rss/(1024**2):.1f}MB, "
            f"VMS={mem_info.vms/(1024**2):.1f}MB\n"
            f"  System: {system_mem.percent}% used, "
            f"{system_mem.available/(1024**2):.1f}MB available\n"
        )
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        return mem_info.rss / (1024**2)  # Return RSS in MB


def setup_signal_handlers(log_file='signal_log.txt'):
    """Setup signal handlers for debugging crashes"""
    
    def signal_handler(sig, frame):
        signal_name = signal.Signals(sig).name
        
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Signal {signal_name} received at {datetime.now()}\n")
            f.write(f"{'='*60}\n")
            traceback.print_stack(frame, file=f)
            
            # Memory info
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            f.write(f"\nProcess memory: RSS={mem_info.rss/(1024**2):.1f}MB\n")
            
            system_mem = psutil.virtual_memory()
            f.write(f"System memory: {system_mem.percent}% used\n")
        
        if sig in (signal.SIGTERM, signal.SIGINT):
            print(f"\n{signal_name} received, cleaning up...")
            sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        signal.signal(signal.SIGSEGV, signal_handler)
    except:
        pass
    
    print(f"Signal handlers configured, logging to {log_file}")


def log_environment():
    """Log comprehensive environment information"""
    
    print("\n" + "="*80)
    print("IDW-PINN CLUSTER TRAINING - ENVIRONMENT INFO")
    print("="*80)
    
    print(f"\nRun information:")
    print(f"  Date/time: {datetime.now()}")
    print(f"  Hostname: {os.uname().nodename}")
    print(f"  PID: {os.getpid()}")
    
    print(f"\nPython environment:")
    print(f"  Python: {sys.version}")
    print(f"  Executable: {sys.executable}")
    print(f"  PYTHONPATH: {os.environ.get('PYTHONPATH', 'not set')}")
    print(f"  Conda env: {os.environ.get('CONDA_DEFAULT_ENV', 'not set')}")
    
    print(f"\nTensorFlow:")
    print(f"  Version: {tf.__version__}")
    print(f"  GPUs: {len(tf.config.list_physical_devices('GPU'))}")
    
    print(f"\nSystem resources:")
    mem = psutil.virtual_memory()
    print(f"  Total memory: {mem.total/(1024**3):.1f} GB")
    print(f"  Available memory: {mem.available/(1024**3):.1f} GB")
    print(f"  CPU count: {psutil.cpu_count()}")
    
    print("="*80 + "\n")


def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description='IDW-PINN cluster training')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Setup monitoring
    setup_signal_handlers()
    memory_monitor = MemoryMonitor()
    log_environment()
    
    print(f"Configuration:")
    print(f"  Config file: {args.config}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output dir: {args.output_dir}")
    print()
    
    memory_monitor.log("Initial state")
    
    try:
        # Load configuration
        print("Loading configuration...")
        config = load_config(args.config)
        config.training.seed = args.seed  # Override with CLI seed
        memory_monitor.log("Config loaded")
        
        # Load and prepare data
        print("Loading data...")
        data = load_data(config)
        train_data = prepare_training_data(data, config)
        memory_monitor.log("Data loaded")
        
        # Create model
        print("Creating model...")
        model = create_model(config)
        memory_monitor.log("Model created")
        
        # Train
        print("\nStarting training...")
        print(f"Target epochs: {config.training.epochs}")
        print()
        
        history = train_model(
            model=model,
            data=train_data,
            config=config,
            verbose=1
        )
        
        memory_monitor.log("Training completed")
        
        # Save results
        print("\nSaving results...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_results(
            model=model,
            history=history,
            data=train_data,
            config=config,
            output_dir=output_dir
        )
        
        memory_monitor.log("Results saved")
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        
        # Final memory cleanup
        gc.collect()
        tf.keras.backend.clear_session()
        
        return 0
        
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR DURING TRAINING")
        print("="*80)
        print(f"\nException: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        memory_monitor.log("Error occurred")
        
        # Save error log
        error_log = Path(args.output_dir) / 'error_log.txt'
        with open(error_log, 'w') as f:
            f.write(f"Error at {datetime.now()}\n")
            f.write(f"Exception: {type(e).__name__}\n")
            f.write(f"Message: {str(e)}\n\n")
            f.write("Traceback:\n")
            traceback.print_exc(file=f)
        
        return 1


if __name__ == '__main__':
    sys.exit(main())
