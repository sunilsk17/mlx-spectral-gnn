"""
Memory and timing measurement utilities for Apple Silicon benchmarking.
"""

import os
import time
import psutil


class MemoryTracker:
    """Track peak memory usage of the current process."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_mb = 0.0
        self.peak_mb = 0.0
    
    def start(self):
        """Record baseline memory."""
        self.baseline_mb = self._current_mb()
        self.peak_mb = self.baseline_mb
    
    def update(self):
        """Update peak memory if current usage is higher."""
        current = self._current_mb()
        if current > self.peak_mb:
            self.peak_mb = current
    
    def get_peak_mb(self):
        """Return peak memory usage above baseline."""
        self.update()
        return self.peak_mb - self.baseline_mb
    
    def get_absolute_peak_mb(self):
        """Return absolute peak memory usage."""
        self.update()
        return self.peak_mb
    
    def _current_mb(self):
        """Get current RSS memory in MB."""
        return self.process.memory_info().rss / (1024 ** 2)


class Timer:
    """Simple context-manager based timer."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0
    
    def start(self):
        self.start_time = time.perf_counter()
    
    def stop(self):
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time
            self.start_time = None
        return self.elapsed
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


def get_system_info():
    """Return basic system info for the experiment log."""
    import platform
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
    }
    try:
        import mlx.core as mx
        info["mlx_device"] = str(mx.default_device())
    except ImportError:
        info["mlx_device"] = "N/A"
    return info
