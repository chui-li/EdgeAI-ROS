from __future__ import annotations

import os
import random
import tempfile
import threading
import time
from dataclasses import dataclass

import psutil


SAFE_MAX_CPU_WORKERS = max(1, min(4, psutil.cpu_count(logical=True) or 1))
SAFE_MAX_MEMORY_STRESS_MB = 2048


def set_safe_torch_threads(num_threads: int):
    import torch
    cpu_count = psutil.cpu_count(logical=True) or 1
    num_threads = max(1, min(num_threads, cpu_count))
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


class SafeBackgroundLoad:
    """Controlled stress injection for CPU, memory, and I/O."""

    def __init__(self):
        self.mode = "none"
        self.stop_event = threading.Event()
        self.threads = []
        self.memory_blocks = []
        self.temp_file = None
        self.lock = threading.Lock()

    def _cpu_worker(self):
        x = random.random() + 1.0
        while not self.stop_event.is_set():
            for _ in range(20000):
                x = x * 1.000001 + 0.000001
            time.sleep(0.001)

    def _memory_worker(self, memory_mb: int):
        memory_mb = max(1, min(memory_mb, SAFE_MAX_MEMORY_STRESS_MB))
        block_size = 8 * 1024 * 1024
        try:
            for _ in range(max(1, memory_mb * 1024 * 1024 // block_size)):
                if self.stop_event.is_set():
                    break
                self.memory_blocks.append(bytearray(block_size))
                time.sleep(0.02)
            while not self.stop_event.is_set():
                time.sleep(0.1)
        except MemoryError:
            self.memory_blocks.clear()

    def _io_worker(self):
        try:
            fd, path = tempfile.mkstemp(prefix="edgeai_os_io_", suffix=".tmp")
            os.close(fd)
            self.temp_file = path
            chunk = os.urandom(1024 * 1024)
            with open(path, "wb") as f:
                while not self.stop_event.is_set():
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                    if f.tell() > 64 * 1024 * 1024:
                        f.seek(0)
                        f.truncate(0)
                    time.sleep(0.05)
        except Exception as e:
            print(f"[WARN] IO stress failed: {e}")

    def start(self, mode: str, cpu_workers: int = 2, memory_mb: int = 512):
        with self.lock:
            self.stop()
            self.mode = mode
            self.stop_event.clear()
            if mode in ["cpu", "mixed"]:
                for _ in range(max(1, min(cpu_workers, SAFE_MAX_CPU_WORKERS))):
                    t = threading.Thread(target=self._cpu_worker, daemon=True)
                    t.start()
                    self.threads.append(t)
            if mode in ["memory", "mixed"]:
                t = threading.Thread(target=self._memory_worker, args=(memory_mb,), daemon=True)
                t.start()
                self.threads.append(t)
            if mode in ["io", "mixed"]:
                t = threading.Thread(target=self._io_worker, daemon=True)
                t.start()
                self.threads.append(t)

    def stop(self):
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=0.5)
        self.threads.clear()
        self.memory_blocks.clear()
        if self.temp_file and os.path.exists(self.temp_file):
            try:
                os.remove(self.temp_file)
            except Exception:
                pass
        self.temp_file = None
        self.mode = "none"
