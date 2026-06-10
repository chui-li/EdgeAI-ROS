from __future__ import annotations

import os
import resource
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import psutil


@dataclass
class OSState:
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    process_rss_mb: float
    process_vms_mb: float
    page_faults: int
    page_faults_delta: int
    voluntary_ctx_switches: int
    involuntary_ctx_switches: int
    ctx_switches_delta: int
    read_bytes: int
    write_bytes: int
    io_read_delta: int
    io_write_delta: int
    cpu_num: int
    cpu_migration_delta: int
    cpu_percent_per_core: str

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class OSMonitor:
    """Collects process-level and system-level OS metrics.

    Metrics include CPU usage, memory pressure, RSS, page faults, context switches,
    and process I/O counters. The first sample may contain initialization spikes;
    downstream plots should interpret it as warm-up behavior.
    """

    def __init__(self, pid: Optional[int] = None):
        self.process = psutil.Process(pid or os.getpid())
        psutil.cpu_percent(interval=None)
        self._last_page_faults = self._get_page_faults()
        self._last_ctx = self._get_ctx_switches_total()
        self._last_read, self._last_write = self._get_io_bytes()
        self._last_cpu_num = self._get_cpu_num()

    @staticmethod
    def _get_page_faults() -> int:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return int(usage.ru_minflt + usage.ru_majflt)

    def _get_ctx_switches_total(self) -> int:
        try:
            ctx = self.process.num_ctx_switches()
            return int(ctx.voluntary + ctx.involuntary)
        except Exception:
            return 0

    def _get_io_bytes(self):
        try:
            io = self.process.io_counters()
            return int(io.read_bytes), int(io.write_bytes)
        except Exception:
            return 0, 0

    def _get_cpu_num(self) -> int:
        try:
            return int(self.process.cpu_num())
        except Exception:
            try:
                return int(os.sched_getcpu())
            except Exception:
                return -1

    def sample(self) -> OSState:
        cpu = float(psutil.cpu_percent(interval=0.01))
        per_core = psutil.cpu_percent(interval=None, percpu=True)
        mem = psutil.virtual_memory()
        p_mem = self.process.memory_info()

        pf = self._get_page_faults()
        pf_delta = max(0, pf - self._last_page_faults)
        self._last_page_faults = pf

        try:
            ctx = self.process.num_ctx_switches()
            vol, invol = int(ctx.voluntary), int(ctx.involuntary)
        except Exception:
            vol, invol = 0, 0
        ctx_total = vol + invol
        ctx_delta = max(0, ctx_total - self._last_ctx)
        self._last_ctx = ctx_total

        read_b, write_b = self._get_io_bytes()
        read_delta = max(0, read_b - self._last_read)
        write_delta = max(0, write_b - self._last_write)
        self._last_read, self._last_write = read_b, write_b
        cpu_num = self._get_cpu_num()
        migration_delta = int(cpu_num != self._last_cpu_num and cpu_num >= 0 and self._last_cpu_num >= 0)
        self._last_cpu_num = cpu_num

        return OSState(
            timestamp=time.time(),
            cpu_percent=cpu,
            memory_percent=float(mem.percent),
            memory_used_mb=float(mem.used / 1024 / 1024),
            memory_available_mb=float(mem.available / 1024 / 1024),
            process_rss_mb=float(p_mem.rss / 1024 / 1024),
            process_vms_mb=float(p_mem.vms / 1024 / 1024),
            page_faults=pf,
            page_faults_delta=pf_delta,
            voluntary_ctx_switches=vol,
            involuntary_ctx_switches=invol,
            ctx_switches_delta=ctx_delta,
            read_bytes=read_b,
            write_bytes=write_b,
            io_read_delta=read_delta,
            io_write_delta=write_delta,
            cpu_num=cpu_num,
            cpu_migration_delta=migration_delta,
            cpu_percent_per_core=";".join(f"{x:.1f}" for x in per_core),
        )
