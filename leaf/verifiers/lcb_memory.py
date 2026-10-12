"""Memory-limit helpers for LiveCodeBench candidate subprocesses."""
from pathlib import Path


GIB = 2**30


def process_virtual_memory_bytes():
    """Return the current Linux process virtual-memory footprint."""
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmSize:'):
            fields = line.split()
            if len(fields) != 3 or fields[2] != 'kB':
                break
            return int(fields[1]) * 1024
    raise RuntimeError('Could not read VmSize from /proc/self/status')


def candidate_memory_limit(extra_bytes=4 * GIB):
    """Absolute RLIMIT that preserves inherited mappings plus a candidate quota."""
    if extra_bytes <= 0:
        raise ValueError('Candidate memory allowance must be positive')
    return process_virtual_memory_bytes() + extra_bytes
