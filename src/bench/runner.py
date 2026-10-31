"""Run one method on one case in a fresh process and measure its memory and I/O.

The parent samples the resident set size (RSS) of the whole worker process tree every
25 ms and stops the run as OOM when it exceeds the memory budget M, and as TIMEOUT when it
exceeds the wall-clock cap; such outcomes are results. After the worker exits, its exact
peak working set is read from the operating system (GetProcessMemoryInfo), and device read
bytes per physical disk are recorded around the run.
"""
import json
import os
import subprocess
import sys
import time
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
PY = os.path.join(os.path.dirname(ROOT), 'VLDB_Batched_SAM', '.venv', 'Scripts', 'python.exe')
RSCRIPT = r'C:\Program Files\R\R-4.5.1\bin\Rscript.exe'
R_METHODS = {'grpreg', 'sparsegl', 'SAM', 'biglasso'}
ENV = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
       'NUMBA_NUM_THREADS': '1', 'PYTHONHASHSEED': '0'}


def tree_rss(proc):
    total = 0
    try:
        procs = [proc] + proc.children(recursive=True)
    except psutil.Error:
        return 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


def _open_handle(pid):
    """Query handle that keeps a process's memory counters readable after it exits."""
    if sys.platform != 'win32':
        return None
    import ctypes
    k32 = ctypes.WinDLL('kernel32')
    k32.OpenProcess.restype = ctypes.c_void_p
    h = k32.OpenProcess(0x1000 | 0x0400, False, int(pid))     # limited query | query information
    return h or None


def _peak_memory(handle):
    """Exact peak working set and peak commit of a (possibly finished) process."""
    if sys.platform != 'win32' or not handle:
        return None, None
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    try:
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        ok = ctypes.WinDLL('psapi').GetProcessMemoryInfo(ctypes.c_void_p(int(handle)), ctypes.byref(c), c.cb)
        return (int(c.PeakWorkingSetSize), int(c.PeakPagefileUsage)) if ok else (None, None)
    except Exception:
        return None, None


def _disk_reads():
    try:
        return {k: v.read_bytes for k, v in psutil.disk_io_counters(perdisk=True).items()}
    except Exception:
        return {}


def run(case, method, out, options=None, budget_bytes=None, cap_seconds=3 * 3600):
    os.makedirs(out, exist_ok=True)
    options = dict(options or {})
    if method in R_METHODS:
        cmd = [RSCRIPT, os.path.join(HERE, 'r_worker.R'), case, method, out, json.dumps(options)]
    else:
        cmd = [PY, os.path.join(HERE, 'py_worker.py'), case, method, out, json.dumps(options)]
    d0 = _disk_reads()
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ENV)
    ps = psutil.Process(proc.pid)
    peak, status = 0, None
    handles = {proc.pid: _open_handle(proc.pid)}
    while proc.poll() is None:
        try:
            for ch in ps.children(recursive=True):
                if ch.pid not in handles:
                    handles[ch.pid] = _open_handle(ch.pid)
        except psutil.Error:
            pass
        rss = tree_rss(ps)
        peak = max(peak, rss)
        el = time.perf_counter() - t0
        if budget_bytes is not None and rss > budget_bytes:
            status = 'OOM'
        elif el > cap_seconds:
            status = 'TIMEOUT'
        if status:
            for p in [ps] + ps.children(recursive=True):
                try:
                    p.kill()
                except psutil.Error:
                    pass
            break
        time.sleep(0.025)
    stdout, stderr = proc.communicate()
    wall = time.perf_counter() - t0
    d1 = _disk_reads()
    peaks = {pid: _peak_memory(h) for pid, h in handles.items()}
    peak_wset = sum(v[0] for v in peaks.values() if v[0]) or None      # upper bound for the tree
    peak_commit = sum(v[1] for v in peaks.values() if v[1]) or None
    if sys.platform == 'win32':
        import ctypes
        for h in handles.values():
            if h:
                ctypes.WinDLL('kernel32').CloseHandle(ctypes.c_void_p(h))
    res = {}
    rpath = os.path.join(out, 'result.json')
    if status is None and os.path.exists(rpath):
        with open(rpath, encoding='utf-8') as fh:
            res = json.load(fh)
    res.update({'method': method, 'wall_seconds': wall, 'peak_rss': max(peak, peak_wset or 0),
                'peak_rss_sampled': peak, 'peak_wset': peak_wset, 'peak_commit': peak_commit,
                'device_read_bytes': {k: d1[k] - d0.get(k, 0) for k in d1},
                'run_status': status or res.get('status', 'error'), 'options': options,
                'budget_bytes': budget_bytes})
    if res['run_status'] not in ('ok',):
        res['stderr_tail'] = stderr.decode('utf-8', 'replace')[-2000:]
    with open(os.path.join(out, 'run.json'), 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1)
    return res


if __name__ == '__main__':
    case, method, out = sys.argv[1:4]
    opts = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    print(json.dumps(run(case, method, out, opts), indent=1)[:3000])
