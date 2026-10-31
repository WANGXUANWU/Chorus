"""Quiet-machine gate for timing runs (amendment 21, addendum of 2026-10-24).

wait_quiet() blocks until processes outside the current process tree use less than `threshold`
percent of the whole CPU for `hold` seconds. Background samples the CPU time of every process
outside the tree while a run executes and reports the mean and peak background load in cores.
"""
import threading
import time
import psutil


def _tree(pid):
    try:
        p = psutil.Process(pid)
        return {pid} | {c.pid for c in p.children(recursive=True)}
    except psutil.Error:
        return {pid}


def _other_cpu(own, prev):
    """CPU seconds used by processes outside `own` since `prev` (dict pid -> seconds)."""
    cur, used = {}, 0.
    for p in psutil.process_iter(['pid', 'cpu_times']):
        pid = p.info['pid']
        ct = p.info['cpu_times']
        if pid in own or ct is None or pid == 0:
            continue
        s = ct.user + ct.system
        cur[pid] = s
        if pid in prev:
            used += max(0., s - prev[pid])
    return used, cur


def wait_quiet(threshold=10., hold=120., step=5., log=None):
    """Block until other processes use < threshold % of all cores for hold seconds."""
    ncpu = psutil.cpu_count() or 1
    own = _tree(psutil.Process().pid)
    _, prev = _other_cpu(own, {})
    quiet_since, waited, t_last = None, 0., time.time()
    while True:
        time.sleep(step)
        now = time.time()
        used, prev = _other_cpu(_tree(psutil.Process().pid), prev)
        pct = 100. * used / max(now - t_last, 1e-6) / ncpu
        t_last = now
        if pct < threshold:
            quiet_since = quiet_since or now
            if now - quiet_since >= hold:
                return waited
        else:
            quiet_since = None
            waited += step
            if log and int(waited) % 600 < step:
                log(f'waiting for a quiet machine: other processes at {pct:.0f}% CPU')


class Background:
    """Context manager sampling the CPU load of processes outside this process tree."""

    def __init__(self, step=2.):
        self.step, self.samples, self._stop = step, [], threading.Event()

    def _run(self):
        own = _tree(psutil.Process().pid)
        _, prev = _other_cpu(own, {})
        t_last = time.time()
        while not self._stop.wait(self.step):
            own = _tree(psutil.Process().pid)
            used, prev = _other_cpu(own, prev)
            now = time.time()
            self.samples.append(used / max(now - t_last, 1e-6))
            t_last = now

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=10)

    @property
    def mean(self):
        return float(sum(self.samples) / len(self.samples)) if self.samples else 0.

    @property
    def peak(self):
        return float(max(self.samples)) if self.samples else 0.
