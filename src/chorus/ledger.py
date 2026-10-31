"""Memory ledger: an enforced bound M on the resident memory of the fitting process.

M0 is the resident set measured when the fit starts (interpreter, libraries, compiled
kernels and the inputs already loaded). Every array the fit allocates afterwards is charged
analytically, with a safety margin delta; a runtime guard compares the measured resident set
with the bound before each penalty level. The planner splits the remaining memory between
per-batch arrays and the Gram matrix of the union working set, whose width is capped. When
the working set would exceed the cap the engine evicts inactive blocks, then splits the
batch, and reports an infeasible budget when a single response does not fit.
"""
import numpy as np


class MemoryBudgetError(RuntimeError):
    """The budget cannot hold the fit of a single response."""


class BatchOverflow(Exception):
    """Raised by the engine when the union working set outgrows its Gram allowance."""

    def __init__(self, level, signature):
        super().__init__(f'working set overflow at level {level}')
        self.level, self.signature = level, signature


class WidthOverflow(Exception):
    """Raised under the statistics strategy when the candidate set of one response outgrows
    the sub-Gram allowance; the fit falls back to the working-set engine."""

    def __init__(self, level, width):
        super().__init__(f'candidate width {width} at level {level}')
        self.level, self.width = level, width


def _rss():
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except Exception:
        return 0


# resident memory added by importing chorus and loading its compiled kernels, measured as the peak
# working set of a fresh worker after warm-up minus the set after the numerical imports (0.17 GiB)
STARTUP_BYTES = 180 << 20


def preflight(M, planned_bytes, fixed_bytes=0, delta=None, eps=0.10, startup_bytes=STARTUP_BYTES, now=None):
    """Refuse a budget before any input is loaded or any kernel is compiled: the resident set now
    (measured by the caller before importing chorus when given), the start-up allowance of the
    kernels, the inputs the caller will keep resident and the fixed arrays of the fit must fit
    below M - delta."""
    d = int(delta if delta is not None else max(64 << 20, 0.05 * M))
    now = _rss() if now is None else now
    need = now + startup_bytes + planned_bytes + (1 + eps) * fixed_bytes
    if need > M - d:
        raise MemoryBudgetError(f'budget {M / 2**30:.2f} GiB is below the start-up requirement '
                                f'{need / 2**30:.2f} GiB (process {now / 2**30:.2f}, kernels '
                                f'{startup_bytes / 2**30:.2f}, resident inputs {planned_bytes / 2**30:.2f}, '
                                f'fixed arrays {fixed_bytes / 2**30:.2f}) plus the margin {d / 2**30:.2f} GiB')
    return need


class Ledger:
    """bytes per float64 value: 8. Fractions: half of the free memory for batch arrays,
    the rest for the Gram matrix with its growth transient and per-response copies."""

    def __init__(self, M, delta=None, eps=0.10, batch_share=0.5):
        self.M = int(M)
        self.delta = int(delta if delta is not None else max(64 << 20, 0.05 * M))
        self.eps, self.batch_share = eps, batch_share
        self.M0 = _rss()
        self.retained = 0
        self.events = {'evictions': 0, 'splits': 0, 'guard': 0, 'peak_seen': self.M0, 'stage2_fallback': 0,
                       'strategy_fallback': 0}

    def rss(self):
        r = _rss()
        self.events['peak_seen'] = max(self.events['peak_seen'], r)
        return r

    def free(self, fixed=0):
        """Bytes available above the start-of-fit resident set after fixed charges and the
        arrays retained across stages (the Gram matrices held by the statistics strategy)."""
        return (self.M - self.delta - self.M0 - (1 + self.eps) * (fixed + self.retained))

    def retain(self, nbytes):
        """Charge (or with a negative value release) arrays kept from one stage to the next."""
        self.retained = max(0, self.retained + int(nbytes))

    @staticmethod
    def per_response_stats(D, G, n_val, C):
        """Bytes per batch column of the statistics strategy: no array of n rows, only
        correlations, coefficients, gradients, best fits, gap temporaries (6D), group
        statistics (14G), validation residuals (3 n_val) and cross-fitting losses (9C)."""
        return 8 * (6 * D + 14 * G + 3 * n_val + 9 * C)

    def plan_stats(self, n, D, G, q, n_val, C, buffer_bytes=0, resident_design=0, max_batch=64,
                   disk=False, panel_cap=512 << 20, min_rows=256, m_floor=1024):
        """Plan of the statistics strategy, or None when the full Gram matrix (8 D^2 bytes) and
        its transients do not fit. Returns dict(B, gram_bytes, panel_bytes, m_sub). The
        transients of the scan (two row panels on disk) and of the solves (the sub-Gram copy of
        one response, bounded by m_sub) never coexist. Does not change the working-set plan."""
        fixed = buffer_bytes + resident_design + self.full_q(D, G, q, C)
        free = self.free(fixed)
        per = (1 + self.eps) * self.per_response_stats(D, G, n_val, C)
        gram = 8 * D * D
        if free <= (1 + self.eps) * gram:
            return None
        B = int(min(q, max_batch, max(1, (self.batch_share * (free - (1 + self.eps) * gram)) // per)))
        avail = (free - (1 + self.eps) * gram - B * per) / (1 + self.eps)
        if avail <= 0:
            return None
        panel = 0
        if disk:
            panel = int(min(panel_cap, avail / 2))
            if panel // (8 * D) < min(n, min_rows):
                return None
        m_sub = int(np.sqrt(0.5 * avail / 8))
        if m_sub < min(D, m_floor):
            return None
        self.fixed = fixed
        return {'B': B, 'gram_bytes': gram, 'panel_bytes': panel, 'm_sub': m_sub}

    @staticmethod
    def per_response(n, D, G, n_val, C):
        """Bytes per batch column of the stage-1 engine (responses, residuals and their
        copies, gradients, coefficients, best fits, group statistics, validation tracking)."""
        return 8 * (6 * n + 7 * D + 14 * G + 3 * n_val + 9 * C)

    @staticmethod
    def full_q(D, G, q, C):
        """Bytes of the arrays kept for all responses (stage-1 fits, statistics, tracking)."""
        return 8 * q * (2 * D + 4 * G + 3 * C)

    def plan(self, n, D, G, q, n_val, C, buffer_bytes, resident_design=0, max_batch=64):
        """Return (B, m_max): batch size and largest union working-set width."""
        fixed = buffer_bytes + resident_design + self.full_q(D, G, q, C)
        free = self.free(fixed)
        per = (1 + self.eps) * self.per_response(n, D, G, n_val, C)
        if free <= 0:
            raise MemoryBudgetError(f'budget {self.M / 2**30:.2f} GiB is below the resident baseline '
                                    f'{self.M0 / 2**30:.2f} GiB plus fixed arrays {fixed / 2**30:.2f} GiB')
        B = int(min(q, max_batch, max(1, (self.batch_share * free) // per)))
        self._free, self._per, self.fixed = free, per, fixed
        m_max = self.m_max_for(B)
        if m_max < 1:
            raise MemoryBudgetError('budget leaves no room for a working set')
        self.B, self.m_max = B, m_max
        return B, m_max

    def m_max_for(self, b):
        """Largest union working-set width for a batch of b responses: the Gram buffer, its
        growth transient and a per-response sub-Gram copy use what the batch arrays leave."""
        gram_bytes = self._free - b * self._per
        return int(np.sqrt(max(gram_bytes, 0) / (8 * 2.5 * (1 + self.eps))))

    def guard(self):
        """True when the measured resident set has entered the safety margin."""
        over = self.rss() > self.M - self.delta / 2
        if over:
            self.events['guard'] += 1
        return over
