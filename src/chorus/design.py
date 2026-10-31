"""Design sources: an in-memory Fortran array or a disk-resident column-block file.

Both expose the same three operations used by the engine, so the solver never needs
the whole design in memory: block reads, the streamed product X^T R, and X_W beta_W.
Disk reads bypass the OS file cache; requested and issued bytes are both counted.
"""
import json
import os
import time
import numpy as np


class InMemoryDesign:
    def __init__(self, x, groups):
        self.x = np.asfortranarray(x, dtype=np.float64)
        self.n, self.D = self.x.shape
        self.groups = np.asarray(groups, dtype=np.int64)
        self.G = len(self.groups)
        self.read_bytes = 0

    def nbytes_resident(self):
        return self.x.nbytes

    def cols(self, idx):
        """Columns of the groups in idx, concatenated (n, sum d)."""
        if len(idx) == 0:
            return np.empty((self.n, 0))
        return np.concatenate([self.x[:, self.groups[j, 0]:self.groups[j, 1]] for j in idx], axis=1)

    def xt_dot(self, r):
        return self.x.T @ r

    def x_dot_cols(self, idx, coef):
        if len(idx) == 0:
            return np.zeros((self.n, coef.shape[1]))
        return self.cols(idx) @ coef

    def cross(self, idx_a, idx_b):
        """X_a^T X_b / n for group lists a and b."""
        if len(idx_a) == 0 or len(idx_b) == 0:
            return np.zeros((self._width(idx_a), self._width(idx_b)))
        return self.cols(idx_a).T @ self.cols(idx_b) / self.n

    def stats_scan(self, y, panel_bytes=None):
        """Full sufficient statistics X^T X / n (C-contiguous) and X^T Y / n, with timings."""
        t0 = time.perf_counter()
        g = self.x.T @ self.x          # symmetric rank-k update, no temporary
        g /= self.n
        t1 = time.perf_counter()
        c = self.x.T @ np.asarray(y, dtype=np.float64) / self.n
        return np.ascontiguousarray(g), c, {'scan_seconds': time.perf_counter() - t0,
                                           'syrk_seconds': t1 - t0, 'read_seconds': 0.}

    def _width(self, idx):
        return int(sum(self.groups[j, 1] - self.groups[j, 0] for j in idx))


class DiskDesign:
    """Column-major float64 design file of shape (n, D), read with unbuffered positioned reads.

    Group columns are contiguous, so a block read is one sequential range. Full passes
    (X^T R) read column chunks of at most `chunk_bytes` with a one-chunk prefetch thread;
    products X_W beta_W read whole group panels; Gram blocks stream row chunks. Reads bypass
    the OS file cache (FILE_FLAG_NO_BUFFERING), so every pass is a device read, and only the
    chunk buffers, never the design, reside in memory. read_bytes counts the bytes the
    algorithm requests; issued_bytes the sector-aligned bytes requested from the device.
    """

    def __init__(self, path, chunk_bytes=64 << 20, buffered=False, prefetch=True):
        from .winio import UnbufferedFile
        meta = json.loads(open(path + '.json', encoding='utf-8').read())
        self.n, self.D = meta['n'], meta['D']
        self.dtype = np.dtype(meta['dtype'])
        self.groups = np.asarray(meta['groups'], dtype=np.int64)
        self.G = len(self.groups)
        self.path = path
        self.file = UnbufferedFile(path, buffered_fallback=buffered)
        self.chunk_bytes = int(chunk_bytes)
        self.prefetch = prefetch
        self.read_bytes = 0
        self._pool = None

    @property
    def issued_bytes(self):
        return self.file.issued_bytes

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        self.file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def nbytes_resident(self):
        return 0

    def buffer_bytes(self):
        """Bytes of read buffers that can be resident at once (for the memory ledger)."""
        return 3 * self.chunk_bytes

    def _cols_raw(self, c0, c1, slot):
        """View (n, c1 - c0) Fortran float64 of columns c0..c1, valid until slot is reused."""
        isz = self.dtype.itemsize
        raw = self.file.read(c0 * self.n * isz, (c1 - c0) * self.n * isz, slot)
        self.read_bytes += (c1 - c0) * self.n * isz
        blk = raw.view(self.dtype).reshape(c1 - c0, self.n).T
        return blk if self.dtype == np.float64 else blk.astype(np.float64, order='F')

    def _read(self, c0, c1):
        return np.array(self._cols_raw(c0, c1, 'tmp'), order='F')

    def _chunks(self, ranges):
        """Yield column blocks for consecutive ranges, prefetching the next one."""
        if not self.prefetch or len(ranges) < 2:
            for i, (c0, c1) in enumerate(ranges):
                yield self._cols_raw(c0, c1, i % 2)
            return
        from concurrent.futures import ThreadPoolExecutor
        if self._pool is None:
            self._pool = ThreadPoolExecutor(1)
        fut = self._pool.submit(self._cols_raw, *ranges[0], 0)
        for i in range(len(ranges)):
            blk = fut.result()
            if i + 1 < len(ranges):
                fut = self._pool.submit(self._cols_raw, *ranges[i + 1], (i + 1) % 2)
            yield blk

    def cols(self, idx):
        """Full columns of the groups in idx (use only for small idx or small n)."""
        if len(idx) == 0:
            return np.empty((self.n, 0))
        out = np.empty((self.n, self._width(idx)), order='F')
        pos = 0
        for j in idx:
            a, b = self.groups[j]
            out[:, pos:pos + b - a] = self._cols_raw(a, b, 'tmp')
            pos += b - a
        return out

    def xt_dot(self, r, out=None):
        out = np.empty((self.D, r.shape[1])) if out is None else out
        step = max(1, self.chunk_bytes // (8 * self.n))
        ranges = [(c0, min(self.D, c0 + step)) for c0 in range(0, self.D, step)]
        for (c0, c1), blk in zip(ranges, self._chunks(ranges)):
            np.dot(blk.T, r, out=out[c0:c1])
        return out

    def _colidx(self, idx):
        return np.concatenate([np.arange(*self.groups[j]) for j in idx])

    def _row_step(self, width):
        return max(1024, self.chunk_bytes // (8 * max(width, 1)))

    def _rows(self, r0, r1, cols, out, slot='rows'):
        # one contiguous segment per column (column-major file)
        isz = self.dtype.itemsize
        for k, c in enumerate(cols):
            out[:, k] = self.file.read((int(c) * self.n + r0) * isz, (r1 - r0) * isz, slot).view(self.dtype)
        self.read_bytes += (r1 - r0) * len(cols) * isz
        return out

    def stats_scan(self, y, panel_bytes=256 << 20):
        """One pass forming X^T X / n and X^T Y / n from row panels of all columns.

        Panels of at most panel_bytes are filled by one positioned read per column; a reader
        thread fills the next panel while the current one updates the lower triangle of the
        Gram matrix in place (BLAS dsyrk, no D x D temporary). Returns the C-contiguous Gram
        matrix, the correlations and the scan timings."""
        from concurrent.futures import ThreadPoolExecutor
        from scipy.linalg.blas import dsyrk
        from .kernels import symmetrize_lower
        t0 = time.perf_counter()
        n, D = self.n, self.D
        y = np.asarray(y, dtype=np.float64)
        rows = int(max(1, min(n, panel_bytes // (8 * D))))
        ranges = [(r0, min(n, r0 + rows)) for r0 in range(0, n, rows)]
        cols = np.arange(D)
        bufs = [np.empty((rows, D), order='F'), np.empty((rows, D), order='F') if len(ranges) > 1 else None]
        g = np.zeros((D, D), order='F')
        c = np.zeros((D, y.shape[1]))
        t_syrk = 0.
        t_wait = 0.

        def fill(i):
            r0, r1 = ranges[i]
            return self._rows(r0, r1, cols, bufs[i % 2][:r1 - r0], slot='stats')

        with ThreadPoolExecutor(1) as pool:
            fut = pool.submit(fill, 0)
            for i, (r0, r1) in enumerate(ranges):
                tw = time.perf_counter()
                p = fut.result()
                t_wait += time.perf_counter() - tw
                if i + 1 < len(ranges):
                    fut = pool.submit(fill, i + 1)
                ts = time.perf_counter()
                g = dsyrk(1.0, p, beta=1.0, c=g, trans=1, lower=1, overwrite_c=1)
                t_syrk += time.perf_counter() - ts
                c += p.T @ y[r0:r1]
        del bufs
        symmetrize_lower(g)
        g /= n
        c /= n
        return g.T, c, {'scan_seconds': time.perf_counter() - t0, 'syrk_seconds': t_syrk,
                        'read_seconds': t_wait, 'panel_rows': rows}

    def probe(self, nbytes=64 << 20):
        """Sequential read bandwidth of the device (bytes per second), measured once on the
        first column chunk and counted in read_bytes."""
        if getattr(self, '_bw', None) is None:
            step = max(1, min(self.D, nbytes // (8 * self.n)))
            t0 = time.perf_counter()
            self._cols_raw(0, step, 'tmp')
            self._bw = step * self.n * self.dtype.itemsize / max(time.perf_counter() - t0, 1e-6)
        return self._bw

    def x_dot_cols(self, idx, coef, out=None):
        """X_W beta_W, reading whole group panels (contiguous) one at a time."""
        out = np.zeros((self.n, coef.shape[1])) if out is None else out
        out[:] = 0.
        pos = 0
        for j in idx:
            a, b = self.groups[j]
            w = int(b - a)
            cj = coef[pos:pos + w]
            if np.any(cj):
                out += self._cols_raw(a, b, 'panel') @ cj
            pos += w
        return out

    def _width(self, idx):
        return int(sum(self.groups[j, 1] - self.groups[j, 0] for j in idx))

    def cross(self, idx_a, idx_b):
        """X_a^T X_b / n streamed over row chunks; only chunk-sized buffers reside."""
        wa, wb = self._width(idx_a), self._width(idx_b)
        acc = np.zeros((wa, wb))
        if wa == 0 or wb == 0:
            return acc
        ca, cb = self._colidx(idx_a), self._colidx(idx_b)
        same = len(ca) == len(cb) and np.array_equal(ca, cb)
        step = min(self.n, self._row_step(wa + (0 if same else wb)))
        A = np.empty((step, wa), order='F')
        Bm = A if same else np.empty((step, wb), order='F')
        for r0 in range(0, self.n, step):
            r1 = min(self.n, r0 + step)
            a_ = self._rows(r0, r1, ca, A[:r1 - r0])
            b_ = a_ if same else self._rows(r0, r1, cb, Bm[:r1 - r0])
            acc += a_.T @ b_
        return acc / self.n


def write_disk_design(path, n, groups, fill, dtype='float32'):
    """Create a column-major memory map and let fill(mm) write it column-block-wise."""
    groups = np.asarray(groups, dtype=np.int64)
    D = int(groups[-1, 1])
    mm = np.memmap(path, dtype=dtype, mode='w+', shape=(n, D), order='F')
    fill(mm)
    mm.flush()
    del mm
    with open(path + '.json', 'w', encoding='utf-8') as fh:
        json.dump({'n': int(n), 'D': D, 'dtype': str(np.dtype(dtype)), 'groups': groups.tolist(),
                   'bytes': int(os.path.getsize(path))}, fh)
    return path
