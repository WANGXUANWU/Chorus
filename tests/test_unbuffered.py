import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from chorus import InMemoryDesign, DiskDesign, write_disk_design


def _make(tmp_path, n=1001, p=7, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p * d))
    groups = np.array([[j * d, (j + 1) * d] for j in range(p)])
    path = str(tmp_path / 'x.f64')
    write_disk_design(path, n, groups, lambda mm: mm.__setitem__(slice(None), X), dtype='float64')
    return X, groups, path


def test_reader_matches_memory(tmp_path):
    X, groups, path = _make(tmp_path)          # n = 1001: column offsets are not sector aligned
    mem = InMemoryDesign(X, groups)
    for buffered in (False, True):
        dd = DiskDesign(path, chunk_bytes=3 * 8 * 1001, buffered=buffered)   # forces several chunks
        R = np.random.default_rng(1).standard_normal((len(X), 4))
        assert np.array_equal(dd.cols([0, 3, 6]), mem.cols([0, 3, 6]))
        assert np.allclose(dd.xt_dot(R), mem.xt_dot(R), atol=1e-12)
        coef = np.random.default_rng(2).standard_normal((15, 4))
        assert np.allclose(dd.x_dot_cols([1, 2, 6], coef), mem.x_dot_cols([1, 2, 6], coef), atol=1e-12)
        assert np.allclose(dd.cross([0, 1], [2, 6]), mem.cross([0, 1], [2, 6]), atol=1e-12)
        assert np.allclose(dd.cross([4, 5], [4, 5]), mem.cross([4, 5], [4, 5]), atol=1e-12)
        assert dd.read_bytes > 0 and dd.issued_bytes >= dd.read_bytes
        if not buffered:
            assert dd.file.unbuffered
        dd.close()
