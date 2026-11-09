import numpy as np

def transform(raw, kind, amount):
    if raw.ndim != 3 or raw.shape[0] != 4:
        raise ValueError('Expected four bands with shape (4, height, width)')
    if kind not in {'base', 'gain', 'nir', 'offset', 'gamma'}:
        raise ValueError('Unknown photometric scenario: ' + str(kind))
    if not np.isfinite(amount) or (kind in {'gain', 'nir', 'gamma'} and amount <= 0):
        raise ValueError('Scenario amount must be finite; gain, nir and gamma require a positive amount')
    if not np.isfinite(raw).all():
        raise ValueError('Non-finite input')
    a = raw.astype(np.float32).copy()
    valid = np.any(raw != 0, axis=0)
    if not valid.any():
        return a
    if kind == 'gain':
        a *= amount
    elif kind == 'nir':
        a[3] *= amount
    elif kind in ['offset', 'gamma']:
        for k in range(4):
            lo = a[k, valid].min()
            hi = a[k, valid].max()
            span = hi - lo
            if kind == 'offset':
                a[k, valid] += amount * span
            elif span > 0:
                u = (a[k, valid] - lo) / span
                assert u.min() >= 0 and u.max() <= 1
                a[k, valid] = lo + span * np.power(u, amount)
    a[:, ~valid] = 0
    assert np.isfinite(a).all()
    return a
