import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

# Evaluate the central 100 m x 100 m area inside a 128 m tile at 0.5 m spacing.
CORE=np.zeros((256,256),bool)
CORE[28:228,28:228]=True

def components(mask):
    """Label foreground objects using SciPy's default 2-D four-neighbour connectivity."""
    lab, n = ndimage.label(mask)
    return [lab == i for i in range(1, n + 1)]

def selected(items):
    """Keep 3 <= area < 200 m2 components whose centroid lies in the central core."""
    out = []
    edge_core = 0
    for original_id, m in enumerate(items, 1):
        if m.shape != CORE.shape:
            raise ValueError('Object eligibility requires a 256 x 256 mask; crop to the evaluation tile first')
        yy, xx = np.where(m)
        if not len(yy):
            continue
        edge = bool(np.any(yy == 0) | np.any(xx == 0) | np.any(yy == 255) | np.any(xx == 255))
        edge_core += int(edge and np.any(m & CORE))
        area = len(yy) * 0.25
        if 3 <= area < 200 and 28 <= xx.mean() < 228 and (28 <= yy.mean() < 228):
            out.append({'original_id': original_id, 'mask': m, 'area': area, 'edge': edge})
    return (out, edge_core)

def match_matrix(ious, threshold):
    if not ious.size:
        return []
    valid = ious >= threshold
    # The fractional IoU bonus breaks ties without outweighing one valid match.
    cost = valid.astype(float) * (1 + ious / (min(ious.shape) + 1))
    rr, cc = linear_sum_assignment(cost, maximize=True)
    return [(int(r), int(c), float(ious[r, c])) for r, c in zip(rr, cc) if valid[r, c]]

def score(ref, pred, t):
    ious = np.zeros((len(ref), len(pred)))
    for i, r in enumerate(ref):
        for j, p in enumerate(pred):
            inter = np.count_nonzero(r['mask'] & p['mask'])
            if inter:
                ious[i, j] = inter / (r['mask'].sum() + p['mask'].sum() - inter)
    pairs = match_matrix(ious, t)
    mr = {a for a, b, v in pairs}
    mp = {b for a, b, v in pairs}
    small = [i for i, r in enumerate(ref) if r['area'] < 10]
    return ({'reference_count': len(ref), 'predicted_count': len(pred), 'tp': len(pairs), 'fp': len(pred) - len(pairs), 'fn': len(ref) - len(pairs), 'small_reference_count': len(small), 'small_fn': sum((i not in mr for i in small)), 'small_predicted_count': sum((p['area'] < 10 for p in pred)), 'small_fp': sum((p['area'] < 10 and i not in mp for i, p in enumerate(pred))), 'count_error': len(pred) - len(ref), 'absolute_count_error': abs(len(pred) - len(ref))}, pairs)

def div(a, b):
    return a / b if b else None

def aggregate(rows):
    keys = ['pixel_tp', 'pixel_fp', 'pixel_fn', 'reference_count', 'predicted_count', 'tp', 'fp', 'fn', 'small_reference_count', 'small_fn', 'small_predicted_count', 'small_fp', 'count_error', 'absolute_count_error']
    s = {k: sum((r[k] for r in rows)) for k in keys}
    return {**s, 'n': len(rows), 'pixel_precision': div(s['pixel_tp'], s['pixel_tp'] + s['pixel_fp']), 'pixel_recall': div(s['pixel_tp'], s['pixel_tp'] + s['pixel_fn']), 'pixel_iou': div(s['pixel_tp'], s['pixel_tp'] + s['pixel_fp'] + s['pixel_fn']), 'object_precision': div(s['tp'], s['tp'] + s['fp']), 'object_recall': div(s['tp'], s['tp'] + s['fn']), 'object_f1': div(2 * s['tp'], 2 * s['tp'] + s['fp'] + s['fn']), 'small_unmatched_rate': div(s['small_fn'], s['small_reference_count']), 'count_MAE': div(s['absolute_count_error'], len(rows)), 'count_bias_mean': div(s['count_error'], len(rows))}
