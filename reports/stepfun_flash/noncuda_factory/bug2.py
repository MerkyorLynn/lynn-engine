def merge_intervals(intervals):
    """Merge overlapping intervals. Returns list of (start, end) tuples."""
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]
