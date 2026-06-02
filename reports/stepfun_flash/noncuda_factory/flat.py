def flatten(x):
    """Return a flat list of all scalar elements from an arbitrarily nested
    structure of lists and tuples (any depth)."""

    def _flatten(obj, out):
        if isinstance(obj, (list, tuple)):
            for item in obj:
                _flatten(item, out)
        else:
            out.append(obj)

    result = []
    _flatten(x, result)
    return result
