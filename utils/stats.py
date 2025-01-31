def clamp(x: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a value between low and high."""
    return max(low, min(x, high))

def normalize(val, min_val, max_val, bigger_is_better=True):
    if max_val <= min_val:
        return 0.0
    norm = (val - min_val) / (max_val - min_val)
    if not bigger_is_better:
        norm = 1.0 - norm
    return clamp(norm)