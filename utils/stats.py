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

def modulate_x_by_y(x, y):
    """
    Modulate x so that it is sharply reduced if y is < 0.3, tapering to no effect as y approaches 1.

    Use case example:
    Modulate a confidence interval based on Cohen's d value.
    
    Weak judges have low ci95 ranges for each model they score.
    For a strong judge that's a good thing, but for the weak judge
    it just means they score *everything* within a narrow band.

    Here we compensate for this by modulating the ci95 range by a
    different measure of separability (cohen's d). When ci95 is
    large but cohen's d is small, the modulated value is also small.
    """

    def modulation_factor(d):
        if d <= 0.3:
            # Steeper rise to reach ~0.95 by 0.3
            return 3.17 * d * (1 - 0.15 * d)
        else:
            # Smooth curve approaching 1 after 0.3 
            t = (d - 0.3) / 0.7  # normalize remaining part to 0-1
            base = 3.17 * 0.3 * (1 - 0.15 * 0.3)  # value at d=0.3
            return base + (1 - base) * (t * (2 - t))
   
    return x * modulation_factor(y)