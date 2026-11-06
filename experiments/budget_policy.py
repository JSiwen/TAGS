"""Integer-exact 0.5% lower budget and conservative prune allowance."""
def matched_budget_floor(target):
    target = int(target)
    if target <= 0:
        raise ValueError("Positive Gaussian target required")
    return target - target // 200


def allowed_geometry_prunes(points, candidates, floor):
    if floor <= 0:
        return int(candidates)
    if points < floor:
        raise ValueError("Model is already below budget floor; cannot restore count by pruning")
    return min(int(candidates), max(0, int(points) - int(floor)))
