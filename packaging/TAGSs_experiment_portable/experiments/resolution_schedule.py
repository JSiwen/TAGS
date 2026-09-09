class ResolutionSchedule:
    def __init__(self, schedule_str):
        self.breakpoints = []
        for item in schedule_str.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"Invalid resolution schedule item '{item}'. Expected 'iteration:scale'.")
            it_str, scale_str = item.split(":")
            until_iter = int(it_str)
            scale = float(scale_str)
            if until_iter <= 0:
                raise ValueError("Resolution schedule iterations must be positive.")
            if scale <= 0:
                raise ValueError("Resolution schedule scales must be positive.")
            self.breakpoints.append((until_iter, scale))
        if not self.breakpoints:
            self.breakpoints = [(10**9, 1.0)]
        self.breakpoints.sort(key=lambda x: x[0])
        self.last_scale = None
        for idx in range(1, len(self.breakpoints)):
            if self.breakpoints[idx][0] <= self.breakpoints[idx - 1][0]:
                raise ValueError("Resolution schedule iteration breakpoints must be strictly increasing.")

    def get_scale(self, iteration):
        for until_iter, scale in self.breakpoints:
            if iteration <= until_iter:
                return scale
        return self.breakpoints[-1][1]

    def first_iteration_at_scale(self, target_scale):
        previous_until = 0
        for until_iter, scale in self.breakpoints:
            if scale == target_scale:
                return previous_until + 1
            previous_until = until_iter
        return None

    @property
    def full_resolution_start_iter(self):
        return self.first_iteration_at_scale(1.0)

    def scale_changed(self, iteration):
        curr = self.get_scale(iteration)
        changed = self.last_scale is None or curr != self.last_scale
        self.last_scale = curr
        return changed

    def scales(self):
        return sorted({scale for _, scale in self.breakpoints})

    def __str__(self):
        return ",".join(f"{until_iter}:{scale:g}" for until_iter, scale in self.breakpoints)
