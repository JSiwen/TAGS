class GroupLRScheduler:
    def __init__(self, cfg):
        self.cfg = cfg
        self._base_lrs = None

    def capture_base_lrs(self, gaussians):
        return {group["name"]: float((self._base_lrs or {}).get(id(group), group["lr"]))
                for group in gaussians.optimizer.param_groups}

    def restore_base_lrs(self, gaussians, base_lrs):
        import math
        groups = gaussians.optimizer.param_groups
        if set(base_lrs) != {group["name"] for group in groups}:
            raise ValueError("Checkpoint optimizer group names differ from current optimizer")
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in base_lrs.values()):
            raise ValueError("Invalid checkpoint base learning rate")
        self._base_lrs = {id(group): float(base_lrs[group["name"]]) for group in groups}

    def _stage(self, iteration):
        s1 = int(getattr(self.cfg, "lr_stage1_end", 6000))
        s2 = int(getattr(self.cfg, "lr_stage2_end", 18000))
        if iteration <= s1:
            return 1
        if iteration <= s2:
            return 2
        return 3

    def _multipliers(self, stage):
        geo = {
            1: float(getattr(self.cfg, "lr_geo_mult_stage1", 1.10)),
            2: float(getattr(self.cfg, "lr_geo_mult_stage2", 1.00)),
            3: float(getattr(self.cfg, "lr_geo_mult_stage3", 0.75)),
        }[stage]
        color = {
            1: float(getattr(self.cfg, "lr_color_mult_stage1", 0.75)),
            2: float(getattr(self.cfg, "lr_color_mult_stage2", 1.05)),
            3: float(getattr(self.cfg, "lr_color_mult_stage3", 0.80)),
        }[stage]
        return geo, color

    def apply(self, iteration, gaussians):
        if not getattr(self.cfg, "enable_group_lr_schedule", False):
            return {}
        if self._base_lrs is None:
            self._base_lrs = {
                id(param_group): float(param_group.get("lr", 0.0))
                for param_group in gaussians.optimizer.param_groups
            }
        stage = self._stage(iteration)
        geo_mult, color_mult = self._multipliers(stage)
        xyz_lr = None
        xyz_base = float(gaussians.xyz_scheduler_args(iteration)) if hasattr(gaussians, "xyz_scheduler_args") else None
        for param_group in gaussians.optimizer.param_groups:
            name = param_group.get("name")
            base_lr = self._base_lrs.get(id(param_group), float(param_group.get("lr", 0.0)))
            if name in ("xyz", "scaling", "rotation"):
                if name == "xyz" and xyz_base is not None:
                    param_group["lr"] = xyz_base * geo_mult
                else:
                    param_group["lr"] = base_lr * geo_mult
                if name == "xyz":
                    xyz_lr = param_group["lr"]
            elif name in ("f_dc", "f_rest", "opacity"):
                param_group["lr"] = base_lr * color_mult
        return {
            "train/lr_stage": float(stage),
            "train/lr_geo_mult": float(geo_mult),
            "train/lr_color_mult": float(color_mult),
            "train/lr_xyz": float(xyz_lr) if xyz_lr is not None else 0.0,
        }
