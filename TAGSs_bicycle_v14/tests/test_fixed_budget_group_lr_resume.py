import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("group_lr_test", ROOT / "experiments/group_lr_scheduler.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class GroupLRResumeTests(unittest.TestCase):
    def cfg(self):
        return SimpleNamespace(enable_group_lr_schedule=True, lr_geo_mult_stage3=.4, lr_color_mult_stage3=.55)
    def model(self):
        return SimpleNamespace(optimizer=SimpleNamespace(param_groups=[
            {"name":"xyz", "lr":.001}, {"name":"scaling", "lr":.005},
            {"name":"rotation", "lr":.001}, {"name":"f_dc", "lr":.0025},
            {"name":"f_rest", "lr":.000125}, {"name":"opacity", "lr":.025}]),
            xyz_scheduler_args=lambda step:.00001)
    def test_saved_and_legacy_fresh_bases_match_uninterrupted(self):
        continuous=self.model(); sched=module.GroupLRScheduler(self.cfg())
        fresh=sched.capture_base_lrs(continuous)
        sched.apply(1,continuous); sched.apply(25000,continuous)
        saved=sched.capture_base_lrs(continuous)
        for bases in (saved, fresh):
            resumed=self.model()
            for a,b in zip(resumed.optimizer.param_groups,continuous.optimizer.param_groups):
                a["lr"]=b["lr"]
            resumed_sched=module.GroupLRScheduler(self.cfg())
            resumed_sched.restore_base_lrs(resumed,bases)
            resumed_sched.apply(25001,resumed); sched.apply(25001,continuous)
            for a,b in zip(resumed.optimizer.param_groups,continuous.optimizer.param_groups):
                self.assertAlmostEqual(a["lr"],b["lr"])
        self.assertAlmostEqual(continuous.optimizer.param_groups[1]["lr"],.002)
        self.assertAlmostEqual(continuous.optimizer.param_groups[3]["lr"],.001375)
    def test_mismatched_groups_rejected(self):
        with self.assertRaisesRegex(ValueError,"group names differ"):
            module.GroupLRScheduler(self.cfg()).restore_base_lrs(self.model(), {"wrong":.1})

if __name__ == "__main__": unittest.main()
