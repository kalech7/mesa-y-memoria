"""Incremental export preserves separate pass and phase-credit observations."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_TENSORBOARD = importlib.util.find_spec("tensorboard") is not None
if HAS_TENSORBOARD:
    from scripts.monitor_training import Exporter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@unittest.skipUnless(HAS_TENSORBOARD, "TensorBoard is installed in the monitor environment")
class TrainingMonitorTest(unittest.TestCase):
    def test_partial_pass_record_and_repoll_do_not_collapse_passes(self):
        with tempfile.TemporaryDirectory() as work:
            source, output = Path(work) / "source", Path(work) / "events"
            source.mkdir()
            progress = [
                {"stage": "started", "plan": {"rollout_optimization_passes": 2}},
                {"stage": "optimization_pass_completed", "step": 1, "loss": 999},
                {"stage": "training", "step": 1, "loss": -.15},
            ]
            (source / "progress.jsonl").write_text("".join(json.dumps(row) + "\n" for row in progress))
            first = dict(step=1, outer_update=1, optimization_pass=1, optimizer_step=1,
                         loss=.2, kl=.01, ratio_min=.9, ratio_max=1.1, grad_norm=.4,
                         clip_fraction_phase_mean=0., phase_losses={"select": .2})
            second = {**first, "optimization_pass": 2, "optimizer_step": 2, "loss": -.15,
                      "ratio_min": .7, "ratio_max": 1.4, "clip_fraction_phase_mean": .25}
            second_line = json.dumps(second) + "\n"
            path = source / "optimizer_passes.jsonl"
            path.write_text(json.dumps(first) + "\n" + second_line[:20])
            group = dict(optimizer_update=1, returns=[0., 1., 0., 1.], task_returns=[0., 1., 0., 1.],
                         std_task=.5, std_total=.5, semantic_action_diversity=2, empirical_action_entropy_nats=.69,
                         phase_returns={"select": [0., 1., 0., 1.]}, phase_std={"select": .5})
            (source / "group_diagnostics.jsonl").write_text(json.dumps(group) + "\n")
            exporter = Exporter(source, output)
            try:
                exporter.poll()
                accumulator = EventAccumulator(str(output)).Reload()
                self.assertEqual(len(accumulator.Scalars("optimization/pass_1/loss")), 1)
                self.assertAlmostEqual(accumulator.Scalars("optimization/pass_1/loss")[0].value, .2)
                self.assertNotIn("optimization/pass_2/loss", accumulator.Tags()["scalars"])
                with path.open("a") as stream:
                    stream.write(second_line[20:])
                exporter.poll()
                exporter.poll()
                accumulator.Reload()
                for pass_index, expected in ((1, .2), (2, -.15)):
                    events = accumulator.Scalars(f"optimization/pass_{pass_index}/loss")
                    self.assertEqual(len(events), 1)
                    self.assertEqual(events[0].step, 1)
                    self.assertAlmostEqual(events[0].value, expected)
                self.assertAlmostEqual(accumulator.Scalars("optimization/loss")[0].value, -.15)
                self.assertEqual(len(accumulator.Scalars("optimization/grad_norm")), 1)
                self.assertEqual(len(accumulator.Scalars("credit/group_1/select/return_std")), 1)
                self.assertAlmostEqual(accumulator.Scalars("credit/group_1/select/return_std")[0].value, .5)
            finally:
                exporter.close()


if __name__ == "__main__":
    unittest.main()
