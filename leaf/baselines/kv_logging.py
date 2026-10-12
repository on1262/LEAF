"""verl trainer that also samples vLLM's KV-cache stats (exploration tuning smokes).

The verl baselines (SDPO / GRPO / SCoRe / SEED-RL) run the fork's plain
``RayPPOTrainer``. verl's vLLM server never prints vLLM's periodic stats line
(``VLLM_LOGGING_LEVEL=WARN``, no API-server log loop), so even with
``disable_log_stats=False`` the job log holds neither the KV capacity nor its
usage. This subclass (``+trainer.trainer_class=leaf.baselines.kv_logging.KvLoggingTrainer``,
added by :func:`leaf.baselines.sdpo.run_trainer` when the config sets
``verl_kv_poller``) only starts :class:`leaf.explore.server_rollout.KvPoller` on the
rollout server for the duration of ``fit()``: one ``kv_stats.jsonl`` row every
5 s (KV usage, running / waiting requests, cumulative preemptions, capacity in
tokens) tagged with the current ``global_steps``. Logging only: the training
loop is the parent's, untouched.
"""

from __future__ import annotations

import time
from pathlib import Path

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class KvLoggingTrainer(RayPPOTrainer):
    def fit(self):
        poller = None
        try:
            from leaf.explore.server_rollout import KvPoller

            class _StepPoller(KvPoller):
                """KvPoller whose step is the trainer's current global step."""

                trainer = None

                def run(self) -> None:  # noqa: D102 -- same loop, step read live
                    import json

                    while not self._halt.wait(self.interval):
                        step = int(getattr(self.trainer, "global_steps", 0) or 0)
                        try:
                            row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "step": step,
                                   **self._scrape()}
                        except Exception as error:  # a missed sample must not stop the run
                            row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "step": step,
                                   "error": str(error)}
                        with self.path.open("a") as handle:
                            handle.write(json.dumps(row) + "\n")

            output = Path(self.config.trainer.rollout_data_dir).parent
            server = self.async_rollout_manager.server_handles[0]
            poller = _StepPoller(server, output / "kv_stats.jsonl")
            poller.trainer = self
            poller.start()
        except Exception as error:  # noqa: BLE001 -- stats are best effort
            print(f"KV poller not started: {error!r}", flush=True)
            poller = None
        try:
            return super().fit()
        finally:
            if poller is not None:
                poller.stop()
