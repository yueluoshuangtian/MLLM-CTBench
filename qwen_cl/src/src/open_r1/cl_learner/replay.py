"""
ER (Experience Replay): mix prior-task samples into the current task's batches.

Implementation:
This learner is a thin wrapper — the actual data mixing happens at dataset
build time in sft.py's main() by passing `--replay_paths task1.json:task2.json`
(colon-separated) and `--replay_ratio 0.2`. The LazySupervisedDataset extends
its `list_data_dict` with random samples from prior datasets.

This wrapper exists so the unified --cl_method dispatch can treat 'replay'
uniformly. No state to keep across tasks (the prior data paths are passed
explicitly through the shell script).
"""

from .base import BaseCLearner, rank0_print


class ReplayLearner(BaseCLearner):

    name = "replay"

    def __init__(self, replay_paths=None, replay_ratio=0.2, **kwargs):
        super().__init__()
        # Just kept for logging; the actual mixing is done in dataset.py.
        self.replay_paths = replay_paths or []
        self.replay_ratio = float(replay_ratio)

    def before_train(self, task_id, model, tokenizer=None, **kwargs):
        rank0_print(
            f"[Replay] task_id={task_id} ratio={self.replay_ratio} "
            f"prior_data_paths={self.replay_paths}"
        )
