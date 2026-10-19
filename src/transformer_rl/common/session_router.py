"""按冻结会话标记路由，不在导入时加载网络或仿真。"""
from pathlib import Path

from .training_control import read_json


def open_session(value):
    root = Path(value).expanduser()
    if not root.exists():
        registry = read_json(Path.home() / ".config/transformer-rl/managed.json", {})
        if str(value) not in registry:
            raise ValueError(f"unknown managed session or queue: {value}")
        root = Path(registry[str(value)])
    if (root / 'structured.json').is_file():
        from ..formal.runtime.session import StructuredTrainingSession
        return StructuredTrainingSession(root)
    from ..research.session import TrainingSession
    from ..research.training_queue import TrainingQueue
    return TrainingQueue(root) if (root / "queue.json").is_file() else TrainingSession(root)
