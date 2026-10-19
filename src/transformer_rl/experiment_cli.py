"""兼容既有命令入口；实现位于职责包内。"""
from .research.experiment_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
