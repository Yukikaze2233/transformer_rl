"""Isaac Lab 启动与设置边界；不改变环境、控制器或学习算法。"""
from __future__ import annotations

import sys


class LabSimulation:
    """显式选择 Kit PhysX；GA 与 beta 的启动差异集中处理。"""

    def __init__(self, device: str):
        from isaaclab import app

        self._scope = None
        self._application = None
        self.physics = None
        self._cuda_index = None
        if device.startswith('cuda'):
            self._cuda_index = int(device.split(':', 1)[1]) if ':' in device else 0
        self.declarative_scene = callable(getattr(app, 'launch_simulation', None))
        arguments = dict(headless=True, device=device, enable_cameras=False, livestream=0)
        argv = sys.argv
        try:
            sys.argv = [argv[0]]
            if self.declarative_scene:
                from isaaclab_physx.physics import PhysxCfg

                # 显式 PhysxCfg 避免 GA 自动选择其他求解器或 kitless 后端。
                self._scope = app.launch_simulation(PhysxCfg(), arguments)
                self.physics = self._scope.__enter__()
            else:
                self._application = app.AppLauncher(**arguments).app
        finally:
            sys.argv = argv

    def configure_settings(self):
        from isaaclab.app import settings_manager

        if self.declarative_scene:
            settings = settings_manager.get_settings_manager()
            if self._cuda_index is not None:
                settings.set('/physics/cudaDevice', self._cuda_index)
                settings.set('/physics/suppressReadback', True)
        else:
            settings = settings_manager.SettingsManager.instance()
            settings.initialize_carb_settings()
            if self._cuda_index is not None:
                settings.set_int('/physics/cudaDevice', self._cuda_index)
                settings.set_bool('/physics/suppressReadback', True)

    def close(self):
        # 失败路径由 worker 先保存原异常并非零退出，不在这里吞掉异常。
        if self._scope is not None:
            self._scope.__exit__(None, None, None)
            self._scope = None
        elif self._application is not None:
            self._application.close()
            self._application = None
