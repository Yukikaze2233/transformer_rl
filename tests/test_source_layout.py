"""源码分层的导入边界与命令入口回归；仅在 Kaiser 执行。"""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('module', [
    'transformer_rl.training_cli',
    'transformer_rl.formal.runtime.session',
    'transformer_rl.common.session_router',
])
def test_management_imports_load_neither_research_nor_torch(module):
    code = (f'import importlib,sys; importlib.import_module({module!r}); '
            "assert 'torch' not in sys.modules; "
            "assert not any(name.startswith('transformer_rl.research') for name in sys.modules)")
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
        timeout=10, env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')})
    assert result.returncode == 0, result.stderr
