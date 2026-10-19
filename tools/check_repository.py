"""Check tracked source hygiene without importing models or running tests."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import tomllib


def main():
    root = Path(__file__).resolve().parents[1]
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    names = [name for name in names if name]
    forbidden_suffixes = ('.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.onnx.data',
                          '.npy', '.npz', '.bag', '.bag.active', '.mcap', '.db3', '.pyc', '.pyo', '.whl', '.log')
    runtime_roots = {'runs', 'logs', 'outputs', 'artifacts', 'checkpoints', 'datasets',
                     'data', 'downloads', 'snapshots', 'recordings', '.venv', '.cache'}
    violations = [name for name in names if Path(name).parts[0] in runtime_roots
                  or name.lower().endswith(forbidden_suffixes)]
    if violations:
        raise ValueError('runtime artifacts must stay outside the tracked source: ' + repr(violations))
    ignored = subprocess.check_output(['git', 'ls-files', '-ci', '--exclude-standard'], cwd=root, text=True)
    if ignored:
        raise ValueError('tracked files also match ignore rules: ' + ignored)
    python_count = json_count = 0
    for name in names:
        path = root / name
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=name)
            python_count += 1
        elif name.startswith('configs/') and path.suffix == '.json':
            json.loads(path.read_text())
            json_count += 1
    project = tomllib.loads((root / 'pyproject.toml').read_text())['project']
    tree = ast.parse((root / 'src/transformer_rl/__init__.py').read_text())
    versions = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == '__version__' for target in node.targets)]
    if versions != [project['version']]:
        raise ValueError('package metadata and source version differ')
    print(f'Repository checks passed: {python_count} Python sources, {json_count} JSON configs, '
          f'version {project["version"]}; no model imports or tests executed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
