"""静态检查已跟踪源码，不导入模型或运行测试。"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import tomllib


def check_source_boundaries(root, names):
    """源码静态依赖：正式与研究互不导入，物理与任务不依赖学习器。"""
    modules = set()
    bindings = {}
    for name in names:
        path = Path(name)
        if name.startswith('src/') and path.suffix == '.py':
            parts = list(path.with_suffix('').parts[1:])
            if parts[-1] == '__init__':
                parts.pop()
            modules.add('.'.join(parts))
            defined = set()
            for node in ast.walk(ast.parse((root / name).read_text())):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    defined.add(node.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    defined.add(node.id)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    defined.update(alias.asname or alias.name.split('.')[0] for alias in node.names)
            bindings['.'.join(parts)] = defined
    independent = {'control', 'tasks', 'simulation'}
    forbidden = {'learning', 'runtime', 'evaluation', 'policy'}
    for name in names:
        if not name.startswith('src/transformer_rl/') or not name.endswith('.py'):
            continue
        path = Path(name)
        parts = list(path.with_suffix('').parts[1:])
        package = parts[:-1]
        namespace = parts[1] if len(parts) > 2 else None
        boundary = parts[2] if len(parts) > 2 else None
        for node in ast.walk(ast.parse((root / name).read_text())):
            if isinstance(node, ast.ImportFrom):
                target = ('.'.join(package[:len(package) - node.level + 1]
                          + ([node.module] if node.module else [])) if node.level else node.module)
                targets = [target] if target else []
            elif isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            else:
                continue
            for target in targets:
                if not target.startswith('transformer_rl'):
                    continue
                if target not in modules:
                    raise ValueError(f'{name}:{node.lineno}: missing local import {target}')
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name != '*' and alias.name not in bindings[target] and target + '.' + alias.name not in modules:
                            raise ValueError(f'{name}:{node.lineno}: missing local name {target}.{alias.name}')
                if (namespace == 'formal' and target.startswith('transformer_rl.research')
                        or namespace == 'research' and target.startswith('transformer_rl.formal')):
                    raise ValueError(f'{name}:{node.lineno}: formal/research cross-import {target}')
                if (namespace == 'common'
                        and target.startswith(('transformer_rl.formal', 'transformer_rl.research'))
                        and path.name not in {'session_router.py', 'training_cli.py'}):
                    raise ValueError(f'{name}:{node.lineno}: shared mechanism imports training business {target}')
                if namespace != 'formal':
                    continue
                scope = target.removeprefix('transformer_rl.formal.').split('.')[0]
                if boundary in independent and scope in forbidden:
                    raise ValueError(f'{name}:{node.lineno}: {boundary} depends on {scope}')
                if boundary in {'contracts', 'policy'} and target.startswith('transformer_rl.formal.'):
                    raise ValueError(f'{name}:{node.lineno}: foundational module depends on {target}')
                if boundary == 'tasks' and scope == 'simulation':
                    raise ValueError(f'{name}:{node.lineno}: task definitions depend on native simulation')


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
    check_source_boundaries(root, names)
    print(f'Repository checks passed: {python_count} Python sources, {json_count} JSON configs, '
          f'version {project["version"]}; no model imports or tests executed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
