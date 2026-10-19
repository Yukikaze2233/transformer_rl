"""Process-local compatibility for the installed native simulation runtime.

The launch owner selects the driver directory before starting Python. Neither
function edits shared SDK files, user settings or simulation parameters.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import inspect
import os
from pathlib import Path


def native_process_environment(environment, *, driver_library_directory=None):
    result = dict(environment)
    if driver_library_directory is not None:
        directory = Path(driver_library_directory)
        if not directory.is_absolute() or not (directory / 'libcuda.so').is_file():
            raise ValueError('native driver directory requires an absolute libcuda.so location')
        previous = result.get('LD_LIBRARY_PATH')
        result['LD_LIBRARY_PATH'] = str(directory) + (os.pathsep + previous if previous else '')
    return result


@contextmanager
def native_proxy_launch(warp, proxy_type):
    """Expose the SDK's exact Warp arrays to kernels that reject ProxyArray.

    This is needed by the installed Warp 1.13 / Isaac Lab 3 combination. Views,
    devices, streams and kernel arguments retain their original ownership. The
    caller enters after AppLauncher and keeps this scope around the SDK only.
    """
    original = warp.launch
    parameters = list(inspect.signature(original).parameters)
    names = ('inputs', 'outputs', 'adj_inputs', 'adj_outputs')
    positions = tuple(parameters.index(name) for name in names if name in parameters)
    statistics = {'arrays_unwrapped_zero_copy': 0}

    def unwrap(values):
        if not isinstance(values, (tuple, list)) or not any(isinstance(v, proxy_type) for v in values):
            return values
        statistics['arrays_unwrapped_zero_copy'] += sum(isinstance(v, proxy_type) for v in values)
        return type(values)(v.warp if isinstance(v, proxy_type) else v for v in values)

    @wraps(original)
    def launch(*args, **kwargs):
        args = list(args)
        for index in positions:
            if index < len(args):
                args[index] = unwrap(args[index])
        for name in names:
            if name in kwargs:
                kwargs[name] = unwrap(kwargs[name])
        return original(*args, **kwargs)

    warp.launch = launch
    try:
        yield statistics
    finally:
        if warp.launch is launch:
            warp.launch = original
