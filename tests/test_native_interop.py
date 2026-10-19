from types import SimpleNamespace

import pytest

from transformer_rl._native_interop import native_process_environment, native_proxy_launch


def test_driver_path_is_private_to_the_child_environment(tmp_path):
    directory = tmp_path / 'driver'
    directory.mkdir()
    (directory / 'libcuda.so').touch()
    original = {'LD_LIBRARY_PATH': '/existing', 'CUDA_VISIBLE_DEVICES': 'selected'}
    child = native_process_environment(original, driver_library_directory=directory)
    assert child == {'LD_LIBRARY_PATH': str(directory) + ':/existing', 'CUDA_VISIBLE_DEVICES': 'selected'}
    assert original['LD_LIBRARY_PATH'] == '/existing'
    assert native_process_environment(original) == original
    with pytest.raises(ValueError):
        native_process_environment(original, driver_library_directory=tmp_path / 'missing')


@pytest.mark.parametrize('argument', ['inputs', 'outputs', 'adj_inputs', 'adj_outputs'])
@pytest.mark.parametrize('positional', [False, True])
def test_proxy_launch_preserves_native_storage_and_execution_arguments(argument, positional):
    received = {}

    def original(kernel, dim, inputs=None, outputs=None, adj_inputs=None, adj_outputs=None, *, stream=None):
        received.update(locals())
        return 'launched'

    class Proxy:
        def __init__(self, array):
            self.warp = array

        @property
        def torch(self):
            raise AssertionError('interop must not convert through a tensor')

    warp = SimpleNamespace(launch=original)
    array, unchanged, kernel, stream = object(), object(), object(), object()
    values = (Proxy(array), unchanged)
    with native_proxy_launch(warp, Proxy) as statistics:
        if positional:
            arguments = [kernel, 19, None, None, None, None]
            arguments[2 + ['inputs', 'outputs', 'adj_inputs', 'adj_outputs'].index(argument)] = values
            result = warp.launch(*arguments, stream=stream)
        else:
            result = warp.launch(kernel, 19, **{argument: values}, stream=stream)
        assert result == 'launched'
        assert received[argument] == (array, unchanged)
        assert received['kernel'] is kernel and received['stream'] is stream and received['dim'] == 19
        assert statistics['arrays_unwrapped_zero_copy'] == 1
    assert warp.launch is original


def test_plain_arrays_are_not_copied_and_kernel_failure_restores_launch():
    values = [object()]

    def original(kernel, dim, inputs=None):
        assert inputs is values
        raise RuntimeError('kernel failure')

    warp = SimpleNamespace(launch=original)
    with pytest.raises(RuntimeError, match='kernel failure'):
        with native_proxy_launch(warp, type('Proxy', (), {})):
            warp.launch(object(), 1, inputs=values)
    assert warp.launch is original
