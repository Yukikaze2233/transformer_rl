"""Export the exact trained packed-frame mean and verify independent runtimes."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import warnings

import numpy as np
import torch

from .checkpoint import _publish_new_files
from .frame_checkpoint import load_frame_checkpoint


@torch.no_grad()
def export_frame_policy(checkpoint, directory, *, onnx=True):
    model, _, config, update, metadata, _ = load_frame_checkpoint(checkpoint)
    policy = model.actor.policy.eval()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    stream = io.BytesIO()
    scripted = torch.jit.script(policy)
    torch.jit.save(scripted, stream)
    data = {directory / "policy.pt": stream.getvalue()}
    generator = torch.Generator().manual_seed(9271)
    cases = [torch.zeros(1, config.model.history_length, config.model.frame_dim)]
    for batch in (1, 7):
        frames = torch.randn(batch, config.model.history_length, config.model.frame_dim, generator=generator)
        cases.extend((frames, frames[:, -1:].expand_as(frames).clone()))
    script_error = 0.0
    for frames in cases:
        error = (scripted(frames) - policy(frames)).abs().max().item()
        script_error = max(script_error, error)
        torch.testing.assert_close(scripted(frames), policy(frames), rtol=1e-5, atol=1e-6)
    validation = {"torchscript_max_abs_error": script_error, "cases": len(cases),
                  "batch_sizes": [1, 7], "inputs": "zeros, random windows and repeat-first windows"}
    if onnx:
        import onnx as onnx_module
        import onnxruntime as ort
        graph = io.BytesIO()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.onnx.export(scripted, cases[0], graph, dynamo=False, opset_version=17,
                              input_names=["frames"], output_names=["mean"],
                              dynamic_axes={"frames": {0: "batch"}, "mean": {0: "batch"}})
        if any(issubclass(w.category, torch.jit.TracerWarning) for w in caught):
            raise RuntimeError("ONNX exporter traced a tensor-dependent Python branch")
        graph_bytes = graph.getvalue()
        checked = onnx_module.load_model_from_string(graph_bytes)
        onnx_module.checker.check_model(checked)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(graph_bytes, sess_options=options, providers=["CPUExecutionProvider"])
        errors = []
        for frames in cases:
            expected = policy(frames).numpy()
            actual = session.run(["mean"], {"frames": frames.numpy()})[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            errors.append(float(np.max(np.abs(actual - expected))))
        validation["onnx_max_abs_error"] = max(errors)
        data[directory / "policy.onnx"] = graph_bytes
    manifest = {"format": "transformer_rl.packed_policy", "schema_version": 1,
                "model": config.to_dict()["model"], "control": config.control,
                "history": {"order": "oldest_to_newest", "reset": "repeat_first", "state": "external_window_only"},
                "output": "raw mean; issued = clip(mean, bounds); targets = offset + scale * issued",
                "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                "checkpoint_update": update, "training_metadata": metadata,
                "files": {path.name: hashlib.sha256(contents).hexdigest() for path, contents in data.items()},
                "validation": validation}
    manifest["runtime_versions"] = {"torch": str(torch.__version__), "numpy": np.__version__}
    if onnx:
        manifest["runtime_versions"].update(onnx=onnx_module.__version__, onnxruntime=ort.__version__)
    data[directory / "manifest.json"] = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    _publish_new_files(data)
    return manifest
