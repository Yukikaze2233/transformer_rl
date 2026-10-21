"""资产与训练设置；不导入 Isaac，可在 CPU 上校验。"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path

from ..control.observation import MechanicalBinding
from .contract import FlatTaskSpec, HISTORY_LENGTH


@dataclass(frozen=True)
class AssetDescription:
    """选中的 USD/URDF/模型规范；只读，不创建场景。"""

    directory: Path
    manifest: dict
    usd_path: Path
    urdf_path: Path
    binding: MechanicalBinding
    nominal_height_m: float
    manifest_sha256: str

    def _source_joints(self):
        path = self.directory / "model_spec.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("joint velocity limits require the selected regular model source")
        with path.open("rb") as source:
            raw = source.read(256 * 1024 + 1)
        expected = self.manifest.get("files_sha256", {}).get("model_spec.json")
        if len(raw) > 256 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("joint velocity limits differ from the selected model source pin")
        joints = json.loads(raw).get("joints")
        names = self.manifest["tree_joint_names"]
        if (not isinstance(joints, list) or len(joints) != len(names)
                or any(not isinstance(joint, dict) or not isinstance(joint.get("name"), str)
                       or joint.get('type') not in ('continuous', 'revolute', 'prismatic')
                       for joint in joints)
                or len({joint.get("name") for joint in joints}) != len(names)
                or {joint.get("name") for joint in joints} != set(names)):
            raise ValueError("joint velocity limits require every selected named tree joint")
        return joints

    def continuous_joint_names(self) -> tuple[str, ...]:
        """Identify periodic coordinates from the selected mechanical source."""
        continuous = {joint['name'] for joint in self._source_joints()
                      if joint['type'] == 'continuous'}
        return tuple(name for name in self.manifest['tree_joint_names'] if name in continuous)

    def joint_velocity_limits(self) -> dict[str, float]:
        """Read source-defined solver limits in rad/s or m/s, not hardware certification."""
        limits = {}
        for joint in self._source_joints():
            limit = joint.get("limit")
            value = limit.get("velocity") if isinstance(limit, dict) else None
            if (joint.get("type") not in ("continuous", "revolute", "prismatic")
                    or type(value) not in (int, float, str)):
                raise ValueError("joint velocity limits require explicit positive native-unit speeds")
            # URDF-derived model sources retain numeric XML attributes as strings.
            try:
                value = float(value)
            except (ValueError, OverflowError) as error:
                raise ValueError("joint velocity limit is not a finite source number") from error
            if not math.isfinite(value) or value <= 0:
                raise ValueError("joint velocity limits require explicit positive native-unit speeds")
            limits[joint["name"]] = value
        return {name: limits[name] for name in self.manifest['tree_joint_names']}


@dataclass(frozen=True)
class LabEnvSettings:
    """DirectRLEnv 适配层需要的显式设置。"""

    asset_directory: str
    num_envs: int = 8192
    env_spacing_m: float = 4.0
    sim_dt_s: float = 0.005
    decimation: int = 2
    episode_length_s: float = 30.0
    device: str = "cuda:0"
    history_length: int = HISTORY_LENGTH
    task: FlatTaskSpec = field(default_factory=FlatTaskSpec)

    def __post_init__(self):
        if type(self.num_envs) is not int or self.num_envs < 1:
            raise ValueError("num_envs 必须是正整数")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for v in (self.env_spacing_m, self.sim_dt_s, self.episode_length_s)):
            raise ValueError("环境间距/物理步长/回合长度必须是有限正数")
        if self.sim_dt_s not in (0.001, 0.005):
            raise ValueError("sim_dt_s 只支持 1kHz 或 200Hz")
        if type(self.decimation) is not int or self.decimation < 1:
            raise ValueError("decimation 必须是正整数")
        if abs(self.decimation * self.sim_dt_s - 0.01) > 1e-12:
            raise ValueError("decimation*sim_dt_s 必须等于 100Hz 策略周期")
        if self.history_length != HISTORY_LENGTH:
            raise ValueError("M1 契约固定 5 帧历史")
        if not isinstance(self.task, FlatTaskSpec):
            raise TypeError("task 必须是 FlatTaskSpec")


def load_asset_description(asset_directory, *, expected_manifest_sha256=None) -> AssetDescription:
    """读取并校验选中资产；不修改资产目录。"""
    root = Path(asset_directory).resolve(strict=True)
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("asset manifest 必须是普通文件")
    raw = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise ValueError("asset manifest hash 与预期不一致")
    manifest = json.loads(raw)
    usd, urdf = manifest.get("usd"), manifest.get("urdf")
    if not isinstance(usd, str) or not isinstance(urdf, str):
        raise ValueError("manifest 必须显式给出 usd 与 urdf")
    usd_path, urdf_path = root / usd, root / urdf
    for path in (usd_path, urdf_path):
        if (path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(root)):
            raise ValueError("资产文件必须位于 asset 目录内")
    binding = MechanicalBinding.from_manifest(manifest, manifest["tree_joint_names"],
                                              manifest["rigid_body_names"])
    height = manifest.get("nominal_base_height_m")
    if type(height) not in (int, float) or not math.isfinite(height) or height <= 0:
        raise ValueError("nominal_base_height_m 必须是有限正数")
    return AssetDescription(root, manifest, usd_path, urdf_path, binding,
                            float(height), manifest_sha)
