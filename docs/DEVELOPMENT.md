# 仓库与版本管理

## 分支与提交

正式训练路径在 `main` 开发和维护。网络、PPO、环境、控制台和配置一起版本化；已有研究入口保留兼容，不通过删除旧实现或覆盖历史实验来切换正式路径。独立的大改动可以使用短期工作分支，完成相关检查后合入 `main`。

源码按 `formal/`、`research/`、`common/` 分开，依赖方向见[架构](ARCHITECTURE.md)。新模块放入对应职责包，顶层只保留命令入口。

一个 commit 解决一个可说明的问题，并带上直接相关的验证与文档。提交信息用 English，author/committer 为 Yukikaze。不重写已存在的共享历史，不在源码提交中混入本地输出。提交到 `main` 不等于已发布训练版本。

## 跟踪边界

| 内容 | 位置与处理 |
|---|---|
| 网络、优化器、环境与 CLI 源码 | `src/transformer_rl/`，提交 |
| 通用训练配置 | `configs/`，提交；实际运行复制为冻结快照 |
| 测试、确定性文本 fixture、工具 | `tests/`、`tools/`，提交 |
| 当前架构及使用文档 | `README.md`、`docs/NETWORK.md`、`docs/ARCHITECTURE.md`、`docs/USAGE.md`、本文件 |
| 旧研究与详细设计 | `docs/archive/`；临时收据和运行审计放忽略的 `outputs/` |
| 原始录包、数组数据、权重、导出模型、运行日志 | `datasets/`、`data/`、`runs/`、`outputs/`、`artifacts/` 等本地目录，不提交 |
| 机械资产 | 源资产与 manifest/hash 一起版本化；生成缓存不提交。引入大型二进制前单独选择外部制品存储或 Git LFS |

`.gitignore` 防止误加入未跟踪文件，不会删除本地数据，也不会让已跟踪文件自动退出版本管理。文本说明、JSON配置、CSV小投影和源 URDF/USDA 不按扩展名整体忽略。不要用 `git add -f` 把唯一 raw、checkpoint 或模型权重塞进源码仓库。

## 版本、运行与恢复

包版本在 `pyproject.toml` 与 `src/transformer_rl/__init__.py` 同步更新。当前 `0.2.0.dev0` 表示尚未发布的正式路径开发版本；变更记在根目录 `CHANGELOG.md`。经过对应验证后才建立版本 tag，不为未运行的机器人训练提前打“正式通过”标签。

运行身份由确切 commit、冻结源码/配置 SHA、资产 manifest SHA、SDK/runtime 和控制时钟共同确定。checkpoint 另外保留模型、Adam、RNG、课程和原消费预算。包版本或分支名相同不代表运行或 checkpoint 可互换；恢复以实际冻结身份和正常退出收据为准。旧 run 的源码快照不随 `main` 热更新。

## 检查位置

本机与 GitHub CI 只做 Git、AST、JSON/TOML、版本一致性等静态仓库检查：

```bash
python tools/check_repository.py
git diff --check
```

模型、unit/interface、性能和仿真检查在 Kaiser 已配置的环境执行，不在本机建 venv、下载依赖或启动模型测试。GitHub 的静态检查通过不代表 Kaiser 数值、闭链、物理或部署检查通过。新提交完成相关检查后再冻结新训练 release；训练期间保持该 release 不变。

场地派生资产可在 Kaiser 完成原资源准入并持有四个 lease 后，用 `tools/check_terrain_assets.py --manifest <prepared/manifest.json> --output <new-directory>` 检查原生碰撞。工具核对输入 SHA、组合后的顶点与物理材质，再比较逐面射线和新增前立面网格；结果必须有 `completed=true`、`passed=true`，不能只看进程 exit 0。`--disable-mesh-welding` 只在独立检查场景覆盖焊接容差，供保留基线的 A/B 排查使用，不改源资产。静态射线通过仍不等于动态接触、整车或8192环境训练验收。
