# 外部行为锚点的采样上限

`training.max_anchors` 是**每个锚点文件的采样上限 K**，不是整个 retention 目标的全局记忆容量。它不改变策略历史长度 L，也不改变 PPO rollout 长度。独立锚点评估按 case 和 anchor seed 生成文件；相同 case 的不同 seed 仍是不同文件。

新 study 可以在 `training` 中显式声明正整数 `max_anchors`，例如 256 或 512。原有 `rollout_steps`、`checkpoint_interval`、`max_seconds`、`retention_coef` 和 `anchor_seeds` 字段仍须提供。省略此字段时，study 与单场/suite CLI 继续采用 256；旧 spec 不会自动插入新字段，所以原 canonical JSON 不变。直接调用底层 Python `evaluate_frame_policy` 时，原有默认 2048 保留；需要比较时应显式传参。

`evaluate --max-anchors K --anchor-output FILE` 与 `evaluate-suite --max-anchors K --anchor-directory DIRECTORY` 使用同一采样上限。suite 对每个 case 单独收集，容量不会在 case 之间分摊。study 仅在启用 retention 的独立锚点评估中传递显式 K，普通验证与最终测试不收集锚点。

新锚点评估报告的 `anchors` 同时记录：

- `max_samples`：请求的每文件上限 K。
- `samples`：实际保存的端点数量，可能低于 K。
- `path`、`sha256`：包含完整历史帧与 teacher 分布的实际文件身份。

例如两个 case、一个 anchor seed、每文件恰好收满时，K=256 会产生两个文件，实际总端点数 N=512；K=512 时 N=1024。评估太短或候选 case 未通过时，实际文件数与 N 可能更小。应根据被后续训练采用的文件逐一核验报告与 SHA，并将各文件 `samples` 相加；不能把 spec 中的 K 直接当成全局 N。

`AnchorRegularizer` 会读取各文件的全部端点，不会将 512 个端点裁成 256。其默认 `batch_size=256` 是每次 KL 计算的抽样数量：先均匀抽一个文件，再在该文件内有放回抽端点。因此 KL batch 大小、每文件采样上限 K、实际文件数与总端点数 N 是四个不同量；文件大小不等时，全体端点也不是均匀抽样。

`retention_coef=0` 时，study 不生成锚点容量的实际收集记录。即使配置中声明了 K，它也只是未使用的请求参数。启用系数但第一阶段尚无外部文件时，实际 retention 同样尚未生效。单独运行锚点评估仅产生可供后续使用的数据，也不证明某个训练阶段已经使用了它。

后续 K 对比应固定 case 集合、anchor seeds、通过条件、文件数、teacher checkpoint、评估步数及采样协议，并报告 K、各文件实际样本数与总 N。学习分析工具 `analyze_frame_learning.py` 目前仍将 `anchors_capacity` 保留为未知；本次补齐的是声明、实际收集与证据核验路径，不从标签推断已经生效的容量。旧 checkpoint 的 anchors identity 与 resume 比较规则保持原样。
