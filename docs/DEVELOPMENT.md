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

场地派生资产可在 Kaiser 完成原资源准入并持有四个 lease 后，用 `tools/check_terrain_assets.py --manifest <prepared/manifest.json> --output <new-directory>` 检查原生碰撞。工具核对输入 SHA、组合后的顶点与物理材质，再比较逐面射线和新增前立面网格；结果必须有 `completed=true`、`passed=true`，不能只看进程 exit 0。同时记录命中距离换算点误差和返回坐标偏离射线的误差，区分几何差异与字段数值问题；诊断不改变通过阈值。`--disable-mesh-welding` 只在独立检查场景覆盖焊接容差，供保留基线的 A/B 排查使用，不改源资产。静态射线通过仍不等于动态接触、整车或8192环境训练验收。

碰撞预处理对每个三角形只循环重排顶点起始顺序，选取入射边长乘积最小的角作为求交基底，避免狭长面使用近乎平行的两条长边。坐标、绕向、面数与逐面来源保持不变；`collision.npz` 的 `cyclic_corner_offset` 记录可逆偏移。该处理不删除有效薄面、不填洞、不改变焊接容差；新派生版本仍须独立通过原生查询检查。

atlas 平移后的 float32 坐标另做一次精确零面积检查，仅移除编码后叉积确为零的面；不按面积或角度阈值删除有效薄面。manifest 的 `float32_face_cleanup` 记录所有被移除面的 tile、part 与来源三角形；NPZ 的 `face_source_part`、`face_source_triangle` 和 `face_pre_cleanup_index` 保留逐面映射，真实场地与人工 padding 分开追溯。

atlas 的 `material_parameters` 按 `[静摩擦, 动摩擦, 恢复系数]` 保存每个物理材质，真实场地继承源 manifest，参数化地形明确采用静／动摩擦相等、恢复系数为零。旧 `material_coefficients` 只保留静摩擦查询语义；不能据此重建完整材质，也不能把轮下地表摩擦冒充 Critic 的轮体材料。派生时另建输出目录，保留原资产和旧 atlas 的身份；未实测标定的摩擦仍不称为标定值。

大批量共享地形使用 `lab_env.isolation`：当前场景的单一、连续 env_0…env_N 克隆计划启用 PhysX Env ID，在 broadphase 早期隔离机器人；USD 碰撞组另保留全局地面可接触关系并逐组回读。异构或部分克隆计划明确拒绝，不修改安装的 SDK，也不把 `env_spacing` 当作碰撞过滤开关。回调须保留到原生初始化结束：attach 与 force-load 可能分别触发复制，第一次 attach 后即注销会使后续加载丢失隔离；初始化完成后再统一释放。回调安装、实际复制次数及注销状态分开记录，managed `lab_setup.json` 保存完成收据。共用地形原点不等于碰撞隔离失效，原生容量错误仍须判失败；8192×48 验收不能由该配置或 CPU 检查代替。

Critic 的六维轮体材料由 `WheelMaterialReader` 读取 PhysX 实际 shape 属性，按左／右轮各自的静摩擦、动摩擦、恢复系数排列。当前无材质随机事件，初始化时读取并缓存；未来若显式改写物理材质，必须随后按受影响行 `refresh`。一轮多个碰撞 shape 的材质不一致时明确拒绝，不平均、不用地表摩擦替代；managed 的 `lab_setup.json` 保留 shape 映射来源。

Lab 快速采样的受控入口使用 `transformer_rl.formal.lab_env.runtime:runtime`，由现有 `StructuredTrainingSession` 管理进程、资源准入和完整学习 checkpoint；不把诊断入口的 `lab_ppo_v1` 当成可直接转换的正式端点。`managed_lab_runtime_v2` 私有配置冻结当前 boot、机械资产目录、响应记录与 atlas 的文件 pin、环境间距、完整 `FlatTaskSpec`、显式 `NativeTaskLimits`、`PairTargetDomain`、五项 PhysX 容量和诊断开关。旧 v1 不隐式升级。 静态 course 的接触载荷来自原生 contact patch：各体正法向标量相加，轮支撑只累计向上分量；水平立面碰撞不算落地。reset 行在下一个真实物理步前的接触标签标为无效，PRE-reset 标签保留自己的有效性快照。`contact_slip` 仍显式缺失，不能用轮心速度代替。M1 简化失败阈值不能作为正式训练的终止合同；正式路径复用逐物理步失败锁存，包括接触确认时间、被动膝越界、闭链残差与气弹簧行程，BODY_BUMP 仅豁免允许的机身接触。课程与 learner 共用 sampler、预算和请求去重状态；地形路由、任务独立 RNG 和 reward 缺失原因存入完整 checkpoint 的 task source。冷恢复在新回合采样前恢复这些状态，再由 worker 在物理准备后恢复全局 RNG；活动物理请求不重放。低频 reward 投影只描述最后一个完整策略区间，不称为整轮汇总，也不触发评价或课程晋级。此入口仍须完成原生场地、全规模更新与实际控制边界验收后，才能用于正式长训。

阶段覆盖诊断的日志和 checkpoint 保留 `validation_only`、`diagnostic_stage` 与 `budget_scope`；预填的阶段偏移不代表实际 PPO 消费，不能作为学习 resume 的来源。旧未标记的 Lab 端点仍须核对课程累计消费与实际 update，不能用诊断偏移扩充预算。warm start 与 resume 必须二选一；完整正式恢复继续使用 managed checkpoint 与 journal。

台阶请求的消费水位和外部请求统计纳入私有 `managed_lab_task_source_v2`：冷恢复保留去重 ID、课程 RNG 与累计统计，终止旧物理请求，不重放台阶动作。旧 v1 快照明确拒绝，不隐式升级。部署台阶 FSM 只消费前一 tick 的有效估计和外部命令；命令高度不作为升高、跌落或成功证据，取消和超时须在同一边界解除模式。

新监督进程在 exec 后两次核对实际 boot、start 和完整 argv，再保存启动身份。父进程退出后，即使收据身份不一致也不再轮询该 PID；界面明确报错并拒绝恢复。不用后来的 status 修补旧收据或补造启动身份。

直接 Lab 诊断与 managed 路径复用 `NativePhysicsHealth`：绑定当前进程继承的独占 stdout/stderr 文件，每轮采样返回后、PPO 更新前只读取新追加部分，不从仿真线程同步回调 Python。managed worker 已由父进程重定向；直接诊断需把两路输出重定向到该验证目录的文件。容量溢出或漏交互区间不能变成有效学习结果；这项检查不在逐 step 路径重做所有权或协议校验。
