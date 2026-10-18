# 训练控制台

`trainctl` 为普通 MLP、history MLP 和 Transformer 的 packed-policy 训练提供同一套终端操作。前端仅依赖 Linux Python 标准库，不需要在查看状态时导入 PyTorch 或启动仿真。

## 创建与使用

安装本包后使用 `trainctl`；从源码直接使用 `python3 tools/training.py`，两者参数相同。不带参数显示帮助，不会启动训练。

先准备完整的 `FrameTrainConfig` JSON 和环境工厂，再创建一个独立会话：

```bash
python3 tools/training.py init /absolute/path/to/new-session \
  --config /absolute/path/to/train-config.json \
  --env-factory transformer_rl.chassis_adapter:make_env \
  --python /absolute/path/to/training-env/bin/python \
  --device cuda:0 --seed 1101 \
  --updates 1200 --rollout-steps 48 --checkpoint-interval 40 \
  --max-seconds 86400
```

`init` 只复制配置与当前学习包的源码，记录 SHA256 和预算。目录必须不存在；同一会话不修改网络、seed、环境、目标更新数或时间预算。环境工厂依赖的外部 SDK、资产和模块仍使用已配置好的训练环境；额外模块搜索目录用可重复的 `--python-path PATH` 指定，它们不在学习源码快照内。

```bash
trainctl start /absolute/path/to/new-session
trainctl watch /absolute/path/to/new-session
trainctl pause /absolute/path/to/new-session --wait 120
trainctl resume /absolute/path/to/new-session
trainctl stop /absolute/path/to/new-session --wait 120
trainctl status /absolute/path/to/new-session --json
trainctl logs /absolute/path/to/new-session --lines 30
trainctl list /absolute/path/to/sessions
```

`start --paused` 先初始化环境并保存 update 0 暂停点，等用户继续后才采样。每个模型、训练 seed 和配方创建自己的会话；工具不自动并行启动其他模型，也不代替 GPU 资源准入。

面板支持 `p` 暂停、`r` 继续、`s` 保存退出、`q` 离开面板。Ctrl+C 和 SSH 断线只结束查看，训练进程独立运行，输出写入会话日志。非交互终端中的 `watch` 打印一次静态状态；`--plain`、`NO_COLOR` 和 `TERM=dumb` 可关闭颜色，JSON 始终没有终端控制字符。

## “无损”的边界

| 操作 | 模型 / Adam / RNG | 仿真回合与历史 | 资源 |
|---|---|---|---|
| `pause` → `resume` | 原进程保留 | 原对象保留；训练循环不再调用 `env.step` | 继续占用 RAM / VRAM |
| `stop` → `start` / `resume` | 校验正常退出的 checkpoint 后恢复 | 重新 reset；历史 repeat-first | 正常退出后释放 worker 资源 |
| 进程崩溃、断电、重启 | 保留已发布的文件证据；拒绝自动回退重放 | 无法保证连续 | 显示中断，需要检查 |

暂停请求可随时提交，**只有当前完整 rollout 与 PPO 更新结束、暂停 checkpoint 发布成功后，状态才显示“已无损暂停”**。同进程继续不重载 checkpoint、不重新播种、不重建 Adam 或历史。保存退出同样等待更新边界，发布 `final.pt`、sidecar 和最终收据后才确认；没有强制 kill 超时兜底。

这一保证针对训练进程和由 `env.step` 推进的同步环境。外部实物机器人、独立推进的仿真服务和墙钟驱动的设备不会因暂停训练循环而冻结。CUDA 本身的非确定性也不能由暂停协议消除。CPU 合成环境的连续性验证不等于机器人仿真或硬件资格验证。

保存退出后的恢复会校验源码、配置、checkpoint 和 sidecar 的内容哈希。目标更新数沿用原预算，只执行剩余更新；已消费的活动时间跨 attempt 累计，暂停时间不计入训练器 deadline。超时只返回“等待确认”，退出码为 2，不把仍在运行的进程说成已暂停或已退出。错误为 1，查看界面的 Ctrl+C 为 130。

主机 boot 改变后，旧进程句柄不会再被 poll。正常退出点也需要用户明确执行 `start/resume --accept-new-boot` 才能启动；没有最终退出收据的运行不会自动从较早 checkpoint 回退。这不意味着可以自动恢复任何既有研究任务。

## SSH

在训练机上准备好 runtime 后执行同一工具，不需要额外守护服务、tmux 或保存 SSH 密码：

```bash
ssh -t -p 2222 kaiser@192.168.64.234 \
  'python3 /absolute/path/to/transformer_rl/tools/training.py watch /absolute/path/to/new-session'
```

创建 worker 时继承命令所在 shell 已设置的 SDK / 动态库环境，因此启动前应像平时一样加载所需 runtime。前端的 `--python` 选择训练解释器；不会自动安装 SDK、改网络、抢占其他任务或修复 GPU 锁。

## 文件与并发

```text
SESSION/session.json                 # 固定配方与源码 / 配置 pins
SESSION/source/transformer_rl/        # 本会话学习包源码快照
SESSION/config.json
SESSION/current.json                 # 当前 attempt
SESSION/run.lock                     # worker 生命周期的继承 flock
SESSION/attempts/0001-.../
  request.json                       # 本次启动与 boot / token 绑定
  launch.json                        # 实际 PID / kernel start / boot / 完整 argv
  worker.log
  control/request.json               # 有序的 pause / resume / stop 意图
  control/status.json                # worker 确认后的有界状态投影
  control/result.json                # 最终 checkpoint 提交收据
  train/                             # 独立 run / metrics / checkpoints / completion
```

命令通过会话锁串行提交，worker 持有同一个继承 flock，阻止并发重复启动。控制状态原子写入并 fsync；状态查看只读取小投影和该会话已知进程身份，不扫描全系统进程，也不从旧 raw 日志推测进度。会话名和日志中的终端控制字符会被清理，避免破坏界面。

## 接入已有研究队列

已有实验通过经过核验的接入计划登记，原 campaign 保持只读。计划明确列出继承结果、无法恢复的中断项和真正未开始的单元，并固定原 manifest、运行环境、训练源码与资源锁身份：

```bash
trainctl adopt research --plan /absolute/path/to/adoption-plan.json \
  --root /absolute/path/to/new-managed-campaign
trainctl jobs research
trainctl start research
trainctl watch research
trainctl pause research --wait 120
trainctl resume research
trainctl stop research --wait 120
```

`adopt` 只登记，不启动。队列沿用原单元顺序、初始化哈希、seed、配置、更新数与评估场景；已完成项继承核验记录，历史 incomplete 保留原计费，缺少完整恢复点的中断项保留并阻止重跑。仅原来确实未开始的单元可以新建训练会话，保存退出后的同一会话继续执行剩余预算。

接入旧学习版本时，工具先核验原包每个文件，再仅在副本的训练和评估循环加入控制边界；策略、PPO、环境、历史与 checkpoint 实现保持原样。可逆补丁及文件哈希记录在独立证明中，训练前再次校验。原包和历史结果不热改、不重新验收。

评估的 `pause` 在 step 边界保留同一进程与环境；`resume` 接着当前窗口执行。评估期间的 `stop` 会等当前完整窗口和结果验收结束，再停止调度；不会丢弃半份结果后重跑。尚无跨进程恢复半份物理评估的保证。暂停评估仍占用资源，保存退出可能需要等待当前窗口完成。

研究队列同时持有自身 lease 和原任务的资源锁，并将全部描述符传给子进程，防止控制器意外退出后另起重复任务。面板的计数来自独立状态投影；完整格完成不代表控制资格、架构选型或部署通过。跨 boot 必须重新核验并建立新的接入计划，不能用旧 PID、游标或 `--accept-new-boot` 绕过。

旧 `tools/trainctl.py` 是估计器 study 的分段恢复工具，其 `pause` 使用正常退出，语义不同，见[旧工具说明](TRAINING_CONTROL.md)。本控制台不改历史 incomplete，不补造恢复收据；接入和恢复仍须原实验合同与实际 boot 授权。
