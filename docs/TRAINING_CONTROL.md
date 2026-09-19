# 本机与SSH训练启停

**状态：2026-09-19本轮研究收尾，`estimator`已确认停止，后续训练计划取消。**
以下是保留的工具说明，启动／恢复命令不代表当前运行安排。
最终结论见[研究结论与归档](RESEARCH_CONCLUSION.md)。

`trainctl`提供命令行和终端菜单，控制已经规划好的估计器study。
控制入口只依赖Linux系统Python；学习过程继续使用该study登记的Python、runtime和冻结源码。

## 日常使用

Kaiser安装后的简短入口为 `~/trainctl`，当前登记名称为 `estimator`：

```bash
~/trainctl status
~/trainctl status estimator --json
~/trainctl start estimator
~/trainctl pause estimator
~/trainctl resume estimator
~/trainctl logs estimator --lines 40
~/trainctl logs estimator --follow
~/trainctl check estimator
~/trainctl menu estimator
```

- `start`和`resume`都继续当前study的剩余工作。已有运行时重复调用只返回状态。
- `pause`正常发送SIGTERM并等待退出，默认最多180秒，可用`--wait`调整。若仍未退出，
  显示`stopping`并返回非零；以状态中的`active=false`及退出收据为准。
- `check`只在登记的运行时进行CPU配置/文件/检查点校验，显示剩余更新数，不创建仿真。
- `logs --follow`按当前worker切换日志；Ctrl+C退出日志查看。
- 菜单支持状态、启动/继续、暂停、日志和退出。打开菜单本身不启动训练。
- 非交互终端中不带操作运行`trainctl`时显示状态，适合SSH单次查询。

通过SSH使用相同命令：

```bash
ssh -p 2222 kaiser@192.168.64.234 '~/trainctl status estimator'
ssh -p 2222 kaiser@192.168.64.234 '~/trainctl pause estimator'
ssh -p 2222 kaiser@192.168.64.234 '~/trainctl resume estimator'
ssh -t -p 2222 kaiser@192.168.64.234 '~/trainctl menu estimator'
```

菜单需要`ssh -t`分配终端。认证由SSH处理，控制工具不保存密码。

## 注册新的study

先用 `tools/run_estimator_study.py prepare` 生成一个新study，再登记：

```bash
~/trainctl register experiment_name \
  --root /absolute/path/to/study \
  --python /absolute/path/to/training-env/bin/python \
  --runtime /absolute/path/to/runtime.sh
```

登记只保存连接到本地study所需的路径，不启动训练。同名且相同配置的登记幂等；
同名改指向其他实验会被拒绝，使用新名称即可。
当前后端支持本仓库的`study.json`分阶段格式，不能把任意其他训练脚本的目录直接当作
可恢复study。它也不改变其他项目的训练会话。

默认registry为 `~/.config/transformer-rl/trainctl.json`；Kaiser包装脚本通过
`TRAINCTL_REGISTRY`指定 `/home/kaiser/robot-rl-sim60/control/registry.json`。
也可以显式使用全局参数 `--registry PATH`。

## 暂停与恢复的含义

暂停使用正常退出而不是SIGSTOP。训练器完成当前可完成的更新，并按自身停止流程保存
模型和优化器；随后释放该worker的资源。异常退出或断电时，从最近已发布检查点恢复。

每次继续都创建新的attempt：

```text
STUDY/control/current.json
STUDY/control/attempts/GENERATION/request.json
STUDY/control/attempts/GENERATION/launch.json
STUDY/control/attempts/GENERATION/status.json
STUDY/control/attempts/GENERATION/exit.json
STUDY/control/attempts/GENERATION/STAGE/jobs/VARIANT/seed_SEED/...
```

原study的`launch.json`、`status.json`、退出收据、模型及日志继续作为原运行证据保存。
当前状态从`control/current.json`指向的attempt解析，因此日常查看以`trainctl status`为准。

恢复规则：

1. 验证冻结源码/配置、检查点配置和seed、模型SHA、报告及轨迹SHA。
2. 已完整验证的job跳过。
3. 训练未完成：选最近有效的最高累计update检查点，只训练`目标−已完成`的新增updates。
4. 训练已完成但评估中断：直接补缺失评估与导出，复用已有模型和有效报告。
5. 需要16M/32M/64M评估时，分别使用对应检查点；缺失早期检查点会报错，不用后期模型冒充。
6. 模型、PPO和估计器Adam均由冻结CLI恢复。仿真、历史窗口和随机数流重新初始化，
   因此仍是分段续训，不是逐位连续复现。

新结果包含`resume.json`、检查点来源和逐训练段的采样计数。正常停止的计数可精确合并；
异常退出且缺失最终计数时明确标为观测下界。丢弃/回退的采样不会被强行改写成理论预算。
每次attempt的wall-clock预算重新开始，累计目标updates保持不变，总耗时应跨attempt统计。

原有可行性门槛仍生效：`needs_review`不能用`start/resume`绕过进入正式长训。
改变模型、奖励或任务配置，应重新规划独立study。

## 进程与并发

- launcher身份包括PID、启动tick、boot ID和完整argv；发送信号使用pidfd。
- SDK自带Python缺少pidfd接口时，自动交由Linux系统Python执行相同身份校验。
- 启动命令锁与运行期lease分别防止重复提交和重复supervisor。
- 已发起但尚未完成启动的pause请求会保留，worker启动后先消费暂停意图。
- supervisor异常退出后，可识别指向该study输出目录的遗留训练/评估进程，阻止重复启动，
  并允许对这些已核验进程正常暂停。
- 学习源码和新控制器代码分开记录，升级控制器不覆盖正在使用的冻结学习源码。

## 本次部署时的真实训练状态

用户要求腾出资源后，估计器队列已于 **2026-09-18 03:49:28 +08:00** 停止。
四项训练均已有update80检查点：direct MLP、direct Transformer、velocity MLP、velocity Transformer。
前三项完整接线已完成；第四项的评估中断。之后恢复应从该评估继续，再处理尚未开始的任务。
控制工具的开发与部署验证使用测试进程及只读CPU检查，真实训练保持暂停。

## 部署验证

- 当前Kaiser控制器提交：`b54a76aa4f3fd9f45138804bda7a9e6761129e9d`，GitHub CI通过。
- 控制器源码：`/home/kaiser/robot-rl-sim60/train-control-20260918`；学习源码仍为原冻结快照。
- 全量CPU检查1101项通过，随后进度显示修订的启停/恢复专项17项通过。
- Kaiser原生tmux测试覆盖启动、重复启动幂等、暂停幂等和新attempt恢复，全部通过。
  该测试只运行标准库睡眠进程，未启动神经网络、仿真或CUDA。
- 实际study的`status/check/pause`及SSH终端菜单已验证；真实任务为`stopped / active=false`。

收据：[training-control-deployment.json](evidence/training-control-deployment.json)、
[estimator-study-stop.json](evidence/estimator-study-stop.json)。
