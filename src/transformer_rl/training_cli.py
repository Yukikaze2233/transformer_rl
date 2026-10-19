"""A dependency-free terminal dashboard for managed packed-policy training."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import select
import shutil
import sys
import termios
import time
import tty
import unicodedata

from .training_control import TrainingSession
from .training_queue import TrainingQueue, open_session


LABELS = {"ready": "待启动", "starting": "启动中", "initializing": "初始化环境", "collecting": "采样中",
          "optimizing": "PPO 更新", "running": "训练中", "checkpointing": "保存暂停点", "paused": "已无损暂停",
          "stopping": "保存并退出中", "stopped": "已保存退出", "completed": "训练完成",
          "closing": "已保存 · 等待环境关闭", "ownership_pending": "会话锁仍被持有 · 等待确认",
          "evaluating": "控制评估中", "training": "训练中", "incomplete": "队列已结束 · 保留未完成项",
          "waiting_resources": "等待原任务资源准入",
          "failed": "训练失败", "interrupted": "中断 · 需要检查"}


def _text(value):
    return "".join(char for char in str(value) if char.isprintable() and char != "\x1b")


def _width(value):
    return sum(2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
               for char in value if not unicodedata.combining(char))


def _fit(value, width):
    value = _text(value)
    result = ""
    for char in value:
        if _width(result + char) > width:
            return result[:-1] + "…" if result else ""
        result += char
    return result


def _duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def render(state, *, width=84, color=False, message=""):
    """Render from the bounded worker projection, without parsing training logs."""
    width = max(44, min(width, 108))
    inner = width - 4
    phase = state["phase"]
    pending = state.get("request_sequence", -1) > state.get("ack_sequence", -1)
    label = LABELS.get(phase, phase)
    if pending and state.get("active"):
        intent = {"pause": "等待更新边界暂停", "resume": "等待继续确认", "stop": "等待保存退出"}
        label += " · " + intent.get(state.get("requested"), "等待确认")
    update, target = state.get("update", 0), state["target_update"]
    ratio = min(1., max(0., update / target))
    bar_width = max(10, inner - 23)
    filled = int(bar_width * ratio)
    bar = "━" * filled + "─" * (bar_width - filled)
    lines = ["TRAINING / 训练控制", "", f"{state['name']}   ·   {label}",
             f"{state['architecture']}    seed {state['seed']}    {state['device']}", "",
             f"{bar}  {ratio:6.1%}", f"更新  {update:,} / {target:,}"]
    if state.get('structured'):
        lines += [f"{state['num_envs']} 环境 · {state['history_length']} 帧含当前 · 策略 100Hz / 物理 200Hz",
                  "闭链 12 / 6 · 每 25 更新保存 · 评估由人手动发起"]
    if state.get("queue_counts"):
        counts = state["queue_counts"]
        lines[6] = f"{'评估步' if state.get('stage') == 'evaluating' else '当前更新'}  {update:,} / {target:,}"
        lines += [f"完整格 {counts['completed']}/{counts['total']} · 保留未完成 {counts['retained_incomplete']} · 未开始 {counts['not_started']}",
                  str(state.get("active_cell") or "待调度")]
        if state.get("evaluation_seed") is not None:
            lines.append(f"开发评估 seed {state['evaluation_seed']} · 四 seed × 50 场景")
    transitions = state.get("cumulative_transitions")
    if transitions is not None:
        lines.append(f"累计采样  {transitions:,} transitions")
    lines.append("活动时间  " + _duration(state.get("session_active_elapsed_s", 0))
                 + "    暂停  " + _duration(state.get("paused_elapsed_s", 0)))
    if state.get("reward_mean") is not None:
        lines.append(f"Reward  {state['reward_mean']:.5g}    LR  {state.get('learning_rate', 0):.3g}")
    if phase == "waiting_resources" and state.get("resources"):
        resources = state["resources"]
        lines += [f"可用 RAM  {resources['MemAvailable'] / 1024**3:.2f} / 所需 {resources['minimum_ram_available_bytes'] / 1024**3:.1f} GiB",
                  f"可用 VRAM {resources['gpu_free_mib'] / 1024:.2f} / 所需 {resources['minimum_gpu_free_mib'] / 1024:.1f} GiB",
                  ("达到原任务资源门槛后自动继续。" if state.get('resources_resume_automatic', True)
                   else "尚未创建训练 attempt；资源恢复后使用 start / resume。")]
    lines += ["", f"最近保存  update {state['checkpoint_update']}" if state.get("checkpoint_update") is not None else "最近保存  尚无检查点"]
    if phase == "paused":
        lines.append("状态保留在原进程内；暂停期间仍占用内存和显存。")
    elif phase == "stopped":
        lines.append("学习状态已保存；下次启动会重置物理回合与历史窗口。")
    elif phase == "completed":
        lines.append("训练预算已完成；控制性能仍需独立评估。")
    if state.get("boot_changed"):
        lines.append("主机 boot 已改变；研究队列需要重新审核接入。" if state.get("queue_counts")
                     else "主机 boot 已改变；跨 boot 启动需显式 --accept-new-boot。")
    if state.get("error"):
        lines.append("错误  " + _text(state["error"]))
    lines += ["", "p 暂停   r 继续   s 保存退出   q 离开面板", "离开面板 / SSH 断线不会停止训练。", "", str(state["root"])]
    if message:
        lines += ["", message]
    accent = "\x1b[38;5;81m" if color else ""
    reset = "\x1b[0m" if color else ""
    output = [accent + "╭" + "─" * (width - 2) + "╮" + reset]
    for index, line in enumerate(lines):
        line = _fit(line, inner)
        content = line + " " * (inner - _width(line))
        if color and index in (0, 2, 5):
            content = "\x1b[1;38;5;81m" + content + reset
        output.append(accent + "│" + reset + " " + content + " " + accent + "│" + reset)
    output.append(accent + "╰" + "─" * (width - 2) + "╯" + reset)
    return "\n".join(output)


@contextmanager
def _terminal():
    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        sys.stdout.write("\x1b[?1049h\x1b[?25l")
        sys.stdout.flush()
        yield
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)
        sys.stdout.write("\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()


def watch(session, *, plain=False):
    if not (sys.stdin.isatty() and sys.stdout.isatty()) or os.environ.get("TERM") == "dumb":
        print(render(session.status(), color=False))
        return
    message = ""
    color = not plain and "NO_COLOR" not in os.environ
    with _terminal():
        while True:
            columns, rows = shutil.get_terminal_size((84, 30))
            state = session.status()
            screen = render(state, width=columns - 1, color=color, message=message)
            if rows < len(screen.splitlines()):
                screen = "\n".join(screen.splitlines()[:max(1, rows - 1)])
            sys.stdout.write("\x1b[H\x1b[2J" + screen)
            sys.stdout.flush()
            readable, _, _ = select.select([sys.stdin], [], [], .5)
            if not readable:
                continue
            key = sys.stdin.read(1).lower()
            if key in ("q", "\x04"):
                return
            operation = {"p": "pause", "r": "resume", "s": "stop"}.get(key)
            if operation:
                try:
                    if operation == "resume" and not state["active"]:
                        session.start()
                    else:
                        session.command(operation)
                    message = "请求已提交；面板显示 worker 确认后的真实状态。"
                except (OSError, ValueError) as error:
                    message = _text(error)


def _positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return result


def _seconds(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("必须是有限非负秒数")
    return result


def _tail(path, lines):
    with Path(path).open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        stream.seek(max(0, size - 65536))
        raw = stream.read(65536)
    values = raw.decode("utf-8", errors="replace").splitlines()
    if size > 65536:
        values = values[1:]
    return "\n".join(_text(line) for line in values[-lines:])


def main(argv=None):
    parser = argparse.ArgumentParser(prog="trainctl", description="训练控制台 · 更新边界暂停 / 同进程继续 / 保存退出",
        epilog="暂停保留完整内存状态；保存退出后的恢复保留学习状态并重置回合。旧冻结 campaign 不会被自动接管。")
    parser.add_argument("--plain", action="store_true", help="禁用颜色；也遵循 NO_COLOR")
    commands = parser.add_subparsers(dest="operation")
    create = commands.add_parser("init", help="冻结配置与学习源码；不启动训练")
    create.add_argument("session", type=Path)
    create.add_argument("--config", type=Path, required=True)
    create.add_argument("--env-factory", required=True)
    create.add_argument("--python", default=sys.executable, help="已配置好运行环境的训练 Python")
    create.add_argument("--python-path", type=Path, action="append", default=[], help="环境工厂的额外模块路径")
    create.add_argument("--updates", type=_positive, required=True)
    create.add_argument("--rollout-steps", type=_positive, default=48)
    create.add_argument("--checkpoint-interval", type=_positive, default=40)
    create.add_argument("--max-seconds", type=_seconds, default=3600.)
    create.add_argument("--seed", type=int, default=0)
    create.add_argument("--device", default="cpu")
    create.add_argument("--name")
    create.add_argument("--tensorboard", action="store_true")
    adopt = commands.add_parser("adopt", help="继承已核验研究记录并登记队列；不启动训练")
    adopt.add_argument("name")
    adopt.add_argument("--plan", type=Path, required=True)
    adopt.add_argument("--root", type=Path, required=True)
    for name, help_text in (("start", "启动新会话或从正常退出点继续"), ("resume", "同进程继续；已退出时恢复学习状态"),
                            ("pause", "等待更新边界，保存后暂停原进程"), ("stop", "等待当前更新完成，保存学习状态并退出"),
                            ("status", "读取控制状态，不读取训练原始日志"), ("watch", "交互面板；q / Ctrl+C 仅离开面板"),
                            ("logs", "显示本会话 worker 最近日志"), ("jobs", "显示研究队列的全部单元")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("session", type=Path)
        command.add_argument("--json", action="store_true", help="机器可读状态")
        if name in ("start", "resume"):
            command.add_argument("--accept-new-boot", action="store_true", help="明确授权本次跨 boot 启动")
        if name == "start":
            command.add_argument("--paused", action="store_true", help="初始化后先保存并暂停，不进行 PPO 更新")
        if name in ("pause", "resume", "stop"):
            command.add_argument("--wait", type=_seconds, default=30., help="最多等待确认的秒数；超时不发送强制信号")
        if name == "logs":
            command.add_argument("--lines", type=_positive, default=30)
    listing = commands.add_parser("list", help="列出目录中的受控会话")
    listing.add_argument("directory", type=Path, nargs="?", default=Path("runs"))
    listing.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.operation is None:
        parser.print_help()
        return 0
    try:
        if args.operation == "init":
            session = TrainingSession.create(args.session, args.config, env_factory=args.env_factory,
                updates=args.updates, python=args.python, device=args.device, seed=args.seed,
                rollout_steps=args.rollout_steps, checkpoint_interval=args.checkpoint_interval,
                max_seconds=args.max_seconds, python_paths=args.python_path, name=args.name, tensorboard=args.tensorboard)
            state = session.status()
        elif args.operation == "adopt":
            state = TrainingQueue.adopt(args.root, args.plan, name=args.name).status()
        elif args.operation == "list":
            states = []
            if args.directory.is_dir():
                for child in sorted(args.directory.iterdir()):
                    if child.is_dir() and any((child / item).is_file() for item in ("session.json", "queue.json", "structured.json")):
                        states.append(open_session(child).status())
            if args.json:
                print(json.dumps(states, ensure_ascii=False, indent=2))
            else:
                print("TRAINING / 受控会话\n")
                for item in states:
                    print(f"{_text(item['name']):24}  {LABELS.get(item['phase'], item['phase'])}  "
                          f"{item.get('update', 0)}/{item['target_update']}  {_text(item['root'])}")
                if not states:
                    print("暂无会话。使用 trainctl init 创建配置快照。")
            return 0
        else:
            session = open_session(args.session)
            if args.operation == "jobs":
                if not isinstance(session, TrainingQueue):
                    raise ValueError("jobs requires a managed research queue")
                rows = session.jobs()
                if args.json:
                    print(json.dumps(rows, ensure_ascii=False, indent=2))
                else:
                    for row in rows:
                        print(f"{row['cell']:48} {row['status']}")
                return 0
            if args.operation == "watch":
                watch(session, plain=args.plain)
                return 0
            if args.operation == "logs":
                path = session.status().get("log")
                print(_tail(path, args.lines) if path else "尚未启动，无 worker 日志。")
                return 0
            if args.operation == "start":
                state = session.start(paused=args.paused, accept_new_boot=args.accept_new_boot)
            elif args.operation == "resume" and not session.status()["active"]:
                state = session.start(accept_new_boot=args.accept_new_boot)
            elif args.operation in ("pause", "resume", "stop"):
                state = session.command(args.operation, wait_seconds=args.wait)
            else:
                state = session.status()
        if getattr(args, "json", False):
            print(json.dumps(state, ensure_ascii=False, indent=2))
        else:
            color = sys.stdout.isatty() and not args.plain and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"
            print(render(state, width=shutil.get_terminal_size((84, 30)).columns - 1, color=color))
        if args.operation in ("pause", "resume", "stop") and state.get("active"):
            return 2 if args.operation == "stop" or state.get("ack_sequence", -1) < state.get("request_sequence", 0) else 0
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError) as error:
        print(f"trainctl: {_text(error)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
