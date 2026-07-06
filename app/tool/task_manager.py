#!/usr/bin/env python3
"""
Task Manager - 任务图（DAG）管理

为长链路复杂任务提供可持久化、跨上下文压缩可恢复的进度跟踪。

存储: memory/{session_id}/tasks.json
结构: { "tasks": {str(id): task}, "next_id": int, "finished": bool }

任务（DAG 节点）:
{
  "id": int,
  "subject": str,          # 简短标题
  "description": str,      # 详细说明 / 完成标准
  "status": "pending" | "in_progress" | "completed",
  "dependencies": [int],   # 依赖的已存在任务 ID（创建时确定，不可变）
  "notes": str,            # 进度笔记（update_task 追加，不覆盖）
  "created_at": str,
  "updated_at": str
}

依赖语义: dependencies 中的任务必须先完成，本任务才能开始。
DAG 至少有一个不依赖任何任务的"起始任务"。
依赖在创建时确定且不可变，新任务不会被已有任务依赖，结构上不可能成环。
"""
import json
import threading
from datetime import datetime
from typing import List, Dict, Optional

from app.llm.memory_manager import MEMORY_DIR
from app.log.logger import LOG

# ============ 状态常量 ============
STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"

_VALID_STATUS = {STATUS_PENDING, STATUS_IN_PROGRESS, STATUS_COMPLETED}

# 状态图标
_ICON = {
    STATUS_PENDING: "○",
    STATUS_IN_PROGRESS: "●",
    STATUS_COMPLETED: "✅",
}

# 内存缓存: session_id -> store
_task_store_cache: Dict[str, Dict] = {}
_lock = threading.Lock()


def _task_file(session_id: str):
    """获取 session 的任务存储文件路径"""
    return MEMORY_DIR / session_id / "tasks.json"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load(session_id: str) -> Dict:
    """
    加载 session 的任务存储（缓存优先）。

    Returns:
        store dict: {"tasks": {id: task}, "next_id": int, "finished": bool}
    """
    if session_id in _task_store_cache:
        return _task_store_cache[session_id]

    path = _task_file(session_id)
    store = {"tasks": {}, "next_id": 1, "finished": False}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            tasks = data.get("tasks", {})
            store["tasks"] = {int(k): v for k, v in tasks.items()}
            store["next_id"] = data.get("next_id", max(store["tasks"], default=0) + 1)
            store["finished"] = data.get("finished", False)
        except Exception as e:
            LOG.warning(f"加载任务存储失败 [{session_id}]: {e}")
    _task_store_cache[session_id] = store
    return store


def _save(session_id: str) -> None:
    """持久化任务存储到磁盘。"""
    store = _task_store_cache.get(session_id)
    if store is None:
        return
    path = _task_file(session_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "tasks": {str(k): v for k, v in store["tasks"].items()},
            "next_id": store["next_id"],
            "finished": store["finished"],
        }
        path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        LOG.exception(f"保存任务存储失败 [{session_id}]: {e}")


def _deps_status_str(deps: List[int], tasks: Dict[int, Dict]) -> str:
    """格式化依赖列表，附带各依赖任务的状态图标。"""
    parts = []
    for d in deps:
        st = tasks.get(d, {}).get("status", STATUS_PENDING)
        parts.append(f"#{d}{_ICON.get(st, '?')}")
    return ", ".join(parts) if parts else "无"


def _is_unblocked(task: Dict, tasks: Dict[int, Dict]) -> bool:
    """任务的所有依赖是否均已完成。"""
    return all(tasks.get(d, {}).get("status") == STATUS_COMPLETED for d in task.get("dependencies", []))


def has_active_tasks(session_id: str) -> bool:
    """是否存在未结束的任务计划（有任务且未 finished）。供上下文压缩逻辑判断。"""
    store = _load(session_id)
    return bool(store["tasks"]) and not store["finished"]


def get_all_tasks(session_id: str) -> List[Dict]:
    """返回按 id 排序的任务列表（供 UI 渲染）。无任务返回空列表。"""
    store = _load(session_id)
    return [store["tasks"][tid] for tid in sorted(store["tasks"])]


def format_task_snapshot(session_id: str, include_notes: bool = True) -> str:
    """
    格式化任务图快照为紧凑文本。无任务返回空字符串。

    用于 list_task 工具输出与上下文压缩时注入摘要消息。
    """
    store = _load(session_id)
    tasks: Dict[int, Dict] = store["tasks"]
    if not tasks:
        return ""

    counts = {STATUS_PENDING: 0, STATUS_IN_PROGRESS: 0, STATUS_COMPLETED: 0}
    for t in tasks.values():
        counts[t.get("status", STATUS_PENDING)] += 1

    lines = [
        f"任务图共 {len(tasks)} 个任务"
        f"（✅{counts[STATUS_COMPLETED]} / ●{counts[STATUS_IN_PROGRESS]} / ○{counts[STATUS_PENDING]}）"
    ]

    for tid in sorted(tasks):
        t = tasks[tid]
        icon = _ICON.get(t.get("status", STATUS_PENDING), "?")
        line = f"#{tid} {icon} {t.get('subject', '')}"
        if t.get("dependencies"):
            dep_str = _deps_status_str(t["dependencies"], tasks)
            line += f" (依赖: {dep_str})"
            if t.get("status") == STATUS_PENDING and not _is_unblocked(t, tasks):
                line += " ← 阻塞中"
        lines.append(line)
        if include_notes and t.get("notes"):
            for note_line in t["notes"].splitlines():
                lines.append(f"   └ {note_line}")

    # 可开始执行的任务: pending 且依赖全部完成
    startable = [tid for tid in sorted(tasks)
                 if tasks[tid].get("status") == STATUS_PENDING and _is_unblocked(tasks[tid], tasks)]
    if startable:
        lines.append(f"可开始执行: {', '.join(f'#{t}' for t in startable)}")
    else:
        in_progress = [tid for tid in sorted(tasks) if tasks[tid].get("status") == STATUS_IN_PROGRESS]
        if in_progress:
            lines.append(f"可开始执行: 无（#{in_progress[0]} 进行中）")
        elif counts[STATUS_COMPLETED] == len(tasks):
            lines.append("可开始执行: 无（全部已完成，可 finish_task 收尾）")
        else:
            lines.append("可开始执行: 无（等待依赖完成）")

    return "\n".join(lines)


def create_tasks(session_id: str, tasks: List[Dict]) -> str:
    """
    批量创建任务。dependencies 只能引用【已存在】的任务 ID。

    Args:
        session_id: 会话 ID
        tasks: 任务列表，每项 {subject(必填), description?, dependencies?}

    Returns:
        操作结果文本（含新建任务与最新任务图快照）
    """
    # 容错：单个 dict 自动包成列表
    if isinstance(tasks, dict):
        tasks = [tasks]
    if not isinstance(tasks, list) or not tasks:
        return "[错误] task_create: tasks 必须是非空数组。"

    with _lock:
        store = _load(session_id)
        existing = store["tasks"]
        created = []
        errors = []
        for idx, t in enumerate(tasks):
            if not isinstance(t, dict):
                errors.append(f"第 {idx + 1} 个任务格式错误（应为对象）")
                continue
            subject = (t.get("subject") or "").strip()
            if not subject:
                errors.append(f"第 {idx + 1} 个任务缺少 subject")
                continue
            deps = t.get("dependencies") or []
            if not isinstance(deps, list):
                errors.append(f"第 {idx + 1} 个任务 dependencies 格式错误（应为整数数组）")
                continue
            try:
                deps = [int(d) for d in deps]
            except (TypeError, ValueError):
                errors.append(f"第 {idx + 1} 个任务 dependencies 含非整数 ID")
                continue
            invalid = [d for d in deps if d not in existing]
            if invalid:
                errors.append(f"第 {idx + 1} 个任务依赖不存在的 ID: {invalid}（dependencies 只能引用已存在的任务）")
                continue
            tid = store["next_id"]
            store["next_id"] += 1
            existing[tid] = {
                "id": tid,
                "subject": subject,
                "description": (t.get("description") or "").strip(),
                "status": STATUS_PENDING,
                "dependencies": deps,
                "notes": "",
                "created_at": _now(),
                "updated_at": _now(),
            }
            created.append(tid)
        store["finished"] = False
        if created:
            _save(session_id)

    if not created:
        return "[错误] task_create 未创建任何任务:\n" + "\n".join(f"- {e}" for e in errors)

    msg = f"已创建 {len(created)} 个任务: {', '.join(f'#{t}' for t in created)}"
    if errors:
        msg += "\n部分失败:\n" + "\n".join(f"- {e}" for e in errors)
    msg += "\n\n" + format_task_snapshot(session_id)
    return msg


def list_tasks(session_id: str) -> str:
    """列出当前任务图。"""
    snap = format_task_snapshot(session_id, include_notes=True)
    return snap if snap else "(当前没有任务计划。如遇复杂任务，可用 task_create 拆解建图。)"


def update_task(
        session_id: str,
        task_id: int,
        status: Optional[str] = None,
        notes: Optional[str] = None,
        description: Optional[str] = None,
) -> str:
    """
    更新任务的状态 / 进度笔记 / 描述。notes 追加（不覆盖），以保留进度历史。

    dependencies 不可改（创建时确定），如需调整请 finish_task 后重新规划。
    """
    try:
        tid = int(task_id)
    except (TypeError, ValueError):
        return f"[错误] update_task: task_id 必须是整数，收到 {task_id!r}。"

    with _lock:
        store = _load(session_id)
        tasks = store["tasks"]
        if tid not in tasks:
            return f"[错误] update_task: 任务 #{tid} 不存在。可用 list_task 查看。"
        t = tasks[tid]
        if status is not None:
            if status not in _VALID_STATUS:
                return f"[错误] update_task: status 必须是 {sorted(_VALID_STATUS)} 之一，收到 {status!r}。"
            t["status"] = status
        if notes is not None and notes.strip():
            existing_notes = t.get("notes", "")
            t["notes"] = (existing_notes + "\n" + notes.strip()).lstrip("\n") if existing_notes else notes.strip()
        if description is not None:
            t["description"] = description.strip()
        t["updated_at"] = _now()
        _save(session_id)

    return f"已更新任务 #{tid}。\n\n" + format_task_snapshot(session_id)


def complete_task(session_id: str, task_id: int) -> str:
    """标记任务完成，并返回因此被解锁的后续待办任务。"""
    try:
        tid = int(task_id)
    except (TypeError, ValueError):
        return f"[错误] complete_task: task_id 必须是整数，收到 {task_id!r}。"

    with _lock:
        store = _load(session_id)
        tasks = store["tasks"]
        if tid not in tasks:
            return f"[错误] complete_task: 任务 #{tid} 不存在。可用 list_task 查看。"
        t = tasks[tid]
        if t["status"] == STATUS_COMPLETED:
            return f"任务 #{tid} 已是完成状态。\n\n" + format_task_snapshot(session_id)
        t["status"] = STATUS_COMPLETED
        t["updated_at"] = _now()
        # 计算被解锁的待办任务: pending、依赖本任务、且现在依赖全部完成
        unlocked = []
        for other_id, other in tasks.items():
            if other_id == tid:
                continue
            if (other.get("status") == STATUS_PENDING
                    and tid in other.get("dependencies", [])
                    and _is_unblocked(other, tasks)):
                unlocked.append(other_id)
        _save(session_id)

    msg = f"✅ 任务 #{tid} 已完成。"
    if unlocked:
        msg += f"\n🔓 已解锁后续任务: {', '.join(f'#{u}' for u in unlocked)}"
    msg += "\n\n" + format_task_snapshot(session_id)
    return msg


def finish_task(session_id: str) -> str:
    """
    结束整个任务计划：要求所有任务已完成，然后清空任务存储。

    清空后可开始新的任务计划。
    """
    with _lock:
        store = _load(session_id)
        tasks = store["tasks"]
        if not tasks:
            return "(当前没有任务计划，无需结束。)"
        incomplete = [tid for tid in sorted(tasks) if tasks[tid].get("status") != STATUS_COMPLETED]
        if incomplete:
            return (f"[错误] finish_task: 还有 {len(incomplete)} 个任务未完成: "
                    f"{', '.join(f'#{t}' for t in incomplete)}。\n"
                    f"请先 complete_task 完成它们；如需放弃请向用户说明。")
        total = len(tasks)
        # 清空存储
        store["tasks"] = {}
        store["next_id"] = 1
        store["finished"] = True
        _save(session_id)
        # 删除文件保持干净
        try:
            _task_file(session_id).unlink()
        except Exception:
            pass
        _task_store_cache.pop(session_id, None)

    return f"🎉 任务计划已结束，共完成 {total} 个任务。任务存储已清空，可开始新的任务计划。"
