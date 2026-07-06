"""任务进度面板渲染测试。

验证 format_task_progress_panel 在无任务 / 多状态混合下的输出格式，
与 webui 嵌入助手回复时的一致性。
"""
from app.tool import task_manager
from app.tool.task_manager import (
    create_tasks,
    update_task,
    complete_task,
    format_task_progress_panel,
    STATUS_IN_PROGRESS,
)

SESSION_ID = "test_task_panel"


def _cleanup():
    """清掉测试 session 的缓存与磁盘文件，避免污染其他测试。"""
    task_manager._task_store_cache.pop(SESSION_ID, None)
    try:
        task_manager._task_file(SESSION_ID).unlink()
    except FileNotFoundError:
        pass


def test_panel_empty_when_no_tasks():
    _cleanup()
    assert format_task_progress_panel(SESSION_ID) == ""


def test_panel_renders_mixed_statuses():
    _cleanup()
    create_tasks(SESSION_ID, [
        {"subject": "分析需求", "description": "拆解用户目标"},
        {"subject": "编写实现", "dependencies": [1]},
        {"subject": "验证结果", "dependencies": [2]},
    ])
    # #1 完成，#2 进行中，#3 待开始
    complete_task(SESSION_ID, 1)
    update_task(SESSION_ID, 2, status=STATUS_IN_PROGRESS)

    panel = format_task_progress_panel(SESSION_ID)
    print(panel)
    lines = panel.splitlines()

    assert lines[0] == "📋 任务进度：1/3", lines[0]
    assert lines[1] == "  ✅ #1 分析需求", lines[1]
    assert lines[2] == "  🔄 #2 编写实现", lines[2]
    assert lines[3] == "  ⬜ #3 验证结果", lines[3]
    assert len(lines) == 4


def test_panel_all_completed():
    _cleanup()
    create_tasks(SESSION_ID, [{"subject": "A"}, {"subject": "B"}])
    complete_task(SESSION_ID, 1)
    complete_task(SESSION_ID, 2)

    panel = format_task_progress_panel(SESSION_ID)
    print(panel)
    assert panel.splitlines()[0] == "📋 任务进度：2/2"


def test_panel_cleared_after_finish():
    _cleanup()
    create_tasks(SESSION_ID, [{"subject": "A"}])
    complete_task(SESSION_ID, 1)
    task_manager.finish_task(SESSION_ID)  # 全部完成后收尾，清空存储

    assert format_task_progress_panel(SESSION_ID) == ""


if __name__ == "__main__":
    test_panel_empty_when_no_tasks()
    test_panel_renders_mixed_statuses()
    test_panel_all_completed()
    test_panel_cleared_after_finish()
    _cleanup()
    print("✅ all panel tests passed")
