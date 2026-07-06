"""render_segments 渲染测试。

复现并锁定：正文(TEXT)夹在两组工具调用(PROCESS)之间时，正文不应被吞进代码块，
而应作为 Markdown 落在两个代码块之间（这是改造前 process_text/pending_text 双缓冲
方案的核心缺陷）。
"""
from web.render import render_segments


def _process(text):
    return {"kind": "process", "text": text}


def _text(text):
    return {"kind": "text", "text": text}


def test_process_then_answer():
    segments = [_process("\n🔧 调用工具: A\n"), _text("最终答案")]
    out = render_segments(segments)
    assert out == "```\n🔧 调用工具: A\n```\n最终答案", out


def test_interleaved_text_not_swallowed():
    """正文夹在两组工具调用之间：应渲染为 代码块 / 正文 / 代码块。"""
    segments = [
        _process("\n🔧 调用工具: A\n👉🏻 工具结果: ra\n"),
        _text("正文内容"),
        _process("\n🔧 调用工具: B\n👉🏻 工具结果: rb\n"),
    ]
    out = render_segments(segments)
    expected = (
        "```\n🔧 调用工具: A\n👉🏻 工具结果: ra\n```\n"
        "正文内容\n"
        "```\n🔧 调用工具: B\n👉🏻 工具结果: rb\n```"
    )
    assert out == expected, out


def test_consecutive_same_kind_each_rendered():
    segments = [_process("p1"), _text("t1"), _text("t2"), _process("p2")]
    out = render_segments(segments)
    assert out == "```\np1\n```\nt1\nt2\n```\np2\n```", out


def test_empty_segments():
    assert render_segments([]) == ""
    assert render_segments([], task_panel="📋 任务进度：0/1") == "📋 任务进度：0/1"


def test_whitespace_only_segments_skipped():
    assert render_segments([_process("   "), _text("\n  ")]) == ""


def test_task_panel_before_last_text():
    segments = [_process("p1"), _text("答案")]
    out = render_segments(segments, task_panel="📋 任务进度：0/1")
    assert out == "```\np1\n```\n📋 任务进度：0/1\n答案", out


def test_task_panel_at_end_when_no_text():
    out = render_segments([_process("p1")], task_panel="📋 任务进度：0/1")
    assert out == "```\np1\n```\n📋 任务进度：0/1", out


if __name__ == "__main__":
    test_process_then_answer()
    test_interleaved_text_not_swallowed()
    test_consecutive_same_kind_each_rendered()
    test_empty_segments()
    test_whitespace_only_segments_skipped()
    test_task_panel_before_last_text()
    test_task_panel_at_end_when_no_text()
    print("✅ all render_segments tests passed")
