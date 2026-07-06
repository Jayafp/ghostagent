import json

from web.history_message import format_history_from_memory, _blocks_to_segments
from web.render import render_segments


def test_format_history_from_memory():
    session_id = 'test'

    print(json.dumps(format_history_from_memory(session_id), indent=2, ensure_ascii=False))


def _outside_codeblock(out, needle):
    """断言 needle 落在 ``` 代码块之外（split 后偶数下标为非代码文本段）。"""
    parts = out.split("```")
    idx = next((i for i, p in enumerate(parts) if needle in p), -1)
    assert idx != -1, f"未找到 {needle!r}:\n{out}"
    assert idx % 2 == 0, f"{needle!r} 被困在代码块内 (part[{idx}]):\n{out}"


def test_history_text_not_trapped_in_codeblock():
    """回归：历史回显中，夹在工具调用之间的正文不应被困在代码块里。

    复现 jinrong 场景：assistant(正文+工具) → tool_result → assistant(报告+收尾工具)
    → tool_result → assistant(总结)。改造前报告会被并进未关闭的 ``` 代码块。
    """
    segments = []
    # 助手：正文 + 工具调用
    _blocks_to_segments([
        {"type": "text", "text": "我来分析一下。"},
        {"type": "tool_use", "input": {"q": "test"}, "name": "search"},
    ], segments)
    # tool_result
    _blocks_to_segments([
        {"type": "tool_result", "content": "结果A"},
    ], segments)
    # 助手：完整报告 + 收尾工具
    _blocks_to_segments([
        {"type": "text", "text": "# 报告\n这是完整报告。"},
        {"type": "tool_use", "input": {}, "name": "complete_task"},
    ], segments)
    _blocks_to_segments([
        {"type": "tool_result", "content": "完成"},
    ], segments)
    # 最终总结
    _blocks_to_segments([
        {"type": "text", "text": "总结：OK。"},
    ], segments)

    out = render_segments(segments)
    print(out)

    # 三段正文都应落在代码块之外
    _outside_codeblock(out, "我来分析一下。")
    _outside_codeblock(out, "# 报告")
    _outside_codeblock(out, "总结：OK。")
    # 工具过程仍应在代码块内
    assert "🔧 调用工具:" in out
    assert "👉🏻 工具结果:结果A" in out


def test_history_thinking_not_rendered():
    segments = []
    _blocks_to_segments([
        {"type": "thinking", "thinking": "内部思考不应回显"},
        {"type": "text", "text": "正文"},
    ], segments)
    out = render_segments(segments)
    assert "内部思考不应回显" not in out
    assert "正文" in out


if __name__ == '__main__':
    test_history_text_not_trapped_in_codeblock()
    test_history_thinking_not_rendered()
    test_format_history_from_memory()
    print("✅ all history_message tests passed")
