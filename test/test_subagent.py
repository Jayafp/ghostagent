import threading
import types

import app.tool.subagent as subagent_mod
from app.tool.tools import TOOLS, TOOL_HANDLERS


def _text_block(text):
    return types.SimpleNamespace(type="text", text=text)


def _tool_use_block(name, input_, bid="tu_1"):
    return types.SimpleNamespace(type="tool_use", id=bid, name=name, input=input_)


def test_child_tools_exclude_subagent():
    """CHILD_TOOLS 必须不含 subagent，结构上不可嵌套递归；其余工具保留。"""
    child_names = [t["name"] for t in subagent_mod.CHILD_TOOLS]
    parent_names = [t["name"] for t in TOOLS]

    assert "subagent" in parent_names          # 父 Agent 拥有 subagent
    assert "subagent" not in child_names       # 子 Agent 不拥有 subagent
    assert set(child_names) == set(parent_names) - {"subagent"}


def test_run_subagent_isolates_session_and_returns_answer(monkeypatch):
    """子 Agent 工具调用注入 sub_session_id（非父），最终返回子 Agent 文本，沙箱被清理。"""
    parent_session_id = "parent-sess-123"

    seen_bash_session_ids = []
    destroyed_ids = []

    def fake_bash(**kw):
        seen_bash_session_ids.append(kw.get("session_id"))
        return "bash-ok"

    # 两次 LLM 调用：第一次 tool_use(bash)，第二次纯文本 "DONE"
    responses = [
        types.SimpleNamespace(
            content=[_tool_use_block("bash", {"command": "echo hi"})],
            stop_reason="tool_use",
        ),
        types.SimpleNamespace(
            content=[_text_block("DONE")],
            stop_reason="end_turn",
        ),
    ]

    fake_client = _make_client(responses)

    monkeypatch.setattr(subagent_mod, "client", fake_client)
    monkeypatch.setitem(TOOL_HANDLERS, "bash", fake_bash)
    monkeypatch.setattr(subagent_mod, "destroy_sandbox", lambda sid: destroyed_ids.append(sid))

    answer = subagent_mod.run_subagent(parent_session_id, "调研 A")

    # 最终答案为子 Agent 第二次返回的文本
    assert answer == "DONE"

    # bash 被调用一次，且 session_id 是 sub_session_id（以父 id + __sub_ 前缀开头），不是父 id
    assert len(seen_bash_session_ids) == 1
    sub_sid = seen_bash_session_ids[0]
    assert sub_sid.startswith(f"{parent_session_id}__sub_")
    assert sub_sid != parent_session_id

    # 沙箱被清理一次，且用的是同一个 sub_session_id
    assert destroyed_ids == [sub_sid]


class _FakeStream:
    """模拟 anthropic MessageStream：进入上下文后 get_final_message 弹出下一条响应。

    响应若是 BaseException 则抛出（用于测试重试 / 异常路径）。
    """
    def __init__(self, pop):
        self._pop = pop

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        item = self._pop()
        if isinstance(item, BaseException):
            raise item
        return item

    def __iter__(self):
        return iter(())


def _client_from_pop(pop):
    """用自定义 pop 闭包构造假 client：每次 messages_stream 返回一个 _FakeStream，
    其 get_final_message 调 pop() 取下一条响应（异常则抛）。"""
    def fake_messages_stream(**kwargs):
        return _FakeStream(pop)
    return types.SimpleNamespace(messages_stream=fake_messages_stream)


def _make_client(responses):
    """构造一个按顺序弹出响应的假 client；响应若是异常则抛出。"""
    return _client_from_pop(lambda: responses.pop(0))


def test_destroy_safe_without_docker():
    """Docker 不可用（_client is None）时 destroy 静默返回，不抛 AttributeError（审查 #1）。"""
    from app.sandbox.config import SandboxConfig
    from app.sandbox.container_manager import ContainerManager

    mgr = ContainerManager(SandboxConfig())
    mgr._client = None
    mgr._docker_available = False
    mgr._containers = {}

    # 不应抛异常（原先会命中 self._client.containers.get -> AttributeError）
    mgr.destroy("never-existed-session")


def test_micro_compact_called_each_round(monkeypatch):
    """每轮 LLM 调用前都调用 micro_compact，防 thinking 堆积撑爆窗口（审查 #2）。"""
    compact_calls = []

    def spy(messages, session_id=None):
        compact_calls.append(len(messages))
        return messages

    monkeypatch.setattr(subagent_mod, "micro_compact", spy)
    monkeypatch.setattr(subagent_mod, "destroy_sandbox", lambda sid: None)
    monkeypatch.setitem(TOOL_HANDLERS, "bash", lambda **kw: "bash-ok")

    responses = [
        types.SimpleNamespace(content=[_tool_use_block("bash", {"command": "echo hi"})],
                              stop_reason="tool_use"),
        types.SimpleNamespace(content=[_text_block("DONE")], stop_reason="end_turn"),
    ]
    monkeypatch.setattr(subagent_mod, "client", _make_client(responses))

    answer = subagent_mod.run_subagent("parent-sess", "调研 A")

    assert answer == "DONE"
    # 两轮迭代 -> 至少两次压缩调用
    assert len(compact_calls) >= 2


def test_max_tokens_continues_loop(monkeypatch):
    """max_tokens 截断不当最终答案，续轮后取完整结论（审查 #6）。"""
    monkeypatch.setattr(subagent_mod, "destroy_sandbox", lambda sid: None)
    create_count = {"n": 0}

    def pop():
        create_count["n"] += 1
        if create_count["n"] == 1:
            return types.SimpleNamespace(content=[_text_block("被截断的不完整片段")],
                                         stop_reason="max_tokens")
        return types.SimpleNamespace(content=[_text_block("完整结论")], stop_reason="end_turn")

    monkeypatch.setattr(subagent_mod, "client", _client_from_pop(pop))

    answer = subagent_mod.run_subagent("parent-sess", "写报告")

    # 续轮后拿到完整结论，而非 max_tokens 的截断片段
    assert answer == "完整结论"
    assert create_count["n"] == 2  # 第一轮 max_tokens 续轮，第二轮 end_turn


def test_iteration_limit_returns_nonempty(monkeypatch):
    """迭代上限时取最后一条含真实文本的产出 + 提示，不返回空串（审查 #4）。"""
    monkeypatch.setattr(subagent_mod, "MAX_ITERATIONS", 2)
    monkeypatch.setattr(subagent_mod, "destroy_sandbox", lambda sid: None)
    monkeypatch.setitem(TOOL_HANDLERS, "bash", lambda **kw: "bash-ok")

    # 每轮都含真实文本 + tool_use，触发迭代上限
    responses = [
        types.SimpleNamespace(
            content=[_text_block("阶段产出1"), _tool_use_block("bash", {"command": "ls"})],
            stop_reason="tool_use"),
        types.SimpleNamespace(
            content=[_text_block("阶段产出2"), _tool_use_block("bash", {"command": "ls"})],
            stop_reason="tool_use"),
    ]
    monkeypatch.setattr(subagent_mod, "client", _make_client(responses))

    answer = subagent_mod.run_subagent("parent-sess", "长任务")

    # 取最后一条真实文本 + 超限提示，非空
    assert "阶段产出2" in answer
    assert "已达最大迭代次数" in answer
    assert answer != ""


def test_long_answer_truncated_with_marker(monkeypatch):
    """超长答案截断到 20000 字符并追加截断标记（审查 #8）。"""
    monkeypatch.setattr(subagent_mod, "destroy_sandbox", lambda sid: None)
    long_text = "X" * 30000
    responses = [types.SimpleNamespace(content=[_text_block(long_text)], stop_reason="end_turn")]
    monkeypatch.setattr(subagent_mod, "client", _make_client(responses))

    answer = subagent_mod.run_subagent("parent-sess", "输出超长内容")

    # 前 20000 字符保留，且追加了截断标记（含原长度信息）
    assert answer[:20000] == "X" * 20000
    assert "已截断至 20000" in answer
    assert "30000" in answer  # 原长度


def test_retry_on_rate_limit(monkeypatch):
    """429 限流按与父循环一致的次数重试，重试耗尽前不判失败（审查 #5）。"""
    monkeypatch.setattr(subagent_mod, "_LLM_RETRY_SLEEP", 0)  # 测试中不真睡

    class _RateLimit(Exception):
        pass

    class _Timeout(Exception):
        pass

    # 替换 anthropic，使 except 子句匹配假异常类
    monkeypatch.setattr(subagent_mod, "anthropic",
                        types.SimpleNamespace(RateLimitError=_RateLimit, APITimeoutError=_Timeout))
    monkeypatch.setattr(subagent_mod, "destroy_sandbox", lambda sid: None)

    create_count = {"n": 0}

    def pop():
        create_count["n"] += 1
        if create_count["n"] <= 2:
            raise _RateLimit("429")
        return types.SimpleNamespace(content=[_text_block("DONE")], stop_reason="end_turn")

    monkeypatch.setattr(subagent_mod, "client", _client_from_pop(pop))

    answer = subagent_mod.run_subagent("parent-sess", "重试任务")

    # 两次限流后第三次成功
    assert answer == "DONE"
    assert create_count["n"] == 3
